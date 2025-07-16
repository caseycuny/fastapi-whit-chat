from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
import os, time
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any, Tuple, Union
from fastapi.responses import JSONResponse
import logging
import httpx
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import json
from pydantic import ValidationError
import asyncio
from dotenv import load_dotenv
from .schemas import (
    ElaborationFeedbackResponse,
    ElaborationSummaryResponse,
    GenerateArgumentParagraphResponse,
    ElaborationModelSentencesResponse,
    ProcessDebateAnalysisResponse,
    GenerateDebateInsightsRequest,
    CachedDebateInsightsRequest,
    DebateInsightsResponse,
    GenerateDebateInsightsResponse,
    CachedDebateInsightsResponse,
    ExitTicketAnalysisRequest,
    ExitTicketAnalysisResponse,
    ExitTicketClasswideRequest,
    ExitTicketClasswideResponse,
    ExitTicketClasswideAnalysisData,
    MixedGroupsRequest,
    MixedGroupsResponse
)
from .utils import extract_json_from_response
from asgiref.sync import sync_to_async
import openai
import traceback
from datetime import datetime
from pprint import pprint
from sqlalchemy.orm import Session
from .db import SessionLocal, Base, engine
from sqlalchemy.orm import selectinload
from .models import Submission, FeedbackCategory, NextInstructionalFocus, InstructionalBlindSpot, WritingPersona, CustomUser, CognitiveSkill, Thread, ExitTicketClasswideAnalysis
import random, re
from collections import defaultdict
from sqlalchemy import text
from starlette.websockets import WebSocketState
import anyio


load_dotenv()

print("DEBUG: FastAPI DATABASE_URL =", os.getenv("DATABASE_URL"))

app = FastAPI()

# Load allowed frontend origins from .env or default
frontend_origins = os.getenv("FRONTEND_ORIGINS", "").split(",")

# Add dev URLs for local testing if not already included
dev_origins = [
    "http://localhost:8000", 
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# Add common production URLs
prod_origins = [
    "https://jarvis-ai-tutor.onrender.com",
    "https://jarvis-ai-tutor-production.up.railway.app",
    "https://jarvis-ai-tutor.herokuapp.com"
]

# Combine all origins
all_origins = frontend_origins + dev_origins + prod_origins

# Clean up any empty strings and remove duplicates
frontend_origins = list(set([origin.strip() for origin in all_origins if origin.strip()]))

print(f"DEBUG: CORS allowed origins: {frontend_origins}")

# Apply middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.5,
)



@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"Unexpected error: {str(exc)}")
    print(f"Stack trace:\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

# Pydantic schemas
class ChatInput(BaseModel):
    message: str
    thread_id: str
    assistant_id: str
    assignment_id: Optional[int] = None

class InitChatInput(BaseModel):
    assignment_id: Optional[int] = None
    assistant_id: str
    message: Optional[str] = None

class TrendData(BaseModel):
    rubric_averages: Dict[str, Any]
    common_strengths: List[str]
    instructional_blind_spots: List[str]
    tone_analysis_trends: Dict[str, Any]
    sentence_structure: Dict[str, Any]
    vocabulary_strength_patterns: Dict[str, Any]
    notable_style_structure_patterns: List[str]
    common_weaknesses_full: Dict[str, Any]

class DebatePromptInput(BaseModel):
    topic: str

class DebatePromptResponse(BaseModel):
    prompts: List[Dict[str, str]]

class DebateInitInput(BaseModel):
    prompt: str
    stance: str

class DebateInitResponse(BaseModel):
    thread_id: str
    run_id: str
    status: str
    assistant_message: str

class DebateChatInput(BaseModel):
    message: str
    thread_id: str

class DebateChatResponse(BaseModel):
    run_id: str
    status: str
    assistant_message: str

class InitializeRequest(BaseModel):
    topic: str
    class_id: int
    user_id: int  # Pass from Django

class FeedbackRequest(BaseModel):
    thread_id: str
    user_message: str
    full_paragraph: str
    user_id: int

class SaveConversationRequest(BaseModel):
    topic: str
    conversation: str
    class_id: int
    thread_id: str
    user_id: int
    assignment_id: Optional[int] = None

class ElaborationInitRequest(BaseModel):
    topic: str
    user_id: int
    assignment_id: Optional[int] = None

class ElaborationFeedbackRequest(BaseModel):
    thread_id: str
    user_message: str
    full_paragraph: str
    user_id: int

class ElaborationSummaryRequest(BaseModel):
    topic: str
    conversation: str
    user_id: int
    assignment_id: Optional[int] = None

class ElaborationModelSentencesRequest(BaseModel):
    topic: str
    user_id: int
    assignment_id: Optional[int] = None

# Add new Pydantic models for rhetorical device practice
class RhetoricalDeviceInitRequest(BaseModel):
    last_debate_topic: str
    student_id: int
    assignment_id: Optional[int] = None

class RhetoricalDeviceProgressRequest(BaseModel):
    thread_id: str
    student_id: int
    assignment_id: Optional[int] = None

class DevicePracticed(BaseModel):
    device_name: str
    occurrences: int
    effectiveness_comment: str

class RhetoricalDeviceProgressResponse(BaseModel):
    success: bool
    devices_practiced: List[DevicePracticed]
    overall_feedback: str
    suggested_focus: List[str]
    message: Optional[str] = None

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DJANGO_API_BASE = os.getenv("DJANGO_API_BASE", "http://localhost:8000")

def load_rubric(rubric_type: str) -> dict:
    """Load a specific rubric from the Rubrics.json file"""
    try:
        # Get the path to the Rubrics.json file in the parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rubrics_path = os.path.join(current_dir, "..", "Rubrics.json")
        
        with open(rubrics_path, 'r') as f:
            rubrics_data = json.load(f)
        
        if rubric_type in rubrics_data["function"]:
            return rubrics_data["function"][rubric_type]
        else:
            logger.error(f"Rubric type '{rubric_type}' not found in Rubrics.json")
            return {}
    except Exception as e:
        logger.error(f"Error loading rubric '{rubric_type}': {str(e)}")
        return {}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    # Just a test endpoint to check DB connection
    result = db.execute("SELECT 1").fetchone()
    return {"result": result[0]}

@app.post("/chat")
async def continue_chat(input: ChatInput):
    try:
        logger.info(f"Continuing chat with input: {input}")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

        teacher_assistant_id = os.getenv("TEACHER_CHAT_BUDDY_ID")
        if not teacher_assistant_id:
            logger.error("TEACHER_CHAT_BUDDY_ID not set")
            raise HTTPException(status_code=500, detail="TEACHER_CHAT_BUDDY_ID not set")

        # Create message
        await client.beta.threads.messages.create(
            thread_id=input.thread_id,
            role="user",
            content=input.message
        )

        # Create and monitor run with the teacher assistant
        run = await client.beta.threads.runs.create(
            thread_id=input.thread_id,
            assistant_id=teacher_assistant_id
        )

        timeout = 60
        start_time = time.time()
        while True:
            status = (await client.beta.threads.runs.retrieve(run.id, thread_id=input.thread_id)).status
            if status == "completed":
                break
            elif status in ["failed", "cancelled"]:
                raise HTTPException(status_code=400, detail=f"Run status: {status}")
            elif time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="Assistant timed out.")
            await asyncio.sleep(1)

        # Get messages
        messages = await client.beta.threads.messages.list(thread_id=input.thread_id, order="asc")
        for msg in messages.data:
            print(f"{msg.role}: {msg.content[0].text.value}")
        return {
            "thread_id": input.thread_id,
            "history": [
                {
                    "role": msg.role,
                    "text": msg.content[0].text.value
                }
                for msg in messages.data if msg.role in ["user", "assistant"]
            ]
        }
    except Exception as e:
        logger.error(f"Error in continue_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_submission_focus_data(assignment_id: int, db: Session) -> dict:
    t0 = time.time()
    logger.info(f"Fetching submission focus data for assignment {assignment_id}")
    print(f"DEBUG: Fetching submission focus data for assignment {assignment_id}")
    submissions = db.query(Submission).filter(Submission.assignment_id == assignment_id).options(
        selectinload(Submission.student)
    ).all()
    logger.info(f"Found {len(submissions)} submissions for assignment {assignment_id}")
    print(f"DEBUG: Found {len(submissions)} submissions for assignment {assignment_id}")
    data = []
    for sub in submissions:
        # Blind spots
        blindspots = [row.blind_spot for row in db.query(InstructionalBlindSpot).filter_by(submission_id=sub.id)]
        # Next steps
        next_steps = [row.focus for row in db.query(NextInstructionalFocus).filter_by(submission_id=sub.id)]
        # Feedback categories
        areas_qs = db.query(FeedbackCategory).filter_by(submission_id=sub.id).all()
        areas = []
        strengths = []
        rubric_scores = {}
        for cat in areas_qs:
            # Areas for improvement
            val = cat.areas_for_improvement
            if val:
                if not isinstance(val, list):
                    try:
                        val = eval(val) if isinstance(val, str) else [str(val)]
                    except Exception:
                        val = [str(val)]
                if isinstance(val, list):
                    areas.extend(val)
            # Strengths
            sval = cat.strengths
            if sval:
                if not isinstance(sval, list):
                    try:
                        sval = eval(sval) if isinstance(sval, str) else [str(sval)]
                    except Exception:
                        sval = [str(sval)]
                if isinstance(sval, list):
                    strengths.extend(sval)
            # Rubric scores
            rubric_scores[cat.name] = cat.score
        # Writing persona
        persona = None
        try:
            persona_obj = db.query(WritingPersona).filter_by(submission_id=sub.id).first()
            if persona_obj:
                persona = persona_obj.type
        except Exception:
            persona = None
        # Pick two random items if possible, else the whole list
        def pick_two_random(lst):
            return random.sample(lst, 2) if len(lst) > 2 else lst
        data.append({
            "student": {
                "first_name": sub.student.first_name if sub.student else None,
                "last_name": sub.student.last_name if sub.student else None,
            },
            "rubric_scores": rubric_scores,
            "areas_for_improvement": pick_two_random(areas),
            "strengths": pick_two_random(strengths),
            "next_instructional_focus": next_steps,
            "instructional_blind_spots": blindspots,
            "writing_persona": persona
        })
    logger.info(f"submission_focus raw data: {data}")
    print(f"DEBUG: submission_focus raw data: {data}")
    # Summarize instructional focus
    def summarize_instructional_focus(student_data):
        focus_keywords = {
            "transitions": ["transition", "flow", "cohesion"],
            "thesis": ["thesis", "claim", "main idea"],
            "evidence": ["evidence", "support", "example", "cite", "citation"],
            "elaboration": ["elaboration", "explain", "analysis", "commentary", "develop"],
            "conciseness": ["concise", "clarity", "wordy", "clarify", "repetitive"],
            "grammar": ["grammar", "comma", "syntax", "fragment", "run-on", "mechanics"],
            "structure": ["structure", "organization", "outline", "paragraphing"],
            "style": ["tone", "style", "voice", "diction", "formal"],
        }
        counts = defaultdict(int)
        for student in student_data:
            combined_text = " ".join(student.get("next_instructional_focus", []) + student.get("instructional_blind_spots", []))
            combined_text = combined_text.lower()
            for tag, keywords in focus_keywords.items():
                if any(re.search(rf"\\b{kw}\\b", combined_text) for kw in keywords):
                    counts[tag] += 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    metrics_summary = summarize_instructional_focus(data)
    logger.info(f"metrics_summary: {metrics_summary}")
    print(f"DEBUG: metrics_summary: {metrics_summary}")
    logger.info(f"Fetched submission focus data in {time.time() - t0:.2f}s")
    print(f"DEBUG: Fetched submission focus data in {time.time() - t0:.2f}s")
    return {
        "assignment_id": assignment_id,
        "metrics_summary": metrics_summary,
        "students": data
    }

@app.post("/initialize_chat")
async def initialize_chat(input: InitChatInput, db: Session = Depends(get_db)):
    try:
        logger.info(f"Initializing chat with input: {input}")
        total_start = time.time()
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        teacher_assistant_id = os.getenv("TEACHER_CHAT_BUDDY_ID")
        if not teacher_assistant_id:
            logger.error("TEACHER_CHAT_BUDDY_ID not set")
            raise HTTPException(status_code=500, detail="TEACHER_CHAT_BUDDY_ID not set")
        # Create thread
        t0 = time.time()
        thread = await client.beta.threads.create()
        thread_id = thread.id
        logger.info(f"Created thread with ID: {thread_id} in {time.time() - t0:.2f}s")
        # Fetch submission feedback from DB
        t1 = time.time()
        if input.assignment_id:
            submission_data = await get_submission_focus_data(input.assignment_id, db)
            logger.info(f"Fetched submission feedback in {time.time() - t1:.2f}s")
            if submission_data:
                context_text = (
                    f"Here is the metrics summary for the assignment:\n{submission_data['metrics_summary']}\n\n"
                    f"Here is the submission feedback data for all students:\n{submission_data['students']}"
                )
            else:
                context_text = "No submission feedback available."
        else:
            context_text = "No assignment context available."
        # Combine context and user's question into a single user message
        if input.message:
            combined_message = f"Using this submission feedback data, collaborate as a thought partner and assistant with the teacher.\n{context_text}\n\n{input.message}"
        else:
            combined_message = f"Using this submission feedback data, collaborate as a thought partner and assistant with the teacher.\n{context_text}"
        # Add the combined message as the first user message
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=combined_message
        )
        logger.info("Added combined user/context message to thread.")
        # Create initial run with the teacher assistant
        t3 = time.time()
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=teacher_assistant_id
        )
        logger.info(f"Created run in {time.time() - t3:.2f}s")
        # Wait for the initial run to complete
        timeout = 60
        start_time = time.time()
        logger.info("Polling OpenAI run for completion...")
        while True:
            status = (await client.beta.threads.runs.retrieve(run.id, thread_id=thread_id)).status
            if status == "completed":
                break
            elif status in ["failed", "cancelled"]:
                logger.error(f"Run status: {status}")
                raise HTTPException(status_code=400, detail=f"Run status: {status}")
            elif time.time() - start_time > timeout:
                logger.error("Assistant timed out.")
                raise HTTPException(status_code=408, detail="Assistant timed out.")
            await asyncio.sleep(1)
        logger.info(f"OpenAI run completed in {time.time() - start_time:.2f}s")
        # Get the initial assistant message
        t4 = time.time()
        messages = await client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        logger.info(f"Fetched messages in {time.time() - t4:.2f}s")
        # Only include the user's question and the assistant's reply in the returned history
        initial_history = []
        if input.message:
            initial_history.append({"role": "user", "text": input.message})
        assistant_msg = next((msg for msg in messages.data if msg.role == "assistant"), None)
        if assistant_msg:
            logger.info(f"RAW ASSISTANT MESSAGE OBJECT: {assistant_msg}")
            initial_history.append({"role": "assistant", "text": assistant_msg.content[0].text.value})
        logger.info(f"Total /initialize_chat time: {time.time() - total_start:.2f}s")
        return {
            "thread_id": thread_id,
            "history": initial_history
        }
    except Exception as e:
        logger.error(f"Error in initialize_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_submission_feedback(assignment_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetches submission feedback data from the Django API.
    Logs detailed diagnostics for debugging in production.
    """
    if not assignment_id:
        logger.warning("No assignment_id provided to get_submission_feedback.")
        return None

    url = f"{DJANGO_API_BASE}/api/submission_focus/{assignment_id}/"
    api_key = os.getenv("INTERNAL_API_KEY")
    headers = {"X-API-KEY": api_key}

    logger.info(f"📡 Starting fetch for submission feedback.")
    logger.info(f"➡️ URL: {url}")
    logger.info(f"🔐 API Key present: {bool(api_key)}, Length: {len(api_key) if api_key else 'None'}")
    
    try:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info("About to send GET request to Django API")
            response = await client.get(url, headers=headers)
            t1 = time.time()
            logger.info(f"✅ HTTP status received: {response.status_code} in {t1 - t0:.2f}s")
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"📦 Response JSON for assignment {assignment_id}: {data}")
            return data

    except httpx.HTTPStatusError as e:
        logger.error(
            f"❌ HTTPStatusError for assignment {assignment_id}: {e.response.status_code} - {e.response.text}",
            exc_info=True
        )
    except httpx.RequestError as e:
        logger.error(
            f"❌ RequestError while fetching submission feedback for assignment {assignment_id}: {type(e).__name__} - {e}",
            exc_info=True
        )
    except Exception as e:
        logger.critical(
            f"🔥 Unexpected error in get_submission_feedback for assignment {assignment_id}: {type(e).__name__} - {str(e)}",
            exc_info=True
        )

    return None

# Improve error handling in get_trend_data
async def get_trend_data(assignment_id: int) -> dict:
    try:
        url = f"{DJANGO_API_BASE}/api/class_trends/{assignment_id}/"
        headers = {"X-API-KEY": os.getenv("INTERNAL_API_KEY")}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching trend data: {str(e)}")
        return {}

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test-cors")
async def test_cors(request: Request):
    """Test endpoint to debug CORS issues"""
    return {
        "status": "ok",
        "origin": request.headers.get("origin"),
        "host": request.headers.get("host"),
        "user-agent": request.headers.get("user-agent"),
        "allowed-origins": frontend_origins
    }

async def handle_run_completion(thread_id: str, run_id: str, timeout: int = 120) -> str:
    """
    Handles the completion of an OpenAI run, including tool calls.
    Returns the assistant's response text.
    """
    start_time = time.time()
    while True:
        run_status = await client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        if run_status.status == 'requires_action':
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                if tool_call.function.name == "debate_prompt_maker":
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": tool_call.function.arguments
                    })
            if tool_outputs:
                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
                continue
        elif run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise ValueError(f"Run failed with status: {run_status.status}")
        
        if time.time() - start_time > timeout:
            raise TimeoutError("Assistant timed out")
        
        await asyncio.sleep(2)

    # Get the response
    messages = await client.beta.threads.messages.list(thread_id=thread_id)
    response = None
    for msg in messages.data:
        if msg.role == "assistant" and msg.content:
            response = msg.content[0].text.value
            break

    if not response:
        raise ValueError("No assistant response found")
    
    return response

async def get_validated_debate_prompts(topic: str, max_retries: int = 2) -> Dict:
    """
    Calls the debate prompt maker assistant, validates the response, and retries if needed.
    Returns validated JSON or raises ValueError after retries.
    """
    assistant_id = os.getenv("DEBATE_PROMPT_MAKER_ID")
    if not assistant_id:
        raise ValueError("DEBATE_PROMPT_MAKER_ID not set")

    prompt = (
        "You are an expert debate coach and educator. Your task is to generate 6 thought-provoking debate prompts "
        "that will engage students in meaningful discussion and critical thinking.\n\n"
        f"Topic: {topic}\n\n"
        "Generate 6 debate prompts that:\n"
        "1. Are clear and concise\n"
        "2. Have balanced arguments on both sides\n"
        "3. Are age-appropriate for high school students\n"
        "4. Encourage critical thinking and analysis\n"
        "5. Can be supported with evidence\n\n"
        "You MUST use the provided tool 'debate_prompt_maker' and return ONLY a JSON object with the following structure:\n\n"
        '''{\n'''
        '''  "prompts": [\n'''
        '''    {\n'''
        '''      "title": "The Emancipation Proclamation marked a turning point in the Civil War by redefining its purpose.",\n'''
        '''      "description": "Argue whether the Proclamation transformed the war from a battle for union into a battle for freedom — or whether its impact has been overstated."\n'''
        '''    },\n'''
        '''    {\n'''
        '''      "title": "The Emancipation Proclamation was more symbolic than practical.",\n'''
        '''      "description": "Discuss whether the value of the Proclamation lies more in its symbolic moral force or in its tangible effects on enslaved people and the war."\n'''
        '''    }\n'''
        '''    // ... 4 more prompts ...\n'''
        '''  ]\n'''
        '''}'''
    )

    for attempt in range(max_retries):
        try:
            # Create thread and send prompt
            thread = await client.beta.threads.create()
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            # Create run with tool choice
            run = await client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                tool_choice={
                    "type": "function",
                    "function": {"name": "debate_prompt_maker"}
                }
            )

            # Handle run completion and tool calls
            response = await handle_run_completion(thread.id, run.id)
            
            # Validate response
            try:
                data = json.loads(response)
                validated = DebatePromptResponse(**data)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Validation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError("Failed to get valid debate prompts after retries")
                continue

        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to generate debate prompts: {str(e)}")
            continue

    raise RuntimeError("Unexpected error in validation loop")

@app.post("/debate-prompts")
async def generate_debate_prompts(input: DebatePromptInput):
    try:
        logger.info(f"Generating debate prompts for topic: {input.topic}")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

        # Get validated prompts with retries
        try:
            prompts = await get_validated_debate_prompts(input.topic)
            return prompts
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Error in generate_debate_prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debate/initialize", response_model=DebateInitResponse)
async def debate_initialize(input: DebateInitInput):
    """
    Initializes a debate session with the student-selected prompt and stance.
    The assistant takes the opposite stance and starts the debate.
    """
    logger.info("=== [DEBUG] /debate/initialize called ===")
    logger.info(f"Input received: prompt={input.prompt!r}, stance={input.stance!r}")

    assistant_id = os.getenv("DEBATOR_STUDENT_ID")
    logger.info(f"Assistant ID from env: {assistant_id!r}")
    if not assistant_id:
        logger.error("[ERROR] DEBATOR_STUDENT_ID not set in environment.")
        raise HTTPException(status_code=500, detail="DEBATOR_STUDENT_ID not set in environment.")

    try:
        # Create a new thread
        thread = await client.beta.threads.create()
        thread_id = thread.id
        logger.info(f"Created thread with ID: {thread_id}")

        # Compose the system message
        system_message = (
            f"This is the debate prompt: '{input.prompt}'. The user stance is: '{input.stance}'. "
            "You MUST take the opposite stance and start with a debate assertion of your position. "
            "Then the user will reply with their assertion, and you will engage in a meaningful debate to promote critical thinking."
        )
        logger.info(f"System message: {system_message}")

        # Send the system message to the thread
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=system_message
        )
        logger.info("System message sent to thread.")

        # Create and run the assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logger.info(f"Run created with ID: {run.id}")

        # Await completion using asyncio
        start_time = time.time()
        timeout = 90
        while True:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            logger.info(f"Run status: {run_status.status}")
            if run_status.status == 'completed':
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                logger.error(f"[ERROR] Run failed with status: {run_status.status}")
                raise HTTPException(status_code=500, detail=f"Run failed with status: {run_status.status}")
            if time.time() - start_time > timeout:
                logger.error("[ERROR] Assistant timed out.")
                raise HTTPException(status_code=408, detail="Assistant timed out.")
            await asyncio.sleep(2)

        # Get the assistant's first message
        messages = await client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        logger.info(f"Retrieved {len(messages.data)} messages from thread.")
        assistant_message = None
        for msg in messages.data:
            logger.info(f"Message role: {msg.role}, content: {msg.content}")
            if msg.role == "assistant" and msg.content:
                assistant_message = msg.content[0].text.value
                logger.info(f"Assistant message found: {assistant_message!r}")
                break
        if not assistant_message:
            logger.error("[ERROR] No assistant response found.")
            raise HTTPException(status_code=500, detail="No assistant response found.")

        logger.info("=== [DEBUG] /debate/initialize completed successfully ===")
        return DebateInitResponse(
            thread_id=thread_id,
            run_id=run.id,
            status=run_status.status,
            assistant_message=assistant_message
        )
    except Exception as e:
        logger.error("[EXCEPTION] Exception in /debate/initialize:", exc_info=True)
        raise

@app.post("/debate/respond", response_model=DebateChatResponse)
async def debate_respond(input: DebateChatInput):
    """
    Handles ongoing debate chat between the student and the assistant.
    Posts the student's message, runs the assistant, and returns the reply.
    """
    assistant_id = os.getenv("DEBATOR_STUDENT_ID")
    if not assistant_id:
        raise HTTPException(status_code=500, detail="DEBATOR_STUDENT_ID not set in environment.")

    # Post the student's message to the thread
    await client.beta.threads.messages.create(
        thread_id=input.thread_id,
        role="user",
        content=input.message
    )

    # Create and run the assistant
    run = await client.beta.threads.runs.create(
        thread_id=input.thread_id,
        assistant_id=assistant_id
    )

    # Await completion using asyncio
    start_time = time.time()
    timeout = 90
    while True:
        run_status = await client.beta.threads.runs.retrieve(
            thread_id=input.thread_id,
            run_id=run.id
        )
        if run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise HTTPException(status_code=500, detail=f"Run failed with status: {run_status.status}")
        if time.time() - start_time > timeout:
            raise HTTPException(status_code=408, detail="Assistant timed out.")
        await asyncio.sleep(2)

    # Get the assistant's reply
    messages = await client.beta.threads.messages.list(thread_id=input.thread_id, order="desc")
    assistant_message = None
    for msg in messages.data:
        if msg.role == "assistant" and msg.content:
            assistant_message = msg.content[0].text.value
            break
    if not assistant_message:
        raise HTTPException(status_code=500, detail="No assistant response found.")

    return DebateChatResponse(
        run_id=run.id,
        status=run_status.status,
        assistant_message=assistant_message
    )

async def initialize_elaboration_tutor_api(topic: str) -> Tuple[str, str]:
    """Handles the raw OpenAI API call for initialization"""
    print("\n🚀 STARTING initialize_elaboration_tutor")
    print(f"📝 Topic: {topic}")
    
    try:
        # Get assistant ID from environment
        assistant_id = os.getenv("ELABORATION_TUTOR_ASSISTANT_ID")
        if not assistant_id:
            print("❌ Assistant ID not found")
            raise HTTPException(status_code=500, detail="Assistant ID not found in environment variables.")
        print(f"✅ Using Assistant ID: {assistant_id}")

        # Create thread
        print("📝 Creating new thread...")
        thread = await client.beta.threads.create()
        print(f"✅ Thread created with ID: {thread.id}")

        # Send initial message
        print("📝 Sending initial message...")
        prompt = (
    f"You must now call the function `generate_argument_paragraph` using the topic: \"{topic}\".\n"
    "Your output must consist **only** of the structured JSON result from the tool function. Do **not** include any additional text, explanation, markdown, or commentary.\n\n"
    "Follow these output instructions:\n"
    "- Create a clear, arguable claim on the topic.\n"
    "- Provide one sentence of supporting evidence with a properly formatted MLA citation.\n"
    "- Prompt the user to elaborate by writing a full paragraph.\n"
    "- ALWAYS End with the sample full paragraph that combines the claim and evidence, this way the user can easily begin writing the elaboration.\n"
    "- Output must match the format below *exactly*.\n\n"
    "EXAMPLE FORMAT:\n"
    "{\n"
    '  "step_1_title": "Step 1: Let\'s Create a Claim",\n'
    '  "claim": "Social media use among teenagers can negatively impact mental health by increasing anxiety and social comparison.",\n'
    '  "step_2_title": "Step 2: Supporting Evidence",\n'
    '  "evidence": "A 2015 study found a strong correlation between frequent Facebook use and increased levels of envy and depression among college students (Tandoc et al. 143).",\n'
    '  "elaboration_prompt": "Now that we have a strong claim and supporting evidence, let\'s combine these ideas and elaborate to form a complete paragraph. Now it\'s your turn to elaborate...",\n'
    '  "full_paragraph": "Social media use among teenagers can negatively impact mental health by increasing anxiety and social comparison. This concern is backed by research: \\"A 2015 study found a strong correlation between frequent Facebook use and increased levels of envy and depression among college students (Tandoc et al. 143).\\""\n'
    "}"
)


        message = await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        print("✅ Initial message sent")

        # Create run
        print("🚀 Starting assistant run...")
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "generate_argument_paragraph"
                }
            }
        )
        print(f"✅ Run created with ID: {run.id}")

        # Monitor run status
        print("⏳ Waiting for completion...")
        start_time = datetime.now()
        timeout_seconds = 180

        while True:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == 'requires_action':
                print("🛠️ Processing function call...")
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = [{"tool_call_id": tc.id, "output": tc.function.arguments} for tc in tool_calls]
                
                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
            
            elif run_status.status == 'completed':
                print("✅ Run completed")
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                print(f"❌ Run {run_status.status}")
                raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
            
            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                raise HTTPException(status_code=504, detail="Request timed out")
            
            await asyncio.sleep(1)

        # Get the response
        print("📝 Retrieving messages...")
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value
        
        # Debug logging
        print("\n🔍 DEBUG: Raw Response Structure")
        print("=" * 50)
        print("Length:", len(response))
        print("Number of lines:", len(response.split('\n')))
        print("\n📝 COMPLETE RAW RESPONSE:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        return response, thread.id

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def elaboration_feedback_api(thread_id: str, user_message: str, full_paragraph: str) -> str:
    """Handles the raw OpenAI API call for feedback"""
    print("\n🚀 STARTING elaboration_feedback")
    print(f"📝 User message length: {len(user_message)}")
    print(f"📝 Full paragraph: {full_paragraph}")

    try:
        # Get assistant ID from environment
        assistant_id = os.getenv("ELABORATION_TUTOR_SECOND_CALL_ID")
        if not assistant_id:
            raise ValueError("❌ Elaboration Tutor Assistant ID not found in environment variables.")

        # Create a new thread for feedback
        print("📝 Creating new thread for feedback...")
        thread = await client.beta.threads.create()
        print(f"✅ Thread created with ID: {thread.id}")

        # Construct the prompt
        prompt = (
            "You are an expert writing tutor. Analyze the student's paragraph, focus your analsis on the text that follows the citation (source), and provide:\n"
            "- strengths,\n"
            "- areas for improvement,\n"
            "- suggestions for elaboration,\n"
            "- two guiding questions,\n"
            "- praise and encouragement.\n\n"
            "You MUST use the exact format of the function output for your response, from elaboration_feedback. At the end, you MUST include the user's full paragraph (as it currently stands) in the full_paragraph field of your JSON output. This is required every time, so the user can easily revise.\n\n"
            "Your response must look exactly like the function output. You must not return any text outside the function call. The end of your response must end with the full_paragraph and absolutely NOTHING after it.\n\n"
            "EXAMPLE FORMAT:\n"
            '''{\n'''
            '''  "strengths": ["Clear claim", "Good use of evidence"],\n'''
            '''  "areas_for_improvement": ["Could use more examples", "Needs stronger conclusion"],\n'''
            '''  "suggestions_for_elaboration": ["Add specific examples", "Explain implications"],\n'''
            '''  "guiding_questions": ["Can you provide a specific example?", "What are the long-term effects?"],\n'''
            '''  "praise_and_encouragement": "Your writing shows strong analytical skills. Keep developing your ideas!",\n'''
            '''  "full_paragraph": "The student's current paragraph text..."\n'''
            '''}\n\n'''
            f"Here is the user's current paragraph, and remember to focus your analysis on the text that follows the citation eg: (source):\n{full_paragraph}\n\n"
            f"Here is the user's full message:\n{user_message}"
        )

        print("📝 Sending message to thread...")
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        print("✅ Feedback prompt sent")

        # Start a run with the same assistant
        print("🚀 Starting assistant run for feedback...")
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "elaboration_feedback"
                }
            }
        )
        print(f"✅ Run created with ID: {run.id}")

        # Wait for completion and handle tool call
        print("⏳ Waiting for completion or tool call...")
        start_time = datetime.now()
        timeout_seconds = 180

        while True:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == 'requires_action':
                print("🛠️ Processing function/tool call...")
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    if tool_call.function.name == "elaboration_feedback":
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": tool_call.function.arguments
                        })
                if tool_outputs:
                    run = await client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    continue
            elif run_status.status == 'completed':
                print("✅ Run completed")
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                print(f"❌ Run {run_status.status}")
                raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
            
            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                raise HTTPException(status_code=504, detail="Request timed out")
            
            await asyncio.sleep(1)

        # Get the response
        print("📝 Retrieving feedback messages...")
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        
        if not messages.data or not messages.data[0].content:
            raise HTTPException(status_code=500, detail="No assistant message found")

        # Look for tool calls in the assistant's message
        for msg in messages.data:
            if msg.role == 'assistant':
                for content in msg.content:
                    if content.type == 'tool_calls':
                        for tool_call in content.tool_calls:
                            if tool_call.function.name == "elaboration_feedback":
                                return tool_call.function.arguments  # This is the JSON string

        # Fallback to text if no tool call found (shouldn't happen with tool_choice)
        response = messages.data[0].content[0].text.value

        # Debug logging
        print("\n🔍 DEBUG: Raw Feedback Response Structure")
        print("=" * 50)
        print("Length:", len(response))
        print("Number of lines:", len(response.split('\n')))
        print("\n📝 COMPLETE RAW FEEDBACK RESPONSE:")
        print("=" * 50)
        print(response)
        print("=" * 50)

        return response

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def elaboration_summary_api(topic: str, conversation: str) -> str:
    """Handles the raw OpenAI API call for summary"""
    print("\n🚀 STARTING elaboration_summary")
    print(f"📝 Topic: {topic}")
    print(f"📝 Conversation length: {len(conversation)}")

    try:
        # Get assistant ID from environment
        assistant_id = os.getenv("ELABORATION_ANALYZER_ID")
        if not assistant_id:
            raise ValueError("❌ Elaboration Tutor Assistant ID not found in environment variables.")

        # Create thread
        thread = await client.beta.threads.create()
        print(f"✅ Thread created with ID: {thread.id}")

        # Construct the prompt
        prompt = f"""
You are an AI assistant analyzing student elaboration in short writing responses (typically 2–3 sentences). Your goal is to extract structured insights for teachers and students. You MUST use the provided tool 'analyze_student_elaboration' and return ONLY a JSON object matching the following schema:

{{
  "strict": true,
  "additionalProperties": false,
  "type": "object",
  "required": [
    "topic",
    "techniques_used",
    "ai_responsiveness",
    "strengths",
    "areas_for_improvement",
    "claim_evidence_reasoning",
    "language_use_and_style",
    "diction_improvement_suggestion",
    "suggested_topics"
  ],
  "properties": {{
    "topic": {{"type": "string", "description": "The student's topic."}},
    "techniques_used": {{"type": "string", "description": "Techniques used in the elaboration."}},
    "ai_responsiveness": {{"type": "string", "description": "How the student responded to AI guidance."}},
    "strengths": {{"type": "array", "items": {{"type": "string"}}, "description": "Specific strengths in the elaboration."}},
    "areas_for_improvement": {{"type": "array", "items": {{"type": "string"}}, "description": "Areas for improvement and growth."}},
    "claim_evidence_reasoning": {{"type": "string", "description": "Discussion of claim-evidence alignment, reasoning depth, or gaps."}},
    "language_use_and_style": {{"type": "string", "description": "Rhetorical verbs, connectors, or metacognitive phrases used."}},
    "diction_improvement_suggestion": {{"type": "object", "properties": {{"word": {{"type": "string"}}, "suggested_alternatives": {{"type": "array", "items": {{"type": "string"}}}}}}, "required": ["word", "suggested_alternatives"], "description": "A word or two that could be improved for precision or connotation, with suggested alternatives."}},
    "suggested_topics": {{"type": "array", "items": {{"type": "string"}}, "description": "Three or four topics similar to the student's topic."}}
  }}
}}

- If a field is not applicable or no information is available, set it to null (for strings/objects) or an empty array (for lists).
- Do NOT return markdown, text, or any other format—ONLY the JSON object.
- You MUST call the tool 'analyze_student_elaboration'.

The student's topic is: {topic}

Here is the conversation:
{conversation}
"""
        # Send message
        print("📝 Sending message...")
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        print("✅ Summary prompt sent")

        # Start a run with the same assistant
        print("🚀 Starting assistant run for summary...")
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "analyze_student_elaboration"
                }
            }
        )
        print(f"✅ Run created with ID: {run.id}")

        # Monitor run status
        print("⏳ Waiting for completion...")
        start_time = datetime.now()
        timeout_seconds = 180

        while True:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == 'requires_action':
                print("🛠️ Processing function call...")
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                
                tool_outputs = []
                for tool_call in tool_calls:
                    if tool_call.function.name == "analyze_student_elaboration":
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": tool_call.function.arguments
                        })
                
                if tool_outputs:
                    run = await client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    continue
            
            elif run_status.status == 'completed':
                print("✅ Run completed")
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                print(f"❌ Run {run_status.status}")
                raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
            
            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                raise HTTPException(status_code=504, detail="Request timed out")
            
            await asyncio.sleep(1)  # Non-blocking sleep

        # Get the response
        print("📝 Retrieving messages...")
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value
        
        # Debug logging
        print("\n🔍 DEBUG: Raw Response Structure")
        print("=" * 50)
        print("Length:", len(response))
        print("Number of lines:", len(response.split('\n')))
        print("\n📝 COMPLETE RAW RESPONSE:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        return response

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def elaboration_model_sentences_api(topic: str) -> str:
    """Handles the raw OpenAI API call for model sentences based on the provided tool schema."""
    print("\n🚀 STARTING elaboration_model_sentences")
    print(f"📝 Topic: {topic}")

    try:
        assistant_id = os.getenv("MODEL_SENTENCE_WRITER_ID")
        if not assistant_id:
            raise ValueError("❌ MODEL_SENTENCE_WRITER_ID not found in environment variables.")

        thread = await client.beta.threads.create()
        print(f"✅ Thread created with ID: {thread.id}")

        prompt = (
            "You are an expert writing tutor. Your task is to generate model sentences that demonstrate different elaboration techniques.\n\n"
            f"Topic: {topic}\n\n"
            "Generate a model sentence for each of the following elaboration techniques. Each sentence should:\n"
            "1. Be clear and concise\n"
            "2. Effectively demonstrate the technique\n"
            "3. Be relevant to the topic\n"
            "4. Be suitable for student learning\n\n"
            "You MUST use the provided tool 'generate_model_sentences' and return ONLY a JSON object with technique names as keys and model sentences as values.\n\n"
            "EXAMPLE FORMAT:\n"
            '''{\n'''
            '''  "Cause and Effect Reasoning": "The invention of the smartphone has revolutionized communication by enabling instant global connectivity.",\n'''
            '''  "If-Then Statements": "If we continue to ignore climate change, then future generations will face severe environmental challenges.",\n'''
            '''  "Definition and Clarification": "Artificial intelligence, the simulation of human intelligence by machines, is transforming various industries.",\n'''
            '''  "Comparisons and Contrasts": "Unlike traditional classrooms, online learning offers flexibility but requires greater self-discipline.",\n'''
            '''  "Analogies and Metaphors": "The human brain is like a supercomputer, processing countless pieces of information simultaneously.",\n'''
            '''  "Rhetorical Questions": "How can we expect students to succeed when they lack access to basic educational resources?",\n'''
            '''  "Explaining Implications": "The rise of social media has profound implications for mental health and social interaction.",\n'''
            '''  "Explaining the Why or How": "Electric cars reduce emissions by eliminating the need for fossil fuels in transportation.",\n'''
            '''  "Hypothetical Examples or Situations": "Imagine a world where renewable energy powers every home and business."\n'''
            '''}'''
        )
        
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            tool_choice={"type": "function", "function": {"name": "generate_model_sentences"}}
        )

        start_time = datetime.now()
        timeout_seconds = 180
        tool_response_json = None  # 🆕 storage

        while True:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

            if run_status.status == 'requires_action':
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tc in tool_calls:
                    tool_response_json = tc.function.arguments  # 🧠 Store it right here
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": tool_response_json
                    })

                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue

            elif run_status.status == 'completed':
                if tool_response_json:
                    return tool_response_json  # ✅ Return it here
                else:
                    raise HTTPException(status_code=500, detail="Tool output was not captured.")

            elif run_status.status in ['failed', 'cancelled', 'expired']:
                raise HTTPException(status_code=500, detail=f"Run {run_status.status}")

            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                raise HTTPException(status_code=504, detail="Request timed out")

            await asyncio.sleep(1)

    except Exception as e:
        print(f"❌ Error in elaboration_model_sentences_api: {str(e)}")
        print(f"Stack trace:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/elaboration/initialize")
async def get_validated_elaboration_tutor_response(request: ElaborationInitRequest) -> Dict[str, Any]:
    """Initialize elaboration tutor conversation with validation and retries."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response, thread_id = await initialize_elaboration_tutor_api(request.topic)
            
            # Try to extract JSON from the response string
            if isinstance(response, str):
                try:
                    json_str = extract_json_from_response(response)
                    data = json.loads(json_str)
                except Exception:
                    # Fallback: try to parse as direct JSON
                    json_str = response
                    data = json.loads(response)
            else:
                data = response
                json_str = json.dumps(response)
            
            # Validate with Pydantic
            validated = GenerateArgumentParagraphResponse(**data)
            return {
                "response": json.loads(json_str),
                "thread_id": thread_id
            }
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid elaboration tutor response after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

@app.post("/api/elaboration/feedback")
async def get_validated_elaboration_feedback_response(request: ElaborationFeedbackRequest) -> Dict[str, Any]:
    """Get feedback on elaboration with validation and retries."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = await elaboration_feedback_api(
                request.thread_id,
                request.user_message,
                request.full_paragraph
            )
            
            # Try to extract JSON from the response string
            if isinstance(response, str):
                try:
                    json_str = extract_json_from_response(response)
                    data = json.loads(json_str)
                except Exception:
                    # Fallback: try to parse as direct JSON
                    json_str = response
                    data = json.loads(response)
            else:
                data = response
                json_str = json.dumps(response)
            
            # Validate with Pydantic
            validated = ElaborationFeedbackResponse(**data)
            return {"response": json.loads(json_str)}
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid elaboration feedback response after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

@app.post("/api/elaboration/summary")
async def get_validated_elaboration_summary(request: ElaborationSummaryRequest) -> Dict[str, Any]:
    """Get summary of elaboration conversation with validation and retries."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = await elaboration_summary_api(
                request.topic,
                request.conversation
            )
            
            # Try to extract JSON from the response string
            if isinstance(response, str):
                try:
                    json_str = extract_json_from_response(response)
                    data = json.loads(json_str)
                except Exception:
                    # Fallback: try to parse as direct JSON
                    json_str = response
                    data = json.loads(response)
            else:
                data = response
                json_str = json.dumps(response)
            
            # Validate with Pydantic
            validated = ElaborationSummaryResponse(**data)
            return {"response": json.loads(json_str)}
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid elaboration summary response after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

@app.post("/api/elaboration/model-sentences")
async def get_validated_elaboration_model_sentences(request: ElaborationModelSentencesRequest) -> Dict[str, Any]:
    """Get model sentences for elaboration with validation and retries."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response_str = await elaboration_model_sentences_api(request.topic)
            
            # The response is a JSON string representing the arguments of the tool call.
            # It should contain a 'techniques' object.
            data = json.loads(response_str)
            
            # Validate with Pydantic
            validated = ElaborationModelSentencesResponse(**data)
            return {"response": validated.techniques}
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid model sentences response after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

@app.get("/debug/raw_submissions/{assignment_id}")
def debug_raw_submissions(assignment_id: int, db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM jarvis_app_submission WHERE assignment_id = :aid"), {"aid": assignment_id})
    rows = result.fetchall()
    return {"count": len(rows), "rows": [dict(row._mapping) for row in rows]}

@app.websocket("/ws/debate")
async def websocket_debate(websocket: WebSocket):
    await websocket.accept()
    thread_id = None
    run_id = None
    assistant_id = os.getenv("DEBATOR_STUDENT_ID")
    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            data = msg.get("data", {})

            if msg_type == "init":
                prompt = data.get("prompt")
                stance = data.get("stance")
                if not prompt or not stance:
                    await websocket.send_json({"type": "error", "data": {"message": "Missing prompt or stance."}})
                    continue
                thread = await client.beta.threads.create()
                thread_id = thread.id
                system_message = (
                    f"This is the debate prompt: '{prompt}'. The user stance is: '{stance}'. "
                    "You MUST take the opposite stance and start with a debate assertion of your position. "
                    "Then the user will reply with their assertion, and you will engage in a meaningful debate to promote critical thinking."
                )
                await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=system_message
                )
                response_stream = await client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    stream=True
                )
                full_message = ""
                async for chunk in response_stream:
                    if (
                        hasattr(chunk, 'data') and
                        hasattr(chunk.data, 'delta') and
                        hasattr(chunk.data.delta, 'content')
                    ):
                        token = chunk.data.delta.content
                        if isinstance(token, list):
                            parts = []
                            for content_block in token:
                                if hasattr(content_block, 'text') and hasattr(content_block.text, 'value') and content_block.text.value is not None:
                                    parts.append(str(content_block.text.value))
                                elif hasattr(content_block, 'text') and isinstance(content_block.text, str):
                                    parts.append(content_block.text)
                                elif hasattr(content_block, 'value') and content_block.value is not None:
                                    parts.append(str(content_block.value))
                                elif isinstance(content_block, str):
                                    parts.append(content_block)
                                else:
                                    parts.append(str(content_block))
                            token = ''.join(parts)
                        full_message += token
                        await websocket.send_json({
                            "type": "token",
                            "data": {"role": "assistant", "content": token}
                        })
                await websocket.send_json({
                    "type": "message",
                    "data": {"role": "assistant", "content": full_message}
                })

            elif msg_type == "message":
                if not thread_id:
                    await websocket.send_json({"type": "error", "data": {"message": "Debate not initialized."}})
                    continue
                user_message = data.get("content")
                if not user_message:
                    await websocket.send_json({"type": "error", "data": {"message": "Missing message content."}})
                    continue
                await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=user_message
                )
                response_stream = await client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    stream=True
                )
                full_message = ""
                async for chunk in response_stream:
                    if (
                        hasattr(chunk, 'data') and
                        hasattr(chunk.data, 'delta') and
                        hasattr(chunk.data.delta, 'content')
                    ):
                        token = chunk.data.delta.content
                        if isinstance(token, list):
                            parts = []
                            for content_block in token:
                                if hasattr(content_block, 'text') and hasattr(content_block.text, 'value') and content_block.text.value is not None:
                                    parts.append(str(content_block.text.value))
                                elif hasattr(content_block, 'text') and isinstance(content_block.text, str):
                                    parts.append(content_block.text)
                                elif hasattr(content_block, 'value') and content_block.value is not None:
                                    parts.append(str(content_block.value))
                                elif isinstance(content_block, str):
                                    parts.append(content_block)
                                else:
                                    parts.append(str(content_block))
                            token = ''.join(parts)
                        full_message += token
                        await websocket.send_json({
                            "type": "token",
                            "data": {"role": "assistant", "content": token}
                        })
                await websocket.send_json({
                    "type": "message",
                    "data": {"role": "assistant", "content": full_message}
                })

            elif msg_type == "end":
                await websocket.close()
                break
            else:
                await websocket.send_json({"type": "error", "data": {"message": f"Unknown message type: {msg_type}"}})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws/rhetorical-device-practice")
async def websocket_rhetorical_device_practice(websocket: WebSocket):
    import asyncio
    from openai import OpenAIError, RateLimitError, APIError
    
    print("[DEBUG] WebSocket connection attempt received for /ws/rhetorical-device-practice")
    await websocket.accept()
    thread_id = None
    assistant_id = os.getenv("RHETORICAL_DEVICE_TUTOR_ID")
    print(f"[DEBUG] RHETORICAL_DEVICE_TUTOR_ID: {assistant_id}")

    if not assistant_id:
        print("[DEBUG] RHETORICAL_DEVICE_TUTOR_ID not configured")
        await websocket.send_json({
            "type": "error", 
            "data": {"message": "RHETORICAL_DEVICE_TUTOR_ID not configured", "fatal": True}
        })
        await websocket.close()
        return

    async def retry_openai_request(request_func, max_retries=3, base_delay=1.0):
        """Retry OpenAI requests with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return await request_func()
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[DEBUG] Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                raise e
            except (APIError, OpenAIError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[DEBUG] OpenAI error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
                    continue
                raise e
        
    try:
        while True:
            data = await websocket.receive_json()
            print(f"[DEBUG] Received data: {data}")
            msg_type = data.get("type")
            
            if msg_type == "init":
                # Extract data from the nested structure
                init_data = data.get("data", {})
                last_debate_topic = init_data.get("last_debate_topic")
                student_id = init_data.get("student_id")
                
                if not last_debate_topic:
                    await websocket.send_json({
                        "type": "error", 
                        "data": {"message": "Missing last debate topic", "fatal": False}
                    })
                    continue
                
                try:
                    # Create thread with retry logic
                    thread = await retry_openai_request(
                        lambda: client.beta.threads.create()
                    )
                    thread_id = thread.id
                    
                    # Create initial system message but don't run the assistant yet
                    system_message = (
                        f"You are a rhetorical device tutor helping a student practice using different rhetorical techniques. "
                        f"The student's last debate topic was: '{last_debate_topic}'. "
                        f"When the student tells you which rhetorical device they want to practice, immediately jump into teaching that specific device. "
                        f"Provide a clear definition, give an example related to their debate topic, and then ask them to create their own example. "
                        f"Do not give a generic introduction or list all devices unless they ask for options. "
                        f"Ask questions in your last two sentences to help guide their thinking and get them started on crafting a sentence using the device. "
                        f"Be encouraging and provide specific feedback on their attempts. "
                        f"***MODEL OUTPUT EXAMPLE: "
                        f"User: I want to learn about Anaphora "
                        f"GPT: Excellent choice! "
                        f"Anaphora is a rhetorical device that involves repeating the same word or phrase at the beginning of successive clauses or sentences. This creates emphasis and can make your argument more memorable and persuasive. "
                        f"Example related to your debate topic: "
                        f"Lincoln inspired a divided nation. Lincoln confronted the horrors of slavery. Lincoln redefined what freedom meant for every American. "
                        f"Now, it's your turn! Try writing two or three sentences about Lincoln that begin with the same word or phrase to emphasize your point. "
                        f"What word would impact the audience in an anaphora style? What point or ideal do you want to emphasize or add importance to? "
                    )
                    
                    await retry_openai_request(
                        lambda: client.beta.threads.messages.create(
                            thread_id=thread_id,
                            role="user",
                            content=system_message
                        )
                    )
                    
                    # Just send thread_id back to client - don't run the assistant yet
                    await websocket.send_json({
                        "type": "thread_created",
                        "data": {"thread_id": thread_id}
                    })
                    
                except Exception as e:
                    print(f"[DEBUG] Error during init: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": "Failed to initialize conversation. Please try again.",
                            "fatal": False,
                            "retry": True
                        }
                    })
                    continue

            elif msg_type == "message":
                if not thread_id:
                    await websocket.send_json({
                        "type": "error", 
                        "data": {"message": "Session not initialized", "fatal": False}
                    })
                    continue
                
                # Extract message from the nested structure
                message_data = data.get("data", {})
                user_message = message_data.get("content")
                if not user_message:
                    await websocket.send_json({
                        "type": "error", 
                        "data": {"message": "Missing message content", "fatal": False}
                    })
                    continue
                
                try:
                    # Add user message to thread with retry
                    await retry_openai_request(
                        lambda: client.beta.threads.messages.create(
                            thread_id=thread_id,
                            role="user",
                            content=user_message
                        )
                    )
                    
                    # Stream the response with retry
                    response_stream = await retry_openai_request(
                        lambda: client.beta.threads.runs.create(
                            thread_id=thread_id,
                            assistant_id=assistant_id,
                            stream=True
                        )
                    )
                    
                    full_message = ""
                    async for chunk in response_stream:
                        if (hasattr(chunk, 'data') and 
                            hasattr(chunk.data, 'delta') and 
                            hasattr(chunk.data.delta, 'content')):
                            
                            token = chunk.data.delta.content
                            if isinstance(token, list):
                                parts = []
                                for content_block in token:
                                    if (hasattr(content_block, 'text') and 
                                        hasattr(content_block.text, 'value') and 
                                        content_block.text.value is not None):
                                        parts.append(str(content_block.text.value))
                                    elif (hasattr(content_block, 'text') and 
                                          isinstance(content_block.text, str)):
                                        parts.append(content_block.text)
                                    elif (hasattr(content_block, 'value') and 
                                          content_block.value is not None):
                                        parts.append(str(content_block.value))
                                    elif isinstance(content_block, str):
                                        parts.append(content_block)
                                    else:
                                        parts.append(str(content_block))
                                token = ''.join(parts)
                            
                            full_message += token
                            await websocket.send_json({
                                "type": "token",
                                "data": {"role": "assistant", "content": token}
                            })
                    
                    await websocket.send_json({
                        "type": "message",
                        "data": {"role": "assistant", "content": full_message}
                    })
                    
                except Exception as e:
                    print(f"[DEBUG] Error during message processing: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": "Failed to process your message. Please try sending it again.",
                            "fatal": False,
                            "retry": True,
                            "last_message": user_message
                        }
                    })
                    continue

            elif msg_type == "end":
                await websocket.close()
                break
                
            else:
                await websocket.send_json({
                    "type": "error", 
                    "data": {"message": f"Unknown message type: {msg_type}", "fatal": False}
                })
                
    except Exception as e:
        print(f"[DEBUG] WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {
                "message": "Connection error occurred. Please close and reopen the conversation.",
                "fatal": True
            }
        })
        await websocket.close()

@app.websocket("/ws/whit_chat")
async def websocket_whit_chat(websocket: WebSocket):
    await websocket.accept()
    thread_id = None
    teacher_assistant_id = os.getenv("TEACHER_CHAT_BUDDY_ID")
    
    if not teacher_assistant_id:
        await websocket.send_json({
            "type": "error", 
            "data": {"message": "TEACHER_CHAT_BUDDY_ID not configured"}
        })
        await websocket.close()
        return

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            data = msg.get("data", {})
            
            if msg_type == "init":
                assignment_id = data.get("assignment_id")
                assistant_id = data.get("assistant_id") or teacher_assistant_id
                user_message = data.get("message")
                
                # Create thread
                thread = await client.beta.threads.create()
                thread_id = thread.id
                
                # Fetch submission feedback from DB (recreate db session for websocket)
                db = SessionLocal()
                try:
                    context_text = "No assignment context available."
                    if assignment_id:
                        submission_data = await get_submission_focus_data(assignment_id, db)
                        if submission_data:
                            context_text = (
                                f"Here is the metrics summary for the assignment:\n{submission_data['metrics_summary']}\n\n"
                                f"Here is the submission feedback data for all students:\n{submission_data['students']}"
                            )
                        else:
                            context_text = "No submission feedback available."
                finally:
                    db.close()
                
                # Combine context and user's question
                if user_message:
                    combined_message = f"Using this submission feedback data, collaborate as a thought partner and assistant with the teacher.\n{context_text}\n\n{user_message}"
                else:
                    combined_message = f"Using this submission feedback data, collaborate as a thought partner and assistant with the teacher.\n{context_text}"
                
                # Add message to thread
                await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=combined_message
                )
                
                # Stream the response
                response_stream = await client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    stream=True
                )
                
                full_message = ""
                async for chunk in response_stream:
                    if (
                        hasattr(chunk, 'data') and
                        hasattr(chunk.data, 'delta') and
                        hasattr(chunk.data.delta, 'content') and
                        chunk.data.delta.content is not None
                    ):
                        token = chunk.data.delta.content
                        if isinstance(token, list):
                            parts = []
                            for content_block in token:
                                if hasattr(content_block, 'text') and hasattr(content_block.text, 'value') and content_block.text.value is not None:
                                    parts.append(str(content_block.text.value))
                                elif hasattr(content_block, 'text') and isinstance(content_block.text, str):
                                    parts.append(content_block.text)
                                elif hasattr(content_block, 'value') and content_block.value is not None:
                                    parts.append(str(content_block.value))
                                elif isinstance(content_block, str):
                                    parts.append(content_block)
                                else:
                                    parts.append(str(content_block))
                            token = ''.join(parts)
                        
                        full_message += token
                        await websocket.send_json({
                            "type": "token",
                            "data": {"content": token}
                        })
                
                # Send final message with thread_id
                await websocket.send_json({
                    "type": "message",
                    "data": {
                        "thread_id": thread_id,
                        "content": full_message,
                        "history": [
                            {"role": "user", "text": user_message} if user_message else None,
                            {"role": "assistant", "text": full_message}
                        ]
                    }
                })
                
            elif msg_type == "message":
                if not thread_id:
                    await websocket.send_json({
                        "type": "error", 
                        "data": {"message": "Chat not initialized"}
                    })
                    continue
                
                user_message = data.get("message")
                assistant_id = data.get("assistant_id") or teacher_assistant_id
                
                if not user_message:
                    await websocket.send_json({
                        "type": "error", 
                        "data": {"message": "Missing message content"}
                    })
                    continue
                
                # Add user message to thread
                await client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=user_message
                )
                
                # Stream the response
                response_stream = await client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    stream=True
                )
                
                full_message = ""
                async for chunk in response_stream:
                    if (
                        hasattr(chunk, 'data') and
                        hasattr(chunk.data, 'delta') and
                        hasattr(chunk.data.delta, 'content') and
                        chunk.data.delta.content is not None
                    ):
                        token = chunk.data.delta.content
                        if isinstance(token, list):
                            parts = []
                            for content_block in token:
                                if hasattr(content_block, 'text') and hasattr(content_block.text, 'value') and content_block.text.value is not None:
                                    parts.append(str(content_block.text.value))
                                elif hasattr(content_block, 'text') and isinstance(content_block.text, str):
                                    parts.append(content_block.text)
                                elif hasattr(content_block, 'value') and content_block.value is not None:
                                    parts.append(str(content_block.value))
                                elif isinstance(content_block, str):
                                    parts.append(content_block)
                                else:
                                    parts.append(str(content_block))
                            token = ''.join(parts)
                        
                        full_message += token
                        await websocket.send_json({
                            "type": "token",
                            "data": {"content": token}
                        })
                
                # Send final message
                await websocket.send_json({
                    "type": "message",
                    "data": {"content": full_message}
                })
                
            elif msg_type == "end":
                await websocket.close()
                break
                
            else:
                await websocket.send_json({
                    "type": "error", 
                    "data": {"message": f"Unknown message type: {msg_type}"}
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws/essay-chat/{submission_id}")
async def websocket_essay_chat(websocket: WebSocket, submission_id: int):
    """WebSocket endpoint for essay-specific chat using saved thread_id and assistant_id"""
    await websocket.accept()
    
    try:
        # Get database session
        db = SessionLocal()
        
        try:
            # Get thread info for this submission
            thread = db.query(Thread).filter(
                Thread.submission_id == submission_id,
                Thread.thread_type == 'essay_analysis',
                Thread.is_active == True
            ).first()
            
            if not thread:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "No chat thread found for this submission"}
                })
                return
            
            thread_id = thread.openai_thread_id
            assistant_id = thread.assistant_id
            
            # Get submission details for context
            submission = db.query(Submission).filter(Submission.id == submission_id).first()
            if not submission:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Submission not found"}
                })
                return
            
            # Send initial context message
            await websocket.send_json({
                "type": "init",
                "data": {
                    "message": f"Ready to chat about your essay: {submission.assignment.title}",
                    "thread_id": thread_id,
                    "assistant_id": assistant_id
                }
            })
            
            # Main chat loop
            while True:
                try:
                    msg = await websocket.receive_json()
                    msg_type = msg.get("type")
                    data = msg.get("data", {})
                    
                    if msg_type == "chat":
                        user_message = data.get("message", "")
                        if not user_message.strip():
                            continue
                        
                        # Add user message to thread
                        await client.beta.threads.messages.create(
                            thread_id=thread_id,
                            role="user",
                            content=user_message
                        )
                        
                        # Create and run with streaming
                        run = await client.beta.threads.runs.create(
                            thread_id=thread_id,
                            assistant_id=assistant_id,
                            stream=True
                        )
                        
                        # Stream the response
                        response_text = ""
                        async for event in run:
                            if event.event == "thread.message.delta":
                                delta = event.data.delta
                                if delta.content:
                                    for content in delta.content:
                                        if content.type == "text":
                                            token = content.text.value
                                            response_text += token
                                            await websocket.send_json({
                                                "type": "token",
                                                "data": {"content": token}
                                            })
                            elif event.event == "thread.run.completed":
                                await websocket.send_json({
                                    "type": "message",
                                    "data": {"content": response_text}
                                })
                                break
                        
                        # Update thread activity
                        thread.last_activity_at = datetime.utcnow()
                        db.commit()
                        
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error in essay chat: {str(e)}")
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": str(e)}
                        })
                    break
                    
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in essay chat websocket: {str(e)}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/exit-ticket-analysis/{thread_id}")
async def websocket_exit_ticket_analysis_chat(websocket: WebSocket, thread_id: str):
    """WebSocket endpoint for exit ticket analysis chat using OpenAI thread_id"""
    await websocket.accept()
    
    try:
        # Get database session
        db = SessionLocal()
        
        try:
            # Get the analysis record to verify the thread exists
            analysis = db.query(ExitTicketClasswideAnalysis).filter(
                ExitTicketClasswideAnalysis.thread_id == thread_id
            ).first()
            
            if not analysis:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "No analysis found for this thread"}
                })
                return
            
            # Use the EXIT_TICKET_CLASS_ID assistant
            assistant_id = os.getenv("EXIT_TICKET_CLASS_ID")
            if not assistant_id:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "EXIT_TICKET_CLASS_ID assistant not configured"}
                })
                return
            
            # Send initial context message
            await websocket.send_json({
                "type": "init",
                "data": {
                    "message": f"Ready to discuss the exit ticket analysis for {analysis.assignment.title}",
                    "thread_id": thread_id,
                    "assistant_id": assistant_id
                }
            })
            
            # Main chat loop
            while True:
                try:
                    msg = await websocket.receive_json()
                    msg_type = msg.get("type")
                    
                    if msg_type == "init":
                        # Handle initial prompt from frontend
                        init_message = msg.get("message", "")
                        if init_message.strip():
                            # Add initial message to thread
                            await client.beta.threads.messages.create(
                                thread_id=thread_id,
                                role="user",
                                content=init_message
                            )
                            
                            # Create and run with streaming
                            run = await client.beta.threads.runs.create(
                                thread_id=thread_id,
                                assistant_id=assistant_id,
                                stream=True
                            )
                            
                            # Stream the response
                            response_text = ""
                            async for event in run:
                                if event.event == "thread.message.delta":
                                    delta = event.data.delta
                                    if delta.content:
                                        for content in delta.content:
                                            if content.type == "text":
                                                token = content.text.value
                                                response_text += token
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "data": {"content": token}
                                                })
                                elif event.event == "thread.run.completed":
                                    await websocket.send_json({
                                        "type": "message",
                                        "data": {"content": response_text}
                                    })
                                    break
                                    
                    elif msg_type == "message":
                        user_message = msg.get("message", "")
                        if not user_message.strip():
                            continue
                        
                        # Add user message to thread
                        await client.beta.threads.messages.create(
                            thread_id=thread_id,
                            role="user",
                            content=user_message
                        )
                        
                        # Create and run with streaming
                        run = await client.beta.threads.runs.create(
                            thread_id=thread_id,
                            assistant_id=assistant_id,
                            stream=True
                        )
                        
                        # Stream the response
                        response_text = ""
                        async for event in run:
                            if event.event == "thread.message.delta":
                                delta = event.data.delta
                                if delta.content:
                                    for content in delta.content:
                                        if content.type == "text":
                                            token = content.text.value
                                            response_text += token
                                            await websocket.send_json({
                                                "type": "token",
                                                "data": {"content": token}
                                            })
                            elif event.event == "thread.run.completed":
                                await websocket.send_json({
                                    "type": "message",
                                    "data": {"content": response_text}
                                })
                                break
                        
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error in exit ticket analysis chat: {str(e)}")
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": str(e)}
                        })
                    
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error in exit ticket analysis chat websocket: {str(e)}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        try:
            await websocket.close()
        except Exception:
            pass


async def debate_analysis_api(transcript: list) -> dict:
    """
    Calls OpenAI Assistant for debate analysis, handles tool call, and returns validated output.
    """
    assistant_id = os.getenv("DEBATE_ANALYZER_ID")
    if not assistant_id:
        raise HTTPException(status_code=500, detail="DEBATE_ANALYZER_ID not set in environment variables.")

    # Compose prompt and sample output
    prompt = (
        "You are an expert debate coach and AI analyst. Analyze the following debate transcript and return a structured JSON object using the required tool 'process_debate_analysis'. "
        "You MUST call the tool process_debate_analysis and return ONLY the JSON object, no extra text. "
        "Here is a sample output:\n"
        "{\n"
        "  \"overall_score\": 87,\n"
        "  \"ai_feedback_summary\": \"Strong argument structure, but needs more evidence.\",\n"
        "  \"bloom_percentages\": {\"remember\": 10, \"understand\": 20, \"apply\": 15, \"analyze\": 20, \"evaluate\": 20, \"create\": 15},\n"
        "  \"category_scores\": {\"argument_quality\": 90, \"critical_thinking\": 85, \"rhetorical_skill\": 80, \"responsiveness\": 88, \"structure_clarity\": 92, \"style_delivery\": 80},\n"
        "  \"persuasive_appeals\": [{\"appeal_type\": \"ethos\", \"count\": 2, \"example_snippets\": [\"As a student...\"], \"effectiveness_score\": 80}],\n"
        "  \"rhetorical_devices\": [{\"device_type\": \"metaphor\", \"raw_label\": \"metaphor\", \"description\": \"Used a metaphor.\", \"count\": 1, \"example_snippets\": [\"The mind is a garden...\"], \"effectiveness_score\": 90}],\n"
        "}\n"
        "Here is the transcript:\n"
        f"{json.dumps(transcript)}"
    )

    # Create thread and send message
    thread = await client.beta.threads.create()
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )

    # Start run with tool choice
    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        tool_choice={"type": "function", "function": {"name": "process_debate_analysis"}},
        tools=[{"type": "function", "function": {"name": "process_debate_analysis"}}]
    )

    # Wait for completion and handle tool call
    start_time = datetime.now()
    timeout_seconds = 180
    while True:
        run_status = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == 'requires_action':
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tc in tool_calls:
                tool_outputs.append({
                    "tool_call_id": tc.id,
                    "output": tc.function.arguments
                })
            run = await client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            continue
        elif run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
        if (datetime.now() - start_time).total_seconds() > timeout_seconds:
            await client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
            raise HTTPException(status_code=504, detail="Request timed out")
        await asyncio.sleep(1)

    # Fetch thread messages and extract tool call output from assistant's message
    messages = await client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == 'assistant':
            for content in msg.content:
                if getattr(content, 'type', None) == 'tool_calls':
                    for tool_call in getattr(content, 'tool_calls', []):
                        if getattr(tool_call.function, 'name', None) == "process_debate_analysis":
                            return json.loads(tool_call.function.arguments)
            # Fallback: try to parse text as JSON if tool_calls not found
            for content in msg.content:
                if getattr(content, 'type', None) == 'text':
                    try:
                        return json.loads(content.text.value)
                    except Exception:
                        continue
    raise HTTPException(status_code=500, detail="No valid tool call output found in assistant's message.")

@app.post("/api/debate/analyze")
async def get_validated_debate_analysis(request: dict) -> dict:
    """
    Analyze a debate transcript, validate output, and retry up to 2 times if needed.
    """
    transcript = request.get("transcript")
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required.")

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response_str = await debate_analysis_api(transcript)
            print("DEBUG: Raw OpenAI response:", response_str)
            # Try to extract JSON from the response string
            if isinstance(response_str, str):
                try:
                    data = json.loads(response_str)
                    print("DEBUG: Parsed JSON data:", data)
                except Exception as e:
                    print("DEBUG: JSON parse error, raw string:", response_str)
                    raise
            else:
                data = response_str
                print("DEBUG: Data is not a string, value:", data)
            # Validate with Pydantic
            validated = ProcessDebateAnalysisResponse(**data)
            print("DEBUG: Pydantic validated data:", validated)
            return {"response": validated.dict()}
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid debate analysis after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1)
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

# Add new Pydantic models for lesson plan generation
class ElaborationLessonPlanRequest(BaseModel):
    class_id: int
    assignment_id: int
    grade: str
    teacher_id: int

# Add debate lesson plan request schema
class DebateLessonPlanRequest(BaseModel):
    class_id: int
    assignment_id: int
    grade: str
    teacher_id: int

class LessonPlanMaterialsRequest(BaseModel):
    lesson_plan: dict
    lesson_plan_id: Union[str, int]  # Accept both string temp_ids and int db_ids

class ElaborationLessonPlanResponse(BaseModel):
    lesson_plans: List[dict]
    success: bool
    message: Optional[str] = None

# Add debate lesson plan response schema
class DebateLessonPlanResponse(BaseModel):
    lesson_plans: List[dict]
    success: bool
    message: Optional[str] = None

class LessonPlanMaterialsResponse(BaseModel):
    materials: str
    success: bool
    message: Optional[str] = None

async def get_classwide_elaboration_data(class_id: int, assignment_id: int, db: Session):
    """Fetch classwide elaboration data from database"""
    try:
        # Query the ClasswideElaborationData table
        result = db.execute(text("""
            SELECT most_popular_topics, unique_topics, most_common_techniques, 
                   least_common_techniques, mixed_usage_observations, average_alignment_score,
                   reasoning_depth_summary, evidence_language_notes, claim_elaboration_gaps,
                   overgeneralizations, rhetorical_verbs, causal_connectors, metacognitive_phrases
            FROM jarvis_app_classwideelaborationdata 
            WHERE class_instance_id = :class_id AND assignment_id = :assignment_id
            LIMIT 1
        """), {"class_id": class_id, "assignment_id": assignment_id})
        
        row = result.fetchone()
        if not row:
            raise ValueError("No classwide elaboration data found")
            
        return {
            "topics": {
                "most_popular": row.most_popular_topics or [],
                "unique": row.unique_topics or []
            },
            "elaboration_techniques": {
                "most_common": row.most_common_techniques or [],
                "least_common": row.least_common_techniques or [],
                "mixed_usage_observations": row.mixed_usage_observations or ""
            },
            "claim_evidence_reasoning": {
                "average_alignment_score": row.average_alignment_score,
                "reasoning_depth_summary": row.reasoning_depth_summary or "",
                "evidence_language_notes": row.evidence_language_notes or "",
                "claim_elaboration_gaps": row.claim_elaboration_gaps or [],
                "overgeneralizations": row.overgeneralizations or []
            },
            "language_use_and_style": {
                "rhetorical_verbs": row.rhetorical_verbs or [],
                "causal_connectors": row.causal_connectors or [],
                "metacognitive_phrases": row.metacognitive_phrases or []
            }
        }
    except Exception as e:
        logger.error(f"Error fetching classwide data: {e}")
        raise

async def elaboration_lesson_plan_generation_api(class_id: int, assignment_id: int, grade: str, elaboration_analysis: dict) -> List[dict]:
    """Generate lesson plans using OpenAI Assistant based on classwide elaboration analysis"""
    print(f"[DEBUG] elaboration_lesson_plan_generation_api called with class_id={class_id}, assignment_id={assignment_id}, grade={grade}")

    # Get the teaching-specific assistant ID
    teaching_assistant_id = os.getenv("LESSON_PLAN_MAKER_ID")
    if not teaching_assistant_id:
        print("[DEBUG] LESSON_PLAN_MAKER_ID not found in environment variables")
        raise ValueError("Teaching Assistant ID not found in environment variables")

    print(f"[DEBUG] Using teaching assistant ID: {teaching_assistant_id}")
    print(f"[DEBUG] Prepared elaboration analysis: {json.dumps(elaboration_analysis, indent=2)}")

    # Create prompt for lesson plan generation (following teach_with_whit structure)
    prompt = f"""
Based on the following classwide elaboration analysis for a {grade} grade class, design lesson plans that address the identified strengths and areas for improvement.

Class Analysis:
{json.dumps(elaboration_analysis, indent=2)}

Please create lesson plans that:
1. Build on the most common techniques students are already using successfully
2. Address the least common or missing techniques that need development
3. Incorporate topics that students are interested in
4. Address any claim-evidence reasoning gaps
5. Is appropriate for {grade} grade level

You must use the lesson_plan_format function to structure your response.
"""

    print(f"[DEBUG] Created prompt for OpenAI")

    # Create thread
    thread = await client.beta.threads.create()
    thread_id = thread.id
    print(f"[DEBUG] Created new thread: {thread_id}")

    # Send message to thread
    print("[DEBUG] Sending message to OpenAI...")
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    # Start the run with explicit tool choice
    print("[DEBUG] Starting OpenAI run...")
    run = await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=teaching_assistant_id,
        tool_choice={
            "type": "function",
            "function": {
                "name": "lesson_plan_format"
            }
        }
    )

    print(f"[DEBUG] Run created with ID: {run.id}")

    # Wait for completion and handle function calls (following teach_with_whit pattern)
    lesson_plans = []
    start_time = time.time()
    timeout = 180

    while True:
        run_status = await client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        print(f"[DEBUG] Run status: {run_status.status}")
        
        if run_status.status == 'requires_action':
            print("🛠️ Processing function call...")
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            
            tool_outputs = []
            for tool_call in tool_calls:
                if tool_call.function.name == "lesson_plan_format":
                    try:
                        plan_data = json.loads(tool_call.function.arguments)
                        lesson_plans.append(plan_data)
                        print(f"[DEBUG] Processed lesson plan: {plan_data.get('title', 'Untitled')}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(plan_data)
                        })
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing lesson plan JSON: {e}")
                        raise ValueError(f"Invalid JSON in lesson plan format: {str(e)}")
            
            if tool_outputs:
                print("[DEBUG] Submitting tool outputs...")
                await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
                
        elif run_status.status == 'completed':
            print("✅ Run completed")
            break
            
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            print(f"❌ Run {run_status.status}")
            raise Exception(f"Run {run_status.status}")
            
        if time.time() - start_time > timeout:
            print("❌ Run timed out")
            raise TimeoutError("Lesson plan generation timed out")
            
        await asyncio.sleep(2)

    if not lesson_plans:
        print("❌ No lesson plans returned from assistant.")
        raise ValueError("No lesson plans generated")

    print(f"[DEBUG] Generated {len(lesson_plans)} lesson plan(s)")
    return lesson_plans

async def lesson_plan_materials_generation_api(lesson_plan_text: str) -> str:
    """Generate materials for a lesson plan using OpenAI Assistant"""
    print("[DEBUG] lesson_plan_materials_generation_api called")
    print(lesson_plan_text[:200] + ("..." if len(lesson_plan_text) > 200 else ""))
    
    # Get assistant ID
    assistant_id = os.getenv("LESSON_PLAN_MATERIAL_MAKER_ID")
    if not assistant_id:
        raise ValueError("❌ LESSON_PLAN_MATERIAL_MAKER_ID not found in environment variables.")
    
    # Create thread and send lesson plan
    thread = await client.beta.threads.create()
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=(
            "Read the following lesson plan and, if it calls for materials (like model paragraphs, outlines, graphic organizers, examples, or worksheets), generate those materials using plain markdown. "
            "Do NOT wrap any section in triple backticks or code blocks unless you are showing actual code (such as Python, JavaScript, etc.). "
            "Use markdown headings (#, ##, ###), lists (-, *), bold (**bold**), italics (_italic_), and blockquotes (>) for structure. "
            "Do NOT use code blocks labeled as 'markdown'. "
            "Your output should be ready for direct markdown-to-HTML rendering. "
            "Avoid excessive blank lines and use clear section breaks with headings.\n\n"
            "Lesson Plan:\n" + lesson_plan_text
        )
    )
    
    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    
    # Wait for completion
    timeout = 120
    start_time = time.time()
    while True:
        status = (await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)).status
        if status == "completed":
            break
        elif status in ["failed", "cancelled", "expired"]:
            raise Exception(f"Run {status}")
        elif time.time() - start_time > timeout:
            raise TimeoutError("Lesson plan materials generation timed out.")
        await asyncio.sleep(2)
    
    # Get the response
    messages = await client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant" and msg.content:
            material_markdown = msg.content[0].text.value.strip()
            
            # Replace all colons with dashes for better formatting
            material_markdown = material_markdown.replace(":", " -")
            
            print("=" * 60)
            print(material_markdown)
            print("=" * 60)
            return material_markdown
    
    raise ValueError("No usable assistant message received for lesson plan materials.")

@app.post("/api/elaboration/lesson-plans", response_model=ElaborationLessonPlanResponse)
async def generate_elaboration_lesson_plans(request: ElaborationLessonPlanRequest, db: Session = Depends(get_db)):
    """Generate lesson plans based on classwide elaboration analysis with retry logic"""
    print(f"[DEBUG] generate_elaboration_lesson_plans called with: {request}")
    
    try:
        # Fetch classwide elaboration data
        elaboration_analysis = await get_classwide_elaboration_data(request.class_id, request.assignment_id, db)
        
        # Try to generate lesson plans with retry logic
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] Attempt {attempt + 1} of {max_retries}")
                lesson_plans = await elaboration_lesson_plan_generation_api(
                    request.class_id, 
                    request.assignment_id, 
                    request.grade,
                    elaboration_analysis
                )
                
                # Add unique IDs for frontend reference
                for i, plan in enumerate(lesson_plans):
                    plan['temp_id'] = f"temp_{request.class_id}_{request.assignment_id}_{i}_{int(time.time())}"
                
                return ElaborationLessonPlanResponse(
                    lesson_plans=lesson_plans,
                    success=True,
                    message=f"Successfully generated {len(lesson_plans)} lesson plan(s)"
                )
                
            except Exception as e:
                last_error = e
                print(f"[DEBUG] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Wait before retry
                continue
        
        # If all retries failed
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate lesson plans after {max_retries} attempts: {str(last_error)}"
        )
        
    except Exception as e:
        print(f"[DEBUG] Error in generate_elaboration_lesson_plans: {str(e)}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lesson-plans/materials", response_model=LessonPlanMaterialsResponse)
async def generate_lesson_plan_materials(request: LessonPlanMaterialsRequest):
    """Generate materials for a lesson plan with retry logic"""
    print(f"[DEBUG] generate_lesson_plan_materials called")
    print(f"[DEBUG] Request data: lesson_plan_id={request.lesson_plan_id}, lesson_plan keys={list(request.lesson_plan.keys()) if request.lesson_plan else 'None'}")
    
    # Convert lesson plan dict to text format
    lesson_plan = request.lesson_plan
    lesson_plan_text = f"""
Title: {lesson_plan.get('title', '')}
Grade Level: {lesson_plan.get('grade_level', '')}
Subject: {lesson_plan.get('subject', '')}

Learning Objectives:
{chr(10).join(f"- {obj}" for obj in lesson_plan.get('learning_objectives', []))}

Warm-Up:
{lesson_plan.get('warm_up', '')}

Mini-Lesson:
{lesson_plan.get('mini_lesson', '')}

Guided Practice:
{lesson_plan.get('guided_practice', '')}

Independent Practice:
{lesson_plan.get('independent_practice', '')}

Formative Assessment:
{lesson_plan.get('formative_assessment', '')}

Closure/Reflection:
{lesson_plan.get('closure_reflection', '')}

Materials:
{chr(10).join(f"- {mat}" for mat in lesson_plan.get('materials', []))}

Key Design Principles:
{json.dumps(lesson_plan.get('key_design_principles', {}), indent=2)}
"""
    
    print(f"[DEBUG] Generated lesson plan text (first 200 chars): {lesson_plan_text[:200]}...")
    
    # Retry logic for materials generation
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Materials generation attempt {attempt + 1} of {max_retries}")
            materials = await lesson_plan_materials_generation_api(lesson_plan_text)
            
            print(f"[DEBUG] Successfully generated materials (length: {len(materials)})")
            
            return LessonPlanMaterialsResponse(
                materials=materials,
                success=True,
                message="Successfully generated materials"
            )
            
        except Exception as e:
            last_error = e
            print(f"[DEBUG] Materials generation attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Wait before retry
                continue
    
    # If all retries failed
    print(f"[DEBUG] All materials generation attempts failed. Last error: {str(last_error)}")
    import traceback
    print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
    raise HTTPException(
        status_code=500, 
        detail=f"Failed to generate materials after {max_retries} attempts: {str(last_error)}"
    )

async def get_classwide_debate_data(class_id: int, assignment_id: int, db: Session):
    """Fetch classwide debate data from database for AI analysis - optimized for production PostgreSQL"""
    try:
        logger.info(f"Fetching classwide debate data for class {class_id}, assignment {assignment_id}")
        
        # Detect database type for optimal query selection
        database_url = os.getenv("DATABASE_URL", "")
        is_postgresql = database_url.startswith("postgresql://") or database_url.startswith("postgres://")
        
        if is_postgresql:
            # PostgreSQL-optimized queries with STRING_AGG
            logger.info("Using PostgreSQL-optimized queries with STRING_AGG")
            
            # Get aggregated debate metrics with AI feedback in single query
            debate_metrics = db.execute(text("""
                SELECT 
                    COUNT(*) as total_debates,
                    AVG(da.overall_score) as avg_overall_score,
                    AVG(da.argument_quality) as avg_argument_quality,
                    AVG(da.critical_thinking) as avg_critical_thinking,
                    AVG(da.rhetorical_skill) as avg_rhetorical_skill,
                    AVG(da.responsiveness) as avg_responsiveness,
                    AVG(da.structure_clarity) as avg_structure_clarity,
                    AVG(da.remember_pct) as avg_remember,
                    AVG(da.understand_pct) as avg_understand,
                    AVG(da.apply_pct) as avg_apply,
                    AVG(da.analyze_pct) as avg_analyze,
                    AVG(da.evaluate_pct) as avg_evaluate,
                    AVG(da.create_pct) as avg_create,
                    STRING_AGG(DISTINCT da.ai_feedback_summary, ' | ') as all_ai_feedback
                FROM jarvis_app_studentdebate sd
                JOIN jarvis_app_customuser u ON sd.student_id = u.id
                LEFT JOIN jarvis_app_debateanalysis da ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                AND da.overall_score IS NOT NULL
            """), {"class_id": class_id, "assignment_id": assignment_id})
            
            # Get aggregated persuasive appeals data with examples
            appeals_data = db.execute(text("""
                SELECT 
                    pa.appeal_type,
                    SUM(pa.count) as total_count,
                    AVG(pa.effectiveness_score) as avg_effectiveness,
                    STRING_AGG(DISTINCT pa.example_snippets::text, ' | ') as sample_examples
                FROM jarvis_app_persuasiveappeal pa
                JOIN jarvis_app_debateanalysis da ON pa.analysis_id = da.id
                JOIN jarvis_app_studentdebate sd ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                GROUP BY pa.appeal_type
                ORDER BY total_count DESC
            """), {"class_id": class_id, "assignment_id": assignment_id})
            
            # Get aggregated rhetorical devices data with examples
            devices_data = db.execute(text("""
                SELECT 
                    COALESCE(NULLIF(rd.device_type, 'other'), rd.raw_label) as device_name,
                    SUM(rd.count) as total_count,
                    AVG(rd.effectiveness_score) as avg_effectiveness,
                    STRING_AGG(DISTINCT rd.example_snippets::text, ' | ') as sample_examples,
                    rd.description
                FROM jarvis_app_rhetoricaldevice rd
                JOIN jarvis_app_debateanalysis da ON rd.analysis_id = da.id
                JOIN jarvis_app_studentdebate sd ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                GROUP BY COALESCE(NULLIF(rd.device_type, 'other'), rd.raw_label), rd.description
                ORDER BY total_count DESC
                LIMIT 10
            """), {"class_id": class_id, "assignment_id": assignment_id})
            
        else:
            # SQLite-compatible queries (separate queries for aggregation)
            logger.info("Using SQLite-compatible queries with separate aggregation")
            
            # Get aggregated debate metrics (without STRING_AGG for SQLite compatibility)
            debate_metrics = db.execute(text("""
                SELECT 
                    COUNT(*) as total_debates,
                    AVG(da.overall_score) as avg_overall_score,
                    AVG(da.argument_quality) as avg_argument_quality,
                    AVG(da.critical_thinking) as avg_critical_thinking,
                    AVG(da.rhetorical_skill) as avg_rhetorical_skill,
                    AVG(da.responsiveness) as avg_responsiveness,
                    AVG(da.structure_clarity) as avg_structure_clarity,
                    AVG(da.remember_pct) as avg_remember,
                    AVG(da.understand_pct) as avg_understand,
                    AVG(da.apply_pct) as avg_apply,
                    AVG(da.analyze_pct) as avg_analyze,
                    AVG(da.evaluate_pct) as avg_evaluate,
                    AVG(da.create_pct) as avg_create
                FROM jarvis_app_studentdebate sd
                JOIN jarvis_app_customuser u ON sd.student_id = u.id
                LEFT JOIN jarvis_app_debateanalysis da ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                AND da.overall_score IS NOT NULL
            """), {"class_id": class_id, "assignment_id": assignment_id})
            
            # Get aggregated persuasive appeals data (without STRING_AGG for SQLite compatibility)
            appeals_data = db.execute(text("""
                SELECT 
                    pa.appeal_type,
                    SUM(pa.count) as total_count,
                    AVG(pa.effectiveness_score) as avg_effectiveness
                FROM jarvis_app_persuasiveappeal pa
                JOIN jarvis_app_debateanalysis da ON pa.analysis_id = da.id
                JOIN jarvis_app_studentdebate sd ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                GROUP BY pa.appeal_type
                ORDER BY total_count DESC
            """), {"class_id": class_id, "assignment_id": assignment_id})
            
            # Get aggregated rhetorical devices data (without STRING_AGG for SQLite compatibility)
            devices_data = db.execute(text("""
                SELECT 
                    COALESCE(NULLIF(rd.device_type, 'other'), rd.raw_label) as device_name,
                    SUM(rd.count) as total_count,
                    AVG(rd.effectiveness_score) as avg_effectiveness,
                    rd.description
                FROM jarvis_app_rhetoricaldevice rd
                JOIN jarvis_app_debateanalysis da ON rd.analysis_id = da.id
                JOIN jarvis_app_studentdebate sd ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                GROUP BY COALESCE(NULLIF(rd.device_type, 'other'), rd.raw_label), rd.description
                ORDER BY total_count DESC
                LIMIT 10
            """), {"class_id": class_id, "assignment_id": assignment_id})
        
        metrics = debate_metrics.fetchone()
        
        if not metrics or metrics.total_debates == 0:
            raise ValueError("No debate data found for this class and assignment")
        
        # Get DebateNoteToTeacher summaries
        notes_data = db.execute(text("""
            SELECT 
                dntt.note,
                dntt.created_at
            FROM jarvis_app_debatenotetoteacher dntt
            JOIN jarvis_app_studentdebate sd ON dntt.student_debate_id = sd.id
            WHERE sd.assignment_id = :assignment_id
            AND sd.student_id IN (
                SELECT student_id FROM jarvis_app_class_students 
                WHERE class_id = :class_id
            )
            ORDER BY dntt.created_at DESC
        """), {"class_id": class_id, "assignment_id": assignment_id})
        
        notes = notes_data.fetchall()
        appeals = appeals_data.fetchall()
        devices = devices_data.fetchall()
        
        # Get prompt analytics
        prompts_data = db.execute(text("""
            SELECT 
                dp.text as prompt_text,
                sd.chosen_side,
                COUNT(*) as usage_count
            FROM jarvis_app_debateprompt dp
            JOIN jarvis_app_studentdebate sd ON sd.prompt_id = dp.id
            WHERE sd.assignment_id = :assignment_id
            AND sd.student_id IN (
                SELECT student_id FROM jarvis_app_class_students 
                WHERE class_id = :class_id
            )
            GROUP BY dp.text, sd.chosen_side
            ORDER BY usage_count DESC
        """), {"class_id": class_id, "assignment_id": assignment_id})
        
        prompts = prompts_data.fetchall()
        
        # Process AI feedback summaries based on database type
        if is_postgresql and hasattr(metrics, 'all_ai_feedback') and metrics.all_ai_feedback:
            # PostgreSQL: Use STRING_AGG result
            all_ai_feedback = [
                feedback.strip() for feedback in metrics.all_ai_feedback.split(" | ")
                if feedback and feedback.strip()
            ]
        else:
            # SQLite: Get AI feedback summaries separately
            ai_feedback_data = db.execute(text("""
                SELECT DISTINCT da.ai_feedback_summary
                FROM jarvis_app_studentdebate sd
                JOIN jarvis_app_customuser u ON sd.student_id = u.id
                LEFT JOIN jarvis_app_debateanalysis da ON da.student_debate_id = sd.id
                WHERE sd.assignment_id = :assignment_id
                AND sd.student_id IN (
                    SELECT student_id FROM jarvis_app_class_students 
                    WHERE class_id = :class_id
                )
                AND da.ai_feedback_summary IS NOT NULL
                AND da.ai_feedback_summary != ''
            """), {"class_id": class_id, "assignment_id": assignment_id})
            
            ai_feedback_rows = ai_feedback_data.fetchall()
            all_ai_feedback = [
                row.ai_feedback_summary.strip() for row in ai_feedback_rows
                if row.ai_feedback_summary and row.ai_feedback_summary.strip()
            ]
        
        feedback_count_to_include = max(1, int(len(all_ai_feedback) * 0.75))  # At least 1, but 75% of total
        
        # Structure the optimized data for AI analysis
        structured_data = {
            "class_id": class_id,
            "assignment_id": assignment_id,
            "total_debates": int(metrics.total_debates),
            "overall_metrics": {
                "avg_overall_score": round(float(metrics.avg_overall_score or 0), 1),
                "avg_argument_quality": round(float(metrics.avg_argument_quality or 0), 1),
                "avg_critical_thinking": round(float(metrics.avg_critical_thinking or 0), 1),
                "avg_rhetorical_skill": round(float(metrics.avg_rhetorical_skill or 0), 1),
                "avg_responsiveness": round(float(metrics.avg_responsiveness or 0), 1),
                "avg_structure_clarity": round(float(metrics.avg_structure_clarity or 0), 1),
                "bloom_averages": {
                    "remember_avg": round(float(metrics.avg_remember or 0), 1),
                    "understand_avg": round(float(metrics.avg_understand or 0), 1),
                    "apply_avg": round(float(metrics.avg_apply or 0), 1),
                    "analyze_avg": round(float(metrics.avg_analyze or 0), 1),
                    "evaluate_avg": round(float(metrics.avg_evaluate or 0), 1),
                    "create_avg": round(float(metrics.avg_create or 0), 1)
                }
            },
            "ai_feedback_summaries": all_ai_feedback[:feedback_count_to_include],  # Include 75% of all AI feedback summaries
            "debate_notes_to_teacher": [
                {"note": note.note, "date": note.created_at.strftime("%Y-%m-%d") if hasattr(note.created_at, 'strftime') else str(note.created_at)}
                for note in notes
            ],  # Include ALL notes to teacher
            "persuasive_appeals_summary": {
                appeal.appeal_type: {
                    "total_count": int(appeal.total_count),
                    "avg_effectiveness": round(float(appeal.avg_effectiveness or 0), 1),
                    "sample_examples": [ex.strip() for ex in (getattr(appeal, 'sample_examples', '') or '').split(' | ') if ex.strip()][:3] if is_postgresql else []
                }
                for appeal in appeals
            },
            "rhetorical_devices_summary": {
                device.device_name: {
                    "total_count": int(device.total_count),
                    "avg_effectiveness": round(float(device.avg_effectiveness or 0), 1),
                    "description": device.description or "",
                    "sample_examples": [ex.strip() for ex in (getattr(device, 'sample_examples', '') or '').split(' | ') if ex.strip()][:3] if is_postgresql else []
                }
                for device in devices
            },
            "prompt_analytics": {
                "most_popular_prompts": [
                    {"prompt": p.prompt_text, "usage_count": p.usage_count, "side": p.chosen_side}
                    for p in prompts[:5]
                ],
                "pro_percentage": round(
                    sum(p.usage_count for p in prompts if p.chosen_side == 'pro') / 
                    sum(p.usage_count for p in prompts) * 100 if prompts else 0, 1
                ),
                "con_percentage": round(
                    sum(p.usage_count for p in prompts if p.chosen_side == 'con') / 
                    sum(p.usage_count for p in prompts) * 100 if prompts else 0, 1
                )
            }
        }
        
        logger.info(f"Successfully structured optimized classwide debate data: {metrics.total_debates} debates using {'PostgreSQL' if is_postgresql else 'SQLite'} queries")
        return structured_data
        
    except Exception as e:
        logger.error(f"Error fetching classwide debate data: {e}")
        raise

async def debate_insights_generation_api(class_id: int, assignment_id: int, debate_data: dict) -> dict:
    """Generate AI insights for debate data using OpenAI Assistant"""
    print(f"[DEBUG] debate_insights_generation_api called with class_id={class_id}, assignment_id={assignment_id}")
    
    # Get the debate analyzer assistant ID
    assistant_id = os.getenv("DEBATE_ANALYZER_ID")
    if not assistant_id:
        print("[DEBUG] DEBATE_ANALYZER_ID not found in environment variables")
        raise ValueError("DEBATE_ANALYZER_ID not found in environment variables")
    
    print(f"[DEBUG] Using debate analyzer assistant ID: {assistant_id}")
    print(f"[DEBUG] Prepared debate data keys: {list(debate_data.keys())}")
    
    # Create prompt for debate insights generation
    prompt = f"""
Based on the following classwide debate analysis data, provide comprehensive insights for the teacher.

Class Analysis Data:
{json.dumps(debate_data, indent=2)}

You MUST use the provided tool 'generate_classwide_debate_insights' and return ONLY a JSON object with the following structure:

{{
  "student_notes_summary": "Summary of student notes to teacher about their key takeaways",
  "general_observations": "Broader observations about class debate skills, trends in argumentation, rhetorical devices, persuasive appeals, and critical thinking levels",
  "teaching_recommendations": "Specific, actionable teaching recommendations in bullet format addressing identified patterns",
  "socratic_seminar_questions": ["question1", "question2", "question3", "question4", "question5", "question6", "question7"]
}}

Focus on:
- Overall performance trends and patterns
- Effective use of persuasive appeals (ethos, pathos, logos)
- Rhetorical device usage and effectiveness
- Bloom's taxonomy distribution and critical thinking levels
- Areas where students excel vs. areas needing improvement
- Specific instructional strategies to address gaps
- Generate 7 thought-provoking Socratic seminar questions based on the debate topics and themes that emerged from the class debates

Provide insights that are practical, specific, and immediately actionable for a teacher planning their next lessons.
"""
    
    print(f"[DEBUG] Created prompt for OpenAI (length: {len(prompt)})")
    
    # Create thread
    thread = await client.beta.threads.create()
    thread_id = thread.id
    print(f"[DEBUG] Created new thread: {thread_id}")
    
    # Send message to thread
    print("[DEBUG] Sending message to OpenAI...")
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )
    
    # Start the run with explicit tool choice
    print("[DEBUG] Starting OpenAI run...")
    run = await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        tool_choice={
            "type": "function",
            "function": {
                "name": "generate_classwide_debate_insights"
            }
        }
    )
    
    print(f"[DEBUG] Run created with ID: {run.id}")
    
    # Wait for completion and handle function calls (following lesson plan pattern)
    insights_data = {}
    start_time = time.time()
    timeout = 180
    
    while True:
        run_status = await client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        print(f"[DEBUG] Run status: {run_status.status}")
        
        if run_status.status == 'requires_action':
            print("🛠️ Processing function call...")
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            
            tool_outputs = []
            for tool_call in tool_calls:
                if tool_call.function.name == "generate_classwide_debate_insights":
                    try:
                        insights_data = json.loads(tool_call.function.arguments)
                        print(f"[DEBUG] Processed debate insights: {list(insights_data.keys())}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(insights_data)
                        })
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing debate insights JSON: {e}")
                        raise ValueError(f"Invalid JSON in debate insights format: {str(e)}")
            
            if tool_outputs:
                print("[DEBUG] Submitting tool outputs...")
                await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
                
        elif run_status.status == 'completed':
            print("✅ Run completed")
            break
            
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            print(f"❌ Run {run_status.status}")
            raise Exception(f"Run {run_status.status}")
            
        if time.time() - start_time > timeout:
            print("❌ Run timed out")
            raise TimeoutError("Debate insights generation timed out")
            
        await asyncio.sleep(2)
    
    if not insights_data:
        print("❌ No insights data returned from assistant.")
        raise ValueError("No debate insights generated")
    
    print(f"[DEBUG] Generated debate insights with keys: {list(insights_data.keys())}")
    return insights_data

@app.get("/api/get-cached-debate-insights/")
async def get_cached_debate_insights(class_id: int, assignment_id: int, db: Session = Depends(get_db)):
    """Check if AI insights are already cached in the database"""
    try:
        logger.info(f"Checking cached debate insights for class {class_id}, assignment {assignment_id}")
        
        # Query the ClasswideDebateData table for existing insights
        result = db.execute(text("""
            SELECT student_notes_summary, general_observations, teaching_recommendations, socratic_seminar_questions
            FROM jarvis_app_classwidedebatedata 
            WHERE class_instance_id = :class_id AND assignment_id = :assignment_id
            LIMIT 1
        """), {"class_id": class_id, "assignment_id": assignment_id})
        
        row = result.fetchone()
        
        if row and row.student_notes_summary and row.general_observations and row.teaching_recommendations:
            # We have cached insights
            socratic_questions = row.socratic_seminar_questions if row.socratic_seminar_questions else []
            insights = DebateInsightsResponse(
                student_notes_summary=row.student_notes_summary,
                general_observations=row.general_observations,
                teaching_recommendations=row.teaching_recommendations,
                socratic_seminar_questions=socratic_questions
            )
            
            return CachedDebateInsightsResponse(
                success=True,
                has_data=True,
                insights=insights,
                message="Cached insights found"
            )
        else:
            # No cached insights
            return CachedDebateInsightsResponse(
                success=True,
                has_data=False,
                insights=None,
                message="No cached insights found"
            )
            
    except Exception as e:
        logger.error(f"Error checking cached debate insights: {str(e)}")
        return CachedDebateInsightsResponse(
            success=False,
            has_data=False,
            insights=None,
            message=f"Error checking cached insights: {str(e)}"
        )

async def get_validated_debate_insights(class_id: int, assignment_id: int, debate_data: dict) -> dict:
    """Generate debate insights with validation and retries"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"Validation attempt {attempt + 1} of {max_retries} for debate insights")
            
            # Generate insights using the API
            insights_data = await debate_insights_generation_api(
                class_id, 
                assignment_id, 
                debate_data
            )
            
            # Validate the response structure
            if not isinstance(insights_data, dict):
                raise ValueError("Response is not a dictionary")
            
            required_fields = ["student_notes_summary", "general_observations", "teaching_recommendations"]
            for field in required_fields:
                if field not in insights_data:
                    raise ValueError(f"Missing required field: {field}")
                if not isinstance(insights_data[field], str):
                    raise ValueError(f"Field {field} must be a string")
                if not insights_data[field].strip():
                    raise ValueError(f"Field {field} cannot be empty")
            
            # Validate with Pydantic
            validated = DebateInsightsResponse(**insights_data)
            return insights_data
            
        except (ValueError, ValidationError) as e:
            logger.error(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid debate insights after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(2)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

@app.post("/api/generate-debate-insights/")
async def generate_debate_insights(request: GenerateDebateInsightsRequest, db: Session = Depends(get_db)):
    """Generate new AI insights for debate data (read-only, JavaScript will save to Django)"""
    try:
        logger.info(f"Generating debate insights for class {request.class_id}, assignment {request.assignment_id}")
        
        # Fetch classwide debate data
        debate_data = await get_classwide_debate_data(request.class_id, request.assignment_id, db)
        
        # Generate and validate AI insights
        insights_data = await get_validated_debate_insights(
            request.class_id, 
            request.assignment_id, 
            debate_data
        )
        
        # Return the validated insights (JavaScript will save to Django)
        insights = DebateInsightsResponse(
            student_notes_summary=insights_data["student_notes_summary"],
            general_observations=insights_data["general_observations"],
            teaching_recommendations=insights_data["teaching_recommendations"],
            socratic_seminar_questions=insights_data.get("socratic_seminar_questions", [])
        )
        
        return GenerateDebateInsightsResponse(
            success=True,
            insights=insights,
            message="Successfully generated debate insights"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper error handling)
        raise
    except Exception as e:
        logger.error(f"Error in generate_debate_insights: {str(e)}")
        return GenerateDebateInsightsResponse(
            success=False,
            insights=None,
            message=f"Error generating insights: {str(e)}"
        )

async def debate_lesson_plan_generation_api(class_id: int, assignment_id: int, grade: str, debate_analysis: dict) -> List[dict]:
    """Generate lesson plans using OpenAI Assistant based on classwide debate analysis"""
    print(f"[DEBUG] debate_lesson_plan_generation_api called with class_id={class_id}, assignment_id={assignment_id}, grade={grade}")

    # Get the teaching-specific assistant ID
    teaching_assistant_id = os.getenv("LESSON_PLAN_MAKER_ID")
    if not teaching_assistant_id:
        print("[DEBUG] LESSON_PLAN_MAKER_ID not found in environment variables")
        raise ValueError("Teaching Assistant ID not found in environment variables")

    print(f"[DEBUG] Using teaching assistant ID: {teaching_assistant_id}")
    print(f"[DEBUG] Prepared debate analysis: {json.dumps(debate_analysis, indent=2)}")

    # Create prompt for lesson plan generation based on debate data
    prompt = f"""
Based on the following classwide debate analysis for a {grade} grade class, design exactly 2 unique and distinct lesson plans that address the identified strengths and areas for improvement in debate skills.

Class Debate Analysis:
{json.dumps(debate_analysis, indent=2)}

Please create exactly 2 different lesson plans that:
1. Build on the strongest debate skills students are already demonstrating
2. Address areas where students need improvement (argument quality, critical thinking, rhetorical skill, responsiveness, structure & clarity)
3. Incorporate effective persuasive appeals and rhetorical devices that students are using well
4. Address gaps in Bloom's taxonomy levels that need development
5. Use student debate notes and feedback to inform instruction
6. Is appropriate for {grade} grade level

Focus on practical debate skills like:
- Constructing stronger arguments with evidence
- Improving critical thinking and analysis
- Developing rhetorical effectiveness
- Enhancing responsiveness to opposing arguments
- Building clear structure and organization
- Advancing students through higher-order thinking (Bloom's taxonomy)

IMPORTANT: Generate exactly 2 distinct lesson plans with different titles and focus areas. Do not repeat lesson plans.

You must use the lesson_plan_format function exactly 2 times to structure your response.
"""

    print(f"[DEBUG] Created prompt for OpenAI")

    # Create thread
    thread = await client.beta.threads.create()
    thread_id = thread.id
    print(f"[DEBUG] Created new thread: {thread_id}")

    # Send message to thread
    print("[DEBUG] Sending message to OpenAI...")
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    # Start the run with explicit tool choice
    print("[DEBUG] Starting OpenAI run...")
    run = await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=teaching_assistant_id,
        tool_choice={
            "type": "function",
            "function": {
                "name": "lesson_plan_format"
            }
        }
    )

    print(f"[DEBUG] Run created with ID: {run.id}")

    # Wait for completion and handle function calls (following teach_with_whit pattern)
    lesson_plans = []
    start_time = time.time()
    timeout = 90  # Reduced timeout since we're generating only 2 lesson plans

    while True:
        run_status = await client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        print(f"[DEBUG] Run status: {run_status.status}")
        
        if run_status.status == 'requires_action':
            print("🛠️ Processing function call...")
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            
            tool_outputs = []
            for tool_call in tool_calls:
                if tool_call.function.name == "lesson_plan_format":
                    try:
                        plan_data = json.loads(tool_call.function.arguments)
                        lesson_plans.append(plan_data)
                        print(f"[DEBUG] Processed lesson plan: {plan_data.get('title', 'Untitled')}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(plan_data)
                        })
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing lesson plan JSON: {e}")
                        raise ValueError(f"Invalid JSON in lesson plan format: {str(e)}")
            
            if tool_outputs:
                print("[DEBUG] Submitting tool outputs...")
                await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
                
        elif run_status.status == 'completed':
            print("✅ Run completed")
            break
            
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            print(f"❌ Run {run_status.status}")
            raise Exception(f"Run {run_status.status}")
            
        if time.time() - start_time > timeout:
            print("❌ Run timed out")
            raise TimeoutError("Lesson plan generation timed out")
            
        # Adaptive polling: start with shorter intervals, increase gradually
        elapsed = time.time() - start_time
        if elapsed < 10:
            await asyncio.sleep(1)  # Check every 1 second for first 10 seconds
        elif elapsed < 30:
            await asyncio.sleep(2)  # Check every 2 seconds for next 20 seconds
        else:
            await asyncio.sleep(3)  # Check every 3 seconds after 30 seconds

    if not lesson_plans:
        print("❌ No lesson plans returned from assistant.")
        raise ValueError("No lesson plans generated")

    # Remove duplicates based on title
    unique_plans = []
    seen_titles = set()
    for plan in lesson_plans:
        title = plan.get('title', '').strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_plans.append(plan)
    
    print(f"[DEBUG] Generated {len(lesson_plans)} lesson plan(s), {len(unique_plans)} unique")
    return unique_plans

@app.post("/api/debate/lesson-plans", response_model=DebateLessonPlanResponse)
async def generate_debate_lesson_plans(request: DebateLessonPlanRequest, db: Session = Depends(get_db)):
    """Generate lesson plans based on classwide debate analysis with retry logic"""
    print(f"[DEBUG] generate_debate_lesson_plans called with: {request}")
    
    try:
        # Fetch classwide debate data
        debate_analysis = await get_classwide_debate_data(request.class_id, request.assignment_id, db)
        
        # Try to generate lesson plans with retry logic
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] Attempt {attempt + 1} of {max_retries}")
                lesson_plans = await debate_lesson_plan_generation_api(
                    request.class_id, 
                    request.assignment_id, 
                    request.grade,
                    debate_analysis
                )
                
                # Add unique IDs for frontend reference
                for i, plan in enumerate(lesson_plans):
                    plan['temp_id'] = f"temp_{request.class_id}_{request.assignment_id}_{i}_{int(time.time())}"
                
                return DebateLessonPlanResponse(
                    lesson_plans=lesson_plans,
                    success=True,
                    message=f"Successfully generated {len(lesson_plans)} lesson plan(s)"
                )
                
            except Exception as e:
                last_error = e
                print(f"[DEBUG] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Wait before retry
                    continue
        
        # If all retries failed
        print(f"[DEBUG] All attempts failed. Last error: {str(last_error)}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plans: {str(last_error)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {str(e)}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plans: {str(e)}")

async def rhetorical_device_progress_analysis_api(thread_id: str) -> dict:
    """
    Analyzes a rhetorical device practice session and returns structured progress data.
    Uses OpenAI Assistant with function calling for consistent output format.
    """
    assistant_id = os.getenv("RHETORICAL_DEVICE_ANALYZER_ID")
    logger.info(f"RHETORICAL_DEVICE_ANALYZER_ID: {assistant_id}")
    if not assistant_id:
        logger.error("RHETORICAL_DEVICE_ANALYZER_ID environment variable not configured")
        raise HTTPException(status_code=500, detail="RHETORICAL_DEVICE_ANALYZER_ID not configured")
    
    # Get the conversation from the thread
    messages = await client.beta.threads.messages.list(thread_id=thread_id, order="asc")
    
    # Format the conversation for analysis
    conversation = []
    for msg in messages.data:
        if msg.role in ["user", "assistant"]:
            conversation.append({
                "role": msg.role,
                "content": msg.content[0].text.value if msg.content else ""
            })
    
    # Create analysis prompt
    analysis_prompt = f"""
    Analyze the following rhetorical device practice session and provide structured feedback.
    
    You MUST use the provided tool 'rhetorical_device_analysis' and return ONLY a JSON object with this structure:
    {{
        "devices_practiced": [
            {{
                "device_name": "metaphor",
                "occurrences": 2,
                "effectiveness_comment": "Used metaphors effectively to create vivid imagery"
            }},
            {{
                "device_name": "rhetorical_question",
                "occurrences": 1,
                "effectiveness_comment": "Good use of rhetorical question to engage the audience"
            }}
        ],
        "overall_feedback": "You showed great improvement in using metaphors effectively. Your rhetorical questions were engaging and helped strengthen your argument. Continue practicing with anaphora to create more powerful emphasis.",
        "suggested_focus": ["anaphora", "parallelism", "hyperbole"]
    }}
    
    Focus on:
    - Which rhetorical devices the student actually attempted to use and how many times
    - How effectively they used each device
    - Constructive overall feedback on their progress
    - Suggested devices to practice next
    
    Conversation:
    {conversation}
    """
    
    # Create new thread for analysis
    analysis_thread = await client.beta.threads.create()
    await client.beta.threads.messages.create(
        thread_id=analysis_thread.id,
        role="user",
        content=analysis_prompt
    )
    
    # Run analysis with tool choice
    run = await client.beta.threads.runs.create(
        thread_id=analysis_thread.id,
        assistant_id=assistant_id,
        tool_choice={
            "type": "function",
            "function": {"name": "rhetorical_device_analysis"}
        }
    )
    
    # Wait for completion and handle tool call
    start_time = datetime.now()
    timeout_seconds = 120
    
    while True:
        run_status = await client.beta.threads.runs.retrieve(
            thread_id=analysis_thread.id,
            run_id=run.id
        )
        
        if run_status.status == 'requires_action':
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tc in tool_calls:
                if tc.function.name == "rhetorical_device_analysis":
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": tc.function.arguments
                    })
            
            if tool_outputs:
                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=analysis_thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
                
        elif run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise HTTPException(status_code=500, detail=f"Analysis run {run_status.status}")
        
        if (datetime.now() - start_time).total_seconds() > timeout_seconds:
            await client.beta.threads.runs.cancel(thread_id=analysis_thread.id, run_id=run.id)
            raise HTTPException(status_code=504, detail="Analysis timed out")
        
        await asyncio.sleep(1)
    
    # Extract the analysis result
    logger.info("Retrieving analysis messages from thread")
    analysis_messages = await client.beta.threads.messages.list(thread_id=analysis_thread.id)
    logger.info(f"Retrieved {len(analysis_messages.data)} messages from analysis thread")
    
    # Log the complete raw response for debugging
    logger.info("=== RAW GPT RESPONSE DEBUG ===")
    for i, msg in enumerate(analysis_messages.data):
        logger.info(f"RAW Message {i}: {msg}")
    logger.info("=== END RAW RESPONSE DEBUG ===")
    
    for i, msg in enumerate(analysis_messages.data):
        logger.info(f"Message {i}: role={msg.role}, content_count={len(msg.content) if msg.content else 0}")
        if msg.role == 'assistant':
            logger.info(f"Processing assistant message {i}")
            for j, content in enumerate(msg.content):
                logger.info(f"Content {j}: type={getattr(content, 'type', 'unknown')}")
                if getattr(content, 'type', None) == 'tool_calls':
                    tool_calls = getattr(content, 'tool_calls', [])
                    logger.info(f"Found {len(tool_calls)} tool calls")
                    for k, tool_call in enumerate(tool_calls):
                        logger.info(f"Tool call {k}: name={getattr(tool_call.function, 'name', 'unknown')}")
                        if getattr(tool_call.function, 'name', None) == "rhetorical_device_analysis":
                            try:
                                logger.info(f"Raw tool call arguments: {tool_call.function.arguments}")
                                result = json.loads(tool_call.function.arguments)
                                logger.info(f"Successfully parsed analysis result: {result}")
                                return result
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse tool call arguments as JSON: {e}")
                                logger.error(f"Raw arguments were: {tool_call.function.arguments}")
                                raise HTTPException(status_code=500, detail=f"Invalid JSON in analysis result: {str(e)}")
                elif getattr(content, 'type', None) == 'text':
                    text_value = getattr(content.text, 'value', 'no text value')
                    logger.info(f"Text content: {text_value[:200]}...")
                    
                    # Try to parse the text content as JSON (fallback)
                    try:
                        # Look for JSON-like content in the text
                        text_content = text_value.strip()
                        if text_content.startswith('{') and text_content.endswith('}'):
                            logger.info("Attempting to parse text content as JSON")
                            result = json.loads(text_content)
                            logger.info(f"Successfully parsed JSON from text content: {result}")
                            return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse text content as JSON: {e}")
                        continue
    
    logger.error("No analysis result found in assistant messages")
    raise HTTPException(status_code=500, detail="No analysis result found")

@app.post("/api/rhetorical-device-progress/", response_model=RhetoricalDeviceProgressResponse)
async def analyze_rhetorical_device_progress(request: RhetoricalDeviceProgressRequest):
    """
    Analyzes a rhetorical device practice session and returns structured progress data.
    This endpoint is called when the student clicks "Track My Progress".
    """
    return await _analyze_rhetorical_device_progress_impl(request)

@app.post("/api/rhetorical-device-progress", response_model=RhetoricalDeviceProgressResponse)
async def analyze_rhetorical_device_progress_no_slash(request: RhetoricalDeviceProgressRequest):
    """
    Same as above but without trailing slash to handle both URL formats
    """
    return await _analyze_rhetorical_device_progress_impl(request)

async def _analyze_rhetorical_device_progress_impl(request: RhetoricalDeviceProgressRequest):
    try:
        logger.info(f"Analyzing rhetorical device progress for thread {request.thread_id}")
        
        # Validate thread_id
        if not request.thread_id:
            raise HTTPException(status_code=400, detail="Thread ID is required")
        
        # Get analysis from OpenAI
        max_retries = 2
        for attempt in range(max_retries):
            try:
                analysis_data = await rhetorical_device_progress_analysis_api(request.thread_id)
                
                # Validate the response structure
                if not isinstance(analysis_data, dict):
                    raise ValueError("Analysis response is not a dictionary")
                
                required_fields = ["devices_practiced", "overall_feedback", "suggested_focus"]
                for field in required_fields:
                    if field not in analysis_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Validate devices_practiced structure
                devices_practiced = []
                for device in analysis_data["devices_practiced"]:
                    devices_practiced.append(DevicePracticed(**device))
                
                # Validate with Pydantic
                validated = RhetoricalDeviceProgressResponse(
                    success=True,
                    devices_practiced=devices_practiced,
                    overall_feedback=analysis_data["overall_feedback"],
                    suggested_focus=analysis_data["suggested_focus"],
                    message="Successfully analyzed rhetorical device progress"
                )
                
                return validated
                
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Analysis validation failed on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Failed to get valid analysis after {max_retries} attempts: {str(e)}"
                    )
                await asyncio.sleep(2)
        
        raise HTTPException(status_code=500, detail="Unexpected error in validation loop")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in analyze_rhetorical_device_progress: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return RhetoricalDeviceProgressResponse(
            success=False,
            devices_practiced=[],
            overall_feedback="",
            suggested_focus=[],
            message=f"Error analyzing progress: {str(e)}"
        )

# Add FRQ schemas and dynamic routing endpoint
class FRQRequest(BaseModel):
    subject: str
    essay_text: str
    prompt: str
    user_id: int
    assignment_id: Optional[int] = None

class GrammarAndSyntaxIssues(BaseModel):
    common_errors: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    suggested_fixes: List[str] = Field(default_factory=list)

class APLangArgumentResponse(BaseModel):
    overall_score: int
    scores: Dict[str, int]
    overall_feedback: str
    feedback: Dict[str, Any]
    excerpts: List[Dict[str, str]]
    revision_priorities: List[str]
    vocabulary_strength: Dict[str, Any]
    writing_persona: Dict[str, str]
    sophistication_suggestions: List[str]
    instructional_blind_spots: List[str]
    grammar_and_syntax_issues: GrammarAndSyntaxIssues
    rhetorical_appeals_used: Dict[str, str]
    next_instructional_focus: List[str]

class APLangRhetoricalResponse(BaseModel):
    overall_score: int
    scores: Dict[str, int]
    overall_feedback: str
    feedback: Dict[str, Any]
    excerpts: List[Dict[str, str]]
    rhetorical_line_of_reasoning: List[Dict[str, str]]
    rhetorical_devices_identified: List[Dict[str, Any]]
    revision_priorities: List[str]
    vocabulary_strength: Dict[str, Any]
    writing_persona: Dict[str, str]
    sophistication_suggestions: List[str]
    instructional_blind_spots: List[str]
    grammar_and_syntax_issues: GrammarAndSyntaxIssues
    rhetorical_appeals_used: Dict[str, str]
    next_instructional_focus: List[str]

class FRQResponse(BaseModel):
    success: bool
    subject: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

async def ap_lang_argument_analysis_api(essay_text: str, prompt: str) -> tuple[dict, str]:
    """
    Calls OpenAI Assistant for AP Language Argumentative essay analysis, handles tool call, and returns validated output with thread ID.
    Returns: (analysis_data, thread_id)
    """
    assistant_id = os.getenv("AP_LANG_FRQ_TUTOR_ID")
    if not assistant_id:
        raise HTTPException(status_code=500, detail="AP_LANG_FRQ_TUTOR_ID not set in environment variables.")

    # Load the AP Lang Argument rubric
    rubric = load_rubric("AP Lang Argument")
    
    # Compose prompt for AP Language Argumentative essay analysis
    analysis_prompt = f"""
You are an expert AP Language and Composition teacher specializing in argumentative essay analysis. 
Analyze the following student essay based on the EXACT AP Language Argumentative essay rubric provided below.

OFFICIAL AP LANG ARGUMENT RUBRIC:
{json.dumps(rubric, indent=2)}

Essay Prompt: {prompt}

Student Essay:
{essay_text}

You MUST use the provided tool 'ap_lang_argument_feedback' and return ONLY the structured JSON object. 

CRITICAL INSTRUCTIONS:
- Grade STRICTLY according to the rubric criteria provided above
- Use the exact point values specified in the rubric (Thesis: 0-1, Evidence/Commentary: 0-4, Sophistication: 0-1)
- Reference specific rubric criteria in your feedback
- Ensure your scoring aligns with the detailed criteria for each point level

Focus on:
- Thesis quality and defensibility (0-1 points) - does it present a defensible position?
- Evidence selection and commentary (0-4 points) - quality of evidence and explanation
- Sophistication of thought and understanding (0-1 points) - complex understanding of rhetorical situation
- Specific feedback for each rubric category based on the criteria
- Exemplar sentences that demonstrate strong evidence/commentary and sophistication
- Prioritized revision suggestions
- Analysis of rhetorical appeals (ethos, pathos, logos)
- Vocabulary and writing style assessment
- Grammar and syntax observations
- Teaching blind spots and next instructional focus

Provide detailed, constructive feedback that helps the student improve their argumentative writing skills according to AP standards.
"""

    # Create thread and send message
    thread = await client.beta.threads.create()
    thread_id = thread.id
    
    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=analysis_prompt
    )

    # Start run with tool choice
    run = await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        tool_choice={"type": "function", "function": {"name": "ap_lang_argument_feedback"}}
    )

    # Wait for completion and handle tool call
    start_time = datetime.now()
    timeout_seconds = 180
    
    while True:
        run_status = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        
        if run_status.status == 'requires_action':
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tc in tool_calls:
                if tc.function.name == "ap_lang_argument_feedback":
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": tc.function.arguments
                    })
            
            if tool_outputs:
                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
                
        elif run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
        
        if (datetime.now() - start_time).total_seconds() > timeout_seconds:
            await client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            raise HTTPException(status_code=504, detail="Request timed out")
        
        await asyncio.sleep(1)

    # Fetch thread messages and extract tool call output from assistant's message
    messages = await client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == 'assistant':
            for content in msg.content:
                if getattr(content, 'type', None) == 'tool_calls':
                    for tool_call in getattr(content, 'tool_calls', []):
                        if getattr(tool_call.function, 'name', None) == "ap_lang_argument_feedback":
                            return json.loads(tool_call.function.arguments), thread_id
            # Fallback: try to parse text as JSON if tool_calls not found
            for content in msg.content:
                if getattr(content, 'type', None) == 'text':
                    try:
                        return json.loads(content.text.value), thread_id
                    except Exception:
                        continue
    
    raise HTTPException(status_code=500, detail="No valid tool call output found in assistant's message.")

async def get_validated_ap_lang_argument_analysis(essay_text: str, prompt: str) -> dict:
    """
    Analyze an AP Language Argumentative essay, validate output, and retry up to 2 times if needed.
    Returns: {"data": validated_data, "thread_id": thread_id}
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"AP Lang analysis attempt {attempt + 1} of {max_retries}")
            
            response_data, thread_id = await ap_lang_argument_analysis_api(essay_text, prompt)
            print(f"DEBUG: Raw OpenAI response: {response_data}")
            print(f"DEBUG: Thread ID: {thread_id}")
            
            # The response should already be a dict from tool call arguments
            if isinstance(response_data, str):
                try:
                    data = json.loads(response_data)
                    print(f"DEBUG: Parsed JSON data: {data}")
                except Exception as e:
                    print(f"DEBUG: JSON parse error, raw string: {response_data}")
                    raise
            else:
                data = response_data
                print(f"DEBUG: Data is already a dict: {data}")
            
            # Validate with Pydantic
            validated = APLangArgumentResponse(**data)
            print(f"DEBUG: Pydantic validated data: {validated}")
            return {
                "data": validated.dict(),
                "thread_id": thread_id
            }
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid AP Lang analysis after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(2)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

async def handle_ap_lang_argument(request: FRQRequest) -> dict:
    """Handle AP Language Argumentative essay analysis"""
    try:
        logger.info(f"Processing AP Lang Argumentative essay for user {request.user_id}")
        
        # Validate required fields
        if not request.essay_text or not request.essay_text.strip():
            raise HTTPException(status_code=400, detail="Essay text is required")
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Essay prompt is required")
        
        # Get validated analysis (this will return both analysis data and thread_id)
        analysis_result = await get_validated_ap_lang_argument_analysis(
            request.essay_text, 
            request.prompt
        )
        
        return {
            "success": True,
            "subject": request.subject,
            "data": analysis_result["data"],
            "thread_id": analysis_result["thread_id"],
            "message": "Successfully analyzed AP Language Argumentative essay"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in handle_ap_lang_argument: {str(e)}")
        return {
            "success": False,
            "subject": request.subject,
            "data": None,
            "thread_id": None,
            "message": f"Error analyzing essay: {str(e)}"
        }

async def handle_ap_lang_rhetorical(request: FRQRequest) -> dict:
    """Handle AP Language Rhetorical Analysis essay analysis"""
    try:
        logger.info(f"Processing AP Lang Rhetorical Analysis essay for user {request.user_id}")
        
        # Validate required fields
        if not request.essay_text or not request.essay_text.strip():
            raise HTTPException(status_code=400, detail="Essay text is required")
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Essay prompt is required")
        
        # Get validated analysis
        analysis_data = await get_validated_ap_lang_rhetorical_analysis(
            request.essay_text, 
            request.prompt
        )
        
        return {
            "success": True,
            "subject": request.subject,
            "data": analysis_data,
            "message": "Successfully analyzed AP Language Rhetorical Analysis essay"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in handle_ap_lang_rhetorical: {str(e)}")
        return {
            "success": False,
            "subject": request.subject,
            "data": None,
            "message": f"Error analyzing essay: {str(e)}"
        }

async def handle_apush_dbq(request: FRQRequest) -> dict:
    """Handle APUSH DBQ analysis (placeholder for future implementation)"""
    return {
        "success": False,
        "subject": request.subject,
        "data": None,
        "message": "APUSH DBQ analysis not yet implemented"
    }

async def handle_ap_psych(request: FRQRequest) -> dict:
    """Handle AP Psychology FRQ analysis (placeholder for future implementation)"""
    return {
        "success": False,
        "subject": request.subject,
        "data": None,
        "message": "AP Psychology FRQ analysis not yet implemented"
    }

@app.post("/api/frq/submit/", response_model=FRQResponse)
async def submit_frq(request: FRQRequest):
    """
    Dynamic routing endpoint for FRQ submissions across different AP subjects.
    Routes to appropriate handler based on subject type.
    """
    try:
        logger.info(f"FRQ submission received for subject: {request.subject}")
        
        # Validate subject
        if not request.subject:
            raise HTTPException(status_code=400, detail="Subject is required")
        
        # Dynamic routing based on subject
        if request.subject == "ap_lang_argument":
            result = await handle_ap_lang_argument(request)
        elif request.subject == "ap_lang_rhetorical":
            result = await handle_ap_lang_rhetorical(request)
        elif request.subject == "apush_dbq":
            result = await handle_apush_dbq(request)
        elif request.subject == "ap_psych":
            result = await handle_ap_psych(request)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported subject: {request.subject}. Supported subjects: ap_lang_argument, ap_lang_rhetorical, apush_dbq, ap_psych"
            )
        
        return FRQResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in submit_frq: {str(e)}")
        return FRQResponse(
            success=False,
            subject=request.subject,
            data=None,
            message=f"Unexpected error: {str(e)}"
        )

async def ap_lang_rhetorical_analysis_api(essay_text: str, prompt: str) -> dict:
    """
    Calls OpenAI Assistant for AP Language Rhetorical Analysis essay analysis, handles tool call, and returns validated output.
    """
    assistant_id = os.getenv("AP_LANG_RHETORICAL_TUTOR_ID")
    if not assistant_id:
        raise HTTPException(status_code=500, detail="AP_LANG_RHETORICAL_TUTOR_ID not set in environment variables.")

    # Load the AP Lang Rhetorical Analysis rubric
    rubric = load_rubric("AP Lang Rhetorical Anal")
    
    # Compose prompt for AP Language Rhetorical Analysis essay analysis
    analysis_prompt = f"""
You are an expert AP Language and Composition teacher specializing in rhetorical analysis essay evaluation. 
Analyze the following student essay based on the EXACT AP Language Rhetorical Analysis essay rubric provided below.

OFFICIAL AP LANG RHETORICAL ANALYSIS RUBRIC:
{json.dumps(rubric, indent=2)}

Essay Prompt: {prompt}

Student Essay:
{essay_text}

You MUST use the provided tool 'ap_lang_rhetorical_feedback' and return ONLY the structured JSON object. 

CRITICAL INSTRUCTIONS:
- Grade STRICTLY according to the rubric criteria provided above
- Use the exact point values specified in the rubric (Thesis: 0-1, Evidence/Commentary: 0-4, Sophistication: 0-1)
- Reference specific rubric criteria in your feedback
- Ensure your scoring aligns with the detailed criteria for each point level

Focus on:
- Thesis quality and defensibility (0-1 points) - does it analyze the writer's rhetorical choices?
- Evidence selection and commentary on rhetorical choices (0-4 points) - explains how rhetorical choices contribute to argument/purpose
- Sophistication of thought and understanding (0-1 points) - complex understanding of rhetorical situation
- Specific feedback for each rubric category based on the criteria
- Rhetorical line of reasoning mapping (choice → effect → significance)
- Identification and analysis of rhetorical devices
- Prioritized revision suggestions
- Analysis of rhetorical appeals (ethos, pathos, logos)
- Vocabulary and writing style assessment
- Grammar and syntax observations
- Teaching blind spots and next instructional focus

Provide detailed, constructive feedback that helps the student improve their rhetorical analysis writing skills according to AP standards.
"""

    # Create thread and send message
    thread = await client.beta.threads.create()
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=analysis_prompt
    )

    # Start run with tool choice
    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        tool_choice={"type": "function", "function": {"name": "ap_lang_rhetorical_feedback"}}
    )

    # Wait for completion and handle tool call
    start_time = datetime.now()
    timeout_seconds = 180
    
    while True:
        run_status = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        
        if run_status.status == 'requires_action':
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tc in tool_calls:
                if tc.function.name == "ap_lang_rhetorical_feedback":
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": tc.function.arguments
                    })
            
            if tool_outputs:
                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                continue
                
        elif run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
        
        if (datetime.now() - start_time).total_seconds() > timeout_seconds:
            await client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
            raise HTTPException(status_code=504, detail="Request timed out")
        
        await asyncio.sleep(1)

    # Fetch thread messages and extract tool call output from assistant's message
    messages = await client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == 'assistant':
            for content in msg.content:
                if getattr(content, 'type', None) == 'tool_calls':
                    for tool_call in getattr(content, 'tool_calls', []):
                        if getattr(tool_call.function, 'name', None) == "ap_lang_rhetorical_feedback":
                            return json.loads(tool_call.function.arguments)
            # Fallback: try to parse text as JSON if tool_calls not found
            for content in msg.content:
                if getattr(content, 'type', None) == 'text':
                    try:
                        return json.loads(content.text.value)
                    except Exception:
                        continue
    
    raise HTTPException(status_code=500, detail="No valid tool call output found in assistant's message.")

async def get_validated_ap_lang_rhetorical_analysis(essay_text: str, prompt: str) -> dict:
    """
    Analyze an AP Language Rhetorical Analysis essay, validate output, and retry up to 2 times if needed.
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"AP Lang Rhetorical analysis attempt {attempt + 1} of {max_retries}")
            
            response_data = await ap_lang_rhetorical_analysis_api(essay_text, prompt)
            print(f"DEBUG: Raw OpenAI response: {response_data}")
            
            # The response should already be a dict from tool call arguments
            if isinstance(response_data, str):
                try:
                    data = json.loads(response_data)
                    print(f"DEBUG: Parsed JSON data: {data}")
                except Exception as e:
                    print(f"DEBUG: JSON parse error, raw string: {response_data}")
                    raise
            else:
                data = response_data
                print(f"DEBUG: Data is already a dict: {data}")
            
            # Validate with Pydantic
            validated = APLangRhetoricalResponse(**data)
            print(f"DEBUG: Pydantic validated data: {validated}")
            return validated.dict()
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Validation failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to get valid AP Lang Rhetorical analysis after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(2)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

# Add FRQ3 prompt generation endpoint
class FRQ3PromptRequest(BaseModel):
    class_id: int
    due_date: str
    due_time: str

class FRQ3PromptResponse(BaseModel):
    success: bool
    prompts: Optional[List[Dict[str, str]]] = None
    message: Optional[str] = None

async def exit_ticket_analysis_api(student_answer: str, question: str, correct_answer: str) -> dict:
    """
    Calls OpenAI Assistant for exit ticket analysis, handles tool call, and returns validated output.
    """
    logger.info("🚀 Starting exit_ticket_analysis_api")
    logger.info(f"📝 Student answer: {student_answer}")
    logger.info(f"❓ Question: {question}")
    logger.info(f"✅ Correct answer: {correct_answer}")
    
    assistant_id = os.getenv("EXIT_TICKET_STUDENT_ID")
    logger.info(f"🤖 Assistant ID: {assistant_id}")
    if not assistant_id:
        raise HTTPException(status_code=500, detail="EXIT_TICKET_STUDENT_ID not set in environment variables.")

    # Compose prompt for exit ticket analysis
    prompt = f"""
You are an expert educator analyzing a student's exit ticket response. You MUST use the analyze_exit_ticket tool function to provide structured analysis. Do NOT respond with plain text.

Question: {question}

Expected/Correct Answer: {correct_answer}

Student's Answer: {student_answer}

REQUIRED: Call the analyze_exit_ticket function with these exact parameters:
- correct: boolean (true/false)
- misconception_analysis: string 
- misconception_type: string
- conceptual_vs_procedural: string (must be "conceptual", "procedural", "both", or "not_applicable")
- intervention_suggestion: string
- error_severity: string (must be "minor", "moderate", "major", or "not_applicable")

You MUST use the tool function. Do not provide analysis as plain text.
"""

    # Create thread and send message
    logger.info("📝 Creating thread and sending message...")
    thread = await client.beta.threads.create()
    logger.info(f"🧵 Thread created: {thread.id}")
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    logger.info("✅ Message sent to thread")

    # Start run with tool choice
    logger.info("🏃 Starting run with tool choice...")
    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        tool_choice={"type": "function", "function": {"name": "analyze_exit_ticket"}},
        tools=[{"type": "function", "function": {"name": "analyze_exit_ticket"}}]
    )
    logger.info(f"🏃 Run created: {run.id}")

    # Wait for completion and handle tool call
    logger.info("⏳ Waiting for completion and handling tool calls...")
    start_time = datetime.now()
    timeout_seconds = 120
    while True:
        run_status = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        logger.info(f"📊 Run status: {run_status.status}")
        if run_status.status == 'requires_action':
            logger.info("🔧 Run requires action - processing tool calls...")
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            logger.info(f"🔧 Processing {len(tool_calls)} tool calls")
            tool_outputs = []
            for tc in tool_calls:
                logger.info(f"🛠️ Tool call: {tc.function.name}")
                logger.info(f"📝 Tool arguments: {tc.function.arguments}")
                
                # Parse the tool arguments (this is what the Assistant wants to call the tool with)
                try:
                    tool_args = json.loads(tc.function.arguments)
                    logger.info(f"✅ Parsed tool arguments: {tool_args}")
                    
                    # For analyze_exit_ticket, we should return the analysis result
                    # The Assistant has already done the analysis and provided it in the arguments
                    tool_output = json.dumps(tool_args)  # Return the analysis as JSON
                    logger.info(f"📤 Tool output: {tool_output}")
                    
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": tool_output
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse tool arguments: {e}")
                    tool_outputs.append({
                        "tool_call_id": tc.id,
                        "output": tc.function.arguments  # Fallback to raw arguments
                    })
            run = await client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            continue
        elif run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            raise HTTPException(status_code=500, detail=f"Run {run_status.status}")
        if (datetime.now() - start_time).total_seconds() > timeout_seconds:
            await client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
            raise HTTPException(status_code=504, detail="Request timed out")
        await asyncio.sleep(1)

    # Fetch thread messages and extract tool call output from assistant's message
    logger.info("📨 Fetching messages to extract tool call output...")
    messages = await client.beta.threads.messages.list(thread_id=thread.id)
    logger.info(f"📬 Found {len(messages.data)} messages")
    
    for i, msg in enumerate(messages.data):
        logger.info(f"📝 Message {i}: role={msg.role}, content_count={len(msg.content)}")
        if msg.role == 'assistant':
            logger.info(f"🤖 Processing assistant message {i}")
            for j, content in enumerate(msg.content):
                content_type = getattr(content, 'type', None)
                logger.info(f"   Content {j}: type={content_type}")
                
                if content_type == 'tool_calls':
                    logger.info("🔧 Found tool_calls content!")
                    tool_calls = getattr(content, 'tool_calls', [])
                    logger.info(f"   Tool calls count: {len(tool_calls)}")
                    for k, tool_call in enumerate(tool_calls):
                        tool_name = getattr(tool_call.function, 'name', None)
                        logger.info(f"   Tool call {k}: name={tool_name}")
                        if tool_name == "analyze_exit_ticket":
                            logger.info("✅ Found matching analyze_exit_ticket tool call!")
                            arguments = tool_call.function.arguments
                            logger.info(f"🔍 Raw tool call arguments: {arguments}")
                            try:
                                result = json.loads(arguments)
                                logger.info(f"✅ Extracted tool call result: {result}")
                                return result
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ Tool call JSON decode error: {e}")
                                continue
                
                elif content_type == 'text':
                    logger.info(f"📄 Found text content: {content.text.value[:200]}...")
            
            # Fallback: try to parse text as JSON if tool_calls not found
            logger.info("🔄 Trying fallback: parsing text content as JSON...")
            for content in msg.content:
                if getattr(content, 'type', None) == 'text':
                    try:
                        result = json.loads(content.text.value)
                        logger.info(f"✅ Parsed text as JSON: {result}")
                        return result
                    except Exception as e:
                        logger.info(f"⚠️ Text content not valid JSON: {e}")
                        continue
    
    logger.error("❌ No valid tool call output found in any assistant message")
    raise HTTPException(status_code=500, detail="No valid tool call output found in assistant's message.")

@app.post("/analyze-exit-ticket/", response_model=ExitTicketAnalysisResponse)
async def analyze_exit_ticket(request: ExitTicketAnalysisRequest):
    """
    Analyze a student's exit ticket response and provide educational insights.
    """
    try:
        logger.info(f"Analyzing exit ticket response for student {request.student_id}")
        
        # Validate required fields
        if not request.student_answer or not request.student_answer.strip():
            raise HTTPException(status_code=400, detail="Student answer is required")
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        if not request.correct_answer or not request.correct_answer.strip():
            raise HTTPException(status_code=400, detail="Correct answer is required")
        
        # Analyze with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"🎯 Exit ticket analysis attempt {attempt + 1}")
                analysis_data = await exit_ticket_analysis_api(
                    request.student_answer,
                    request.question,
                    request.correct_answer
                )
                logger.info(f"✅ Raw analysis_data received: {analysis_data}")
                logger.info(f"📊 Data type: {type(analysis_data)}")
                
                # Validate the response structure
                if not isinstance(analysis_data, dict):
                    raise ValueError("Analysis response is not a dictionary")
                
                required_fields = [
                    "correct", "misconception_analysis", "misconception_type",
                    "conceptual_vs_procedural", "intervention_suggestion", "error_severity"
                ]
                logger.info(f"🔍 Checking required fields in analysis_data...")
                missing_fields = []
                for field in required_fields:
                    if field not in analysis_data:
                        missing_fields.append(field)
                        logger.error(f"❌ Missing field: {field}")
                    else:
                        logger.info(f"✅ Found field: {field} = {analysis_data[field]}")
                
                if missing_fields:
                    raise ValueError(f"Missing required fields: {missing_fields}")
                
                # Return validated response
                return ExitTicketAnalysisResponse(
                    success=True,
                    correct=analysis_data["correct"],
                    misconception_analysis=analysis_data["misconception_analysis"],
                    misconception_type=analysis_data["misconception_type"],
                    conceptual_vs_procedural=analysis_data["conceptual_vs_procedural"],
                    intervention_suggestion=analysis_data["intervention_suggestion"],
                    error_severity=analysis_data["error_severity"],
                    message="Successfully analyzed exit ticket response"
                )
                
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Analysis failed on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Failed to get valid exit ticket analysis after {max_retries} attempts: {str(e)}"
                    )
                await asyncio.sleep(2)
        
        raise HTTPException(status_code=500, detail="Unexpected error in validation loop")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in analyze_exit_ticket: {str(e)}")
        return ExitTicketAnalysisResponse(
            success=False,
            correct=False,
            misconception_analysis="",
            misconception_type="",
            conceptual_vs_procedural="",
            intervention_suggestion="",
            error_severity="",
            message=f"Error analyzing exit ticket: {str(e)}"
        )


@app.post("/exit_ticket_classwide_api/", response_model=ExitTicketClasswideResponse)
async def exit_ticket_classwide_api(request: ExitTicketClasswideRequest, db: Session = Depends(get_db)):
    """
    Perform classwide analysis of exit ticket responses.
    Collects individual AI analyses and generates class insights using EXIT_TICKET_CLASS_ID assistant.
    """
    try:
        logger.info(f"Starting classwide exit ticket analysis for assignment {request.assignment_id}")
        
        # Get exit ticket data
        exit_ticket_query = text("""
            SELECT et.*, a.title as assignment_title, a.due_date
            FROM jarvis_app_exitticket et
            JOIN jarvis_app_assignment a ON et.assignment_id = a.id
            WHERE a.id = :assignment_id
        """)
        
        exit_ticket_result = db.execute(exit_ticket_query, {"assignment_id": request.assignment_id}).fetchone()
        
        if not exit_ticket_result:
            raise HTTPException(status_code=404, detail="Exit ticket not found")
        
        # Get all student responses with AI analysis
        responses_query = text("""
            SELECT 
                etr.response_text,
                etr.selected_options,
                etr.true_false_answer,
                etr.ai_analysis,
                etr.is_correct,
                etr.submitted_at,
                u.first_name,
                u.last_name
            FROM jarvis_app_exitticketresponse etr
            JOIN jarvis_app_customuser u ON etr.student_id = u.id
            WHERE etr.exit_ticket_id = :exit_ticket_id
            ORDER BY etr.submitted_at DESC
        """)
        
        responses = db.execute(responses_query, {"exit_ticket_id": exit_ticket_result.id}).fetchall()
        
        if not responses:
            raise HTTPException(status_code=400, detail="No student responses found")
        
        logger.info(f"Found {len(responses)} student responses")
        
        # Prepare data for AI assistant
        analysis_data = {
            "assignment_title": exit_ticket_result.assignment_title,
            "question": exit_ticket_result.question_text,
            "correct_answer": exit_ticket_result.correct_answer,
            "question_type": exit_ticket_result.question_type,
            "total_responses": len(responses),
            "individual_analyses": []
        }
        
        # Process each response
        for response in responses:
            response_data = {
                "student_name": f"{response.first_name} {response.last_name}",
                "response_text": response.response_text or "",
                "is_correct": response.is_correct,
                "submitted_at": str(response.submitted_at) if response.submitted_at else ""
            }
            
            # Include AI analysis if available
            if response.ai_analysis:
                response_data["ai_feedback"] = response.ai_analysis
            
            # Handle different question types
            if exit_ticket_result.question_type == 'multiple_choice' and response.selected_options:
                response_data["selected_options"] = response.selected_options
            elif exit_ticket_result.question_type == 'true_false' and response.true_false_answer is not None:
                response_data["true_false_answer"] = response.true_false_answer
            
            analysis_data["individual_analyses"].append(response_data)
        
        # Call EXIT_TICKET_CLASS_ID assistant
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        assistant_id = os.getenv("EXIT_TICKET_CLASS_ID")
        
        if not assistant_id:
            raise HTTPException(status_code=500, detail="EXIT_TICKET_CLASS_ID assistant not configured")
        
        logger.info(f"Calling OpenAI assistant {assistant_id}")
        
        # Create proper prompt for the assistant
        prompt = f"""You are a teaching assistant and co-teacher. Your job is to analyze the classwide exit ticket data and return your insights and analysis in exactly the format of the analyze_classwide_exit_tickets tool.

Please analyze the following exit ticket data for the class and provide comprehensive insights for instructional planning:

Assignment: {analysis_data['assignment_title']}
Question: {analysis_data['question']}
Total Students: {analysis_data['total_students']}
Total Responses: {len(analysis_data['individual_analyses'])}

Individual Student Analyses:
{json.dumps(analysis_data['individual_analyses'], indent=2)}

You MUST use the analyze_classwide_exit_tickets tool to provide your analysis. Focus on:
1. Overall mastery percentage and class readiness
2. Most common misconceptions across all students
3. Error severity distribution (minor, moderate, major)
4. Conceptual vs procedural error breakdown
5. Similar intervention needs for grouping students
6. Priority areas that need immediate attention
7. Whole-class reteaching recommendations
8. Question clarity issues if multiple students made similar errors
9. Success indicators from students who got it right"""

        # Create thread and message with error handling
        try:
            thread = client.beta.threads.create()
            
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # Create run with tool_choice to force using the tool
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                tool_choice={"type": "function", "function": {"name": "analyze_classwide_exit_tickets"}}
            )
        except Exception as e:
            logger.error(f"OpenAI API error during setup: {str(e)}")
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        # Poll for completion and handle tool outputs
        max_wait = 300  # 5 minutes
        start_time = time.time()
        tool_output_json = None
        
        try:
            while True:
                if time.time() - start_time > max_wait:
                    raise HTTPException(status_code=408, detail="AI analysis timeout")
                
                await asyncio.sleep(2)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                logger.info(f"Run status: {run.status}")
                
                if run.status == 'requires_action':
                    logger.info("🔧 Run requires action - processing tool calls...")
                    tool_calls = run.status.required_action.submit_tool_outputs.tool_calls
                    logger.info(f"🔧 Processing {len(tool_calls)} tool calls")
                    
                    tool_outputs = []
                    for tc in tool_calls:
                        logger.info(f"🛠️ Tool call: {tc.function.name}")
                        
                        if tc.function.name == "analyze_classwide_exit_tickets":
                            # Store the analysis result
                            tool_output_json = tc.function.arguments
                            logger.info(f"📤 Tool analysis received: {tool_output_json[:200]}...")
                            
                            tool_outputs.append({
                                "tool_call_id": tc.id,
                                "output": tool_output_json
                            })
                    
                    if tool_outputs:
                        run = await client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                        continue
                        
                elif run.status == 'completed':
                    break
                    
                elif run.status in ['failed', 'cancelled', 'expired']:
                    logger.error(f"Run failed with status: {run.status}")
                    raise HTTPException(status_code=500, detail=f"AI analysis failed: {run.status}")
                    
        except Exception as e:
            logger.error(f"OpenAI API error during polling: {str(e)}")
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        if tool_output_json:
            logger.info("✅ Tool output received, validating with Pydantic...")
            
            try:
                # Parse and validate response using Pydantic
                analysis_result = json.loads(tool_output_json)
                validated_analysis = ExitTicketClasswideAnalysisData(**analysis_result)
                
                # Calculate additional metrics
                correct_count = sum(1 for r in responses if r.is_correct)
                completion_rate = len(responses) / analysis_data.get("total_students", len(responses)) if analysis_data.get("total_students") else None
                
                logger.info(f"✅ Analysis validated successfully - Mastery: {validated_analysis.mastery_percentage}%")
                
                return ExitTicketClasswideResponse(
                    success=True,
                    analysis=validated_analysis,
                    total_responses=len(responses),
                    completion_rate=completion_rate,
                    thread_id=thread.id
                )
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Invalid tool output format: {str(e)}")
                logger.error(f"Raw tool output: {tool_output_json}")
                raise HTTPException(status_code=422, detail=f"Invalid AI analysis format: {str(e)}")
        
        else:
            logger.error("No tool output received from assistant")
            raise HTTPException(status_code=500, detail="AI analysis failed: No tool output received")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in exit_ticket_classwide_api: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error in classwide analysis: {str(e)}")


@app.post("/api/ap_lang_frq3_prompt_maker/", response_model=FRQ3PromptResponse)
async def ap_lang_frq3_prompt_maker(request: FRQ3PromptRequest):
    """
    Generate 3 AP Language FRQ3 Argumentative prompts using OpenAI Assistant.
    Uses the AP_LANG_ARGUMENT_PROMPT_MAKER_ID assistant with AP Lang Sample Prompts FRQ3 vector store.
    """
    try:
        logger.info(f"Generating FRQ3 prompts for class {request.class_id}")
        
        # Get the assistant ID from environment
        assistant_id = os.getenv("AP_LANG_ARGUMENT_PROMPT_MAKER_ID")
        if not assistant_id:
            raise HTTPException(status_code=500, detail="AP_LANG_ARGUMENT_PROMPT_MAKER_ID not configured")
        
        # Create a thread for this request
        thread = await client.beta.threads.create()
        
        # Send the prompt generation request
        prompt_message = """Please create three new prompts for AP Language and Composition FRQ 3 (Argumentative Essay) using the vector store as your context and reference. 

Each prompt should:
1. Be appropriate for AP Language and Composition students
2. Focus on argumentative writing skills
3. Include a clear thesis/position statement
4. Be engaging and relevant to current issues or timeless topics

Please format your response as a JSON array with exactly 3 prompts, each having:
- "title": A concise title/thesis statement (max 100 characters)
- "prompt": The full argumentative prompt text

Example format:
[
  {
    "title": "Technology's Impact on Human Connection",
    "prompt": "In an age where digital communication dominates our interactions..."
  },
  {
    "title": "The Role of Failure in Success",
    "prompt": "Many successful individuals attribute their achievements to lessons learned from failure..."
  },
  {
    "title": "Individual vs. Collective Responsibility",
    "prompt": "Consider the balance between personal accountability and societal obligation..."
  }
]"""
        
        # Add message to thread
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt_message
        )
        
        # Run the assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        
        # Wait for completion
        max_attempts = 30
        attempt = 0
        while attempt < max_attempts:
            run_status = await client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise HTTPException(status_code=500, detail=f"Assistant run failed: {run_status.status}")
            
            await asyncio.sleep(1)
            attempt += 1
        
        if attempt >= max_attempts:
            raise HTTPException(status_code=500, detail="Assistant run timed out")
        
        # Get the response
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = None
        
        for message in messages.data:
            if message.role == "assistant":
                assistant_message = message.content[0].text.value
                break
        
        if not assistant_message:
            raise HTTPException(status_code=500, detail="No response from assistant")
        
        # Parse the JSON response
        try:
            # Extract JSON from the response (in case there's extra text)
            json_start = assistant_message.find('[')
            json_end = assistant_message.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_text = assistant_message[json_start:json_end]
                prompts = json.loads(json_text)
            else:
                # Try to parse the entire response as JSON
                prompts = json.loads(assistant_message)
            
            # Validate the structure
            if not isinstance(prompts, list) or len(prompts) != 3:
                raise ValueError("Expected exactly 3 prompts")
            
            for prompt in prompts:
                if not isinstance(prompt, dict) or 'title' not in prompt or 'prompt' not in prompt:
                    raise ValueError("Each prompt must have 'title' and 'prompt' fields")
            
            logger.info(f"Successfully generated {len(prompts)} FRQ3 prompts")
            
            return FRQ3PromptResponse(
                success=True,
                prompts=prompts,
                message="Successfully generated FRQ3 prompts"
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse assistant response: {str(e)}")
            logger.error(f"Assistant response: {assistant_message}")
            raise HTTPException(status_code=500, detail="Failed to parse assistant response")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in ap_lang_frq3_prompt_maker: {str(e)}")
        return FRQ3PromptResponse(
            success=False,
            prompts=None,
            message=f"Error generating prompts: {str(e)}"
        )

async def get_mixed_groups_data(assignment_id: int, db: Session) -> List[dict]:
    """
    Optimized query to fetch only the 5 essential models for mixed grouping:
    1. FeedbackCategory - Rubric scores and detailed feedback
    2. CognitiveSkill - Analysis, synthesis, evaluation skills  
    3. InstructionalBlindSpot - Specific skill gaps
    4. NextInstructionalFocus - Learning targets
    5. WritingPersona - Learning style and approach
    """
    logger.info(f"Fetching mixed groups data for assignment {assignment_id}")
    
    # Efficient query with selective loading of only essential relationships
    submissions = db.query(Submission).filter(
        Submission.assignment_id == assignment_id
    ).options(
        selectinload(Submission.student)
    ).all()
    
    logger.info(f"Found {len(submissions)} submissions for mixed groups analysis")
    
    student_data = []
    for submission in submissions:
        if not submission.student:
            continue
            
        # 1. FeedbackCategory - Essential for rubric-based grouping
        feedback_categories = db.query(FeedbackCategory).filter_by(
            submission_id=submission.id
        ).all()
        
        rubric_scores = {}
        strengths = []
        areas_for_improvement = []
        
        for cat in feedback_categories:
            rubric_scores[cat.name] = cat.score
            if cat.strengths:
                strengths.extend(cat.strengths)
            if cat.areas_for_improvement:
                areas_for_improvement.extend(cat.areas_for_improvement)
        
        # 2. CognitiveSkill - Essential for collaborative grouping
        cognitive_skill = db.query(CognitiveSkill).filter_by(
            submission_id=submission.id
        ).first()
        
        cognitive_skills = {}
        if cognitive_skill:
            cognitive_skills = {
                "analysis": cognitive_skill.analysis,
                "synthesis": cognitive_skill.synthesis,
                "evaluation": cognitive_skill.evaluation
            }
        
        # 3. InstructionalBlindSpot - Essential for targeted grouping
        blind_spots = [
            bs.blind_spot for bs in 
            db.query(InstructionalBlindSpot).filter_by(submission_id=submission.id).all()
        ]
        
        # 4. NextInstructionalFocus - Essential for learning targets
        focus_areas = [
            nif.focus for nif in 
            db.query(NextInstructionalFocus).filter_by(submission_id=submission.id).all()
        ]
        
        # 5. WritingPersona - Essential for learning style matching
        writing_persona = db.query(WritingPersona).filter_by(
            submission_id=submission.id
        ).first()
        
        persona_info = {}
        if writing_persona:
            persona_info = {
                "type": writing_persona.type,
                "description": writing_persona.description
            }
        
        # Compile optimized student data structure
        student_info = {
            "student_name": submission.student.get_full_name() if hasattr(submission.student, 'get_full_name') else f"{submission.student.first_name} {submission.student.last_name}",
            "overall_score": submission.score,
            "rubric_scores": rubric_scores,
            "cognitive_skills": cognitive_skills,
            "strengths": strengths[:3],  # Top 3 strengths to limit tokens
            "areas_for_improvement": areas_for_improvement[:3],  # Top 3 areas to limit tokens
            "instructional_blind_spots": blind_spots,
            "focus_areas": focus_areas,
            "writing_persona": persona_info
        }
        
        student_data.append(student_info)
    
    logger.info(f"Compiled mixed groups data for {len(student_data)} students")
    return student_data

@app.post("/api/mixed-groups/create", response_model=dict)
async def create_mixed_groups_endpoint(request: MixedGroupsRequest, db: Session = Depends(get_db)):
    """
    Create mixed ability groups using optimized student data and OpenAI assistant.
    Follows robust patterns with tool calls, validation, and retry logic.
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Environment validation
            if not os.getenv("OPENAI_API_KEY"):
                raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
            
            assistant_id = os.getenv("TEACHING_ASSISTANT_ID")
            if not assistant_id:
                raise HTTPException(status_code=500, detail="TEACHING_ASSISTANT_ID not set")
            
            logger.info(f"Creating mixed groups for assignment {request.assignment_id}, attempt {attempt + 1}")
            
            # Get optimized student data (5 essential models only)
            student_data = await get_mixed_groups_data(request.assignment_id, db)
            
            if not student_data:
                raise HTTPException(
                    status_code=404, 
                    detail="No student submissions found for this assignment"
                )
            
            if len(student_data) < 6:  # Need at least 6 students for 2 groups of 3
                raise HTTPException(
                    status_code=422, 
                    detail="Need at least 6 student submissions to create mixed groups"
                )
            
            # Create comprehensive prompt for mixed groups
            prompt = f"""Please analyze the following student data and create mixed ability groups for collaborative learning. 

REQUIREMENTS:
- Create groups of 3-4 students each
- Mix students with complementary strengths and learning needs
- Focus on pairing stronger students with developing students
- Consider cognitive skills, writing personas, and instructional needs
- Ensure each group has diverse abilities for peer learning

STUDENT DATA:
{json.dumps(student_data, indent=2)}

RESPONSE FORMAT: You MUST use the create_mixed_groups tool and return a JSON response with this exact structure:
{{
  "mixed_groups": [
    {{
      "group_number": 1,
      "focus_area": "Primary learning focus area",
      "teaching_points": ["Specific teaching strategy 1", "Specific teaching strategy 2"],
      "students": [
        {{"name": "Student Name", "strength": "Key strength they bring"}},
        {{"name": "Student Name", "strength": "Key strength they bring"}}
      ],
      "group_strategy": "How this group should collaborate effectively"
    }}
  ]
}}

Create balanced groups that maximize peer learning opportunities."""

            # Create thread and run with tool choice
            thread = await client.beta.threads.create()
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            run = await client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                tool_choice={"type": "function", "function": {"name": "create_mixed_groups"}},
                tools=[{"type": "function", "function": {"name": "create_mixed_groups"}}]
            )
            
            # Handle completion with timeout and tool calls
            start_time = datetime.now()
            timeout_seconds = 180
            
            while True:
                run_status = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                
                if run_status.status == 'requires_action':
                    # Handle tool calls
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    for tc in tool_calls:
                        if tc.function.name == "create_mixed_groups":
                            tool_outputs.append({
                                "tool_call_id": tc.id,
                                "output": tc.function.arguments
                            })
                    
                    await client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    continue
                    
                elif run_status.status == 'completed':
                    break
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    raise HTTPException(status_code=500, detail=f"OpenAI run {run_status.status}")
                
                if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                    await client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                    raise HTTPException(status_code=504, detail="Request timed out")
                
                await asyncio.sleep(1)
            
            # Extract response from tool calls
            messages = await client.beta.threads.messages.list(thread_id=thread.id)
            response_data = None
            
            for msg in messages.data:
                if msg.role == 'assistant':
                    for content in msg.content:
                        if getattr(content, 'type', None) == 'tool_calls':
                            for tool_call in getattr(content, 'tool_calls', []):
                                if getattr(tool_call.function, 'name', None) == "create_mixed_groups":
                                    response_data = json.loads(tool_call.function.arguments)
                                    break
                        # Fallback to text parsing
                        elif getattr(content, 'type', None) == 'text' and not response_data:
                            try:
                                json_str = extract_json_from_response(content.text.value)
                                response_data = json.loads(json_str)
                            except Exception:
                                continue
            
            if not response_data:
                raise ValueError("No valid response found from OpenAI assistant")
            
            # Validate response structure
            try:
                validated = MixedGroupsResponse(**response_data)
                logger.info(f"Successfully created {len(validated.mixed_groups)} mixed groups")
                return {"success": True, "response": response_data}
                
            except ValidationError as ve:
                raise ValueError(f"Response validation failed: {str(ve)}")
            
        except HTTPException:
            # Re-raise HTTP exceptions immediately
            raise
            
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Validation/parsing failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to create valid mixed groups after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Failed to create mixed groups: {str(e)}")
            await asyncio.sleep(1)
    
    raise HTTPException(status_code=500, detail="Unexpected error in validation loop")

class ExitTicketAnalysisRequest(BaseModel):
    student_answer: str
    question: str
    correct_answer: str
    assignment_id: int
    student_id: int




