from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
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
    ProcessDebateAnalysisResponse
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
from .models import Submission, FeedbackCategory, NextInstructionalFocus, InstructionalBlindSpot, WritingPersona, CustomUser
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
for origin in ["http://localhost:8000", "http://127.0.0.1:8000"]:
    if origin not in frontend_origins:
        frontend_origins.append(origin)

# Clean up any empty strings (in case env was empty)
frontend_origins = [origin.strip() for origin in frontend_origins if origin.strip()]

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

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DJANGO_API_BASE = os.getenv("DJANGO_API_BASE", "http://localhost:8000")

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

class LessonPlanMaterialsRequest(BaseModel):
    lesson_plan: dict
    lesson_plan_id: Union[str, int]  # Accept both string temp_ids and int db_ids

class ElaborationLessonPlanResponse(BaseModel):
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

