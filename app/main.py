from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
import os, time
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any, Tuple
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
    ElaborationModelSentencesResponse
)
from .utils import extract_json_from_response
from asgiref.sync import sync_to_async
import openai
import traceback
from datetime import datetime
from pprint import pprint


load_dotenv()


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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DJANGO_API_BASE = os.getenv("DJANGO_API_BASE", "http://localhost:8000")

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
        client.beta.threads.messages.create(
            thread_id=input.thread_id,
            role="user",
            content=input.message
        )

        # Create and monitor run with the teacher assistant
        run = client.beta.threads.runs.create(
            thread_id=input.thread_id,
            assistant_id=teacher_assistant_id
        )

        timeout = 60
        start_time = time.time()
        while True:
            status = client.beta.threads.runs.retrieve(run.id, thread_id=input.thread_id).status
            if status == "completed":
                break
            elif status in ["failed", "cancelled"]:
                raise HTTPException(status_code=400, detail=f"Run status: {status}")
            elif time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="Assistant timed out.")
            await asyncio.sleep(1)

        # Get messages
        messages = client.beta.threads.messages.list(thread_id=input.thread_id, order="asc")
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

@app.post("/initialize_chat")
async def initialize_chat(input: InitChatInput):
    try:
        logger.info(f"Initializing chat with input: {input}")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

        teacher_assistant_id = os.getenv("TEACHER_CHAT_BUDDY_ID")
        if not teacher_assistant_id:
            logger.error("TEACHER_CHAT_BUDDY_ID not set")
            raise HTTPException(status_code=500, detail="TEACHER_CHAT_BUDDY_ID not set")

        # Create thread
        thread = client.beta.threads.create()
        thread_id = thread.id
        logger.info(f"Created thread with ID: {thread_id}")

        if input.assignment_id:
            # Get submission feedback from Django API
            submission_data = await get_submission_feedback(input.assignment_id)
            if submission_data:
                context_text = f"Here is the submission feedback data:\n{submission_data}"
            else:
                context_text = "No submission feedback available."
        else:
            context_text = "No assignment context available."

        # Initialize thread with context
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"Using this submission feedback data, collaborate as a thought partner and assistant with the teacher.\n{context_text}"
        )

        # Create initial run with the teacher assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=teacher_assistant_id
        )

        # Wait for the initial run to complete
        timeout = 60
        start_time = time.time()
        while True:
            status = client.beta.threads.runs.retrieve(run.id, thread_id=thread_id).status
            if status == "completed":
                break
            elif status in ["failed", "cancelled"]:
                raise HTTPException(status_code=400, detail=f"Run status: {status}")
            elif time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="Assistant timed out.")
            await asyncio.sleep(1)

        # Get the initial assistant message
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        initial_history = [
            {
                "role": msg.role,
                "text": msg.content[0].text.value
            }
            for msg in messages.data if msg.role in ["user", "assistant"]
        ]

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
    """
    if not assignment_id:
        return None
    
    url = f"{DJANGO_API_BASE}/api/submission_feedback/{assignment_id}/"
    api_key = os.getenv("INTERNAL_API_KEY")
    headers = {"X-API-KEY": api_key}
    
    # Add debug logging
    logger.info(f"Fetching submission feedback from {url}")
    logger.info(f"API Key present: {bool(api_key)}")
    logger.info(f"API Key length: {len(api_key) if api_key else 0}")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            logger.info(f"Response status: {resp.status_code}")
            logger.info(f"Response headers: {dict(resp.headers)}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching submission feedback for assignment {assignment_id}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error fetching submission feedback for assignment {assignment_id}: {str(e)}")
    
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
        run_status = client.beta.threads.runs.retrieve(
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
                run = client.beta.threads.runs.submit_tool_outputs(
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
    messages = client.beta.threads.messages.list(thread_id=thread_id)
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
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            # Create run with tool choice
            run = client.beta.threads.runs.create(
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
        thread = client.beta.threads.create()
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
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=system_message
        )
        logger.info("System message sent to thread.")

        # Create and run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        logger.info(f"Run created with ID: {run.id}")

        # Await completion using asyncio
        start_time = time.time()
        timeout = 90
        while True:
            run_status = client.beta.threads.runs.retrieve(
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
        messages = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
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
    client.beta.threads.messages.create(
        thread_id=input.thread_id,
        role="user",
        content=input.message
    )

    # Create and run the assistant
    run = client.beta.threads.runs.create(
        thread_id=input.thread_id,
        assistant_id=assistant_id
    )

    # Await completion using asyncio
    start_time = time.time()
    timeout = 90
    while True:
        run_status = client.beta.threads.runs.retrieve(
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
    messages = client.beta.threads.messages.list(thread_id=input.thread_id, order="desc")
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
        thread = await sync_to_async(openai.beta.threads.create)()
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


        message = await sync_to_async(openai.beta.threads.messages.create)(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        print("✅ Initial message sent")

        # Create run
        print("🚀 Starting assistant run...")
        run = await sync_to_async(openai.beta.threads.runs.create)(
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
            run_status = await sync_to_async(openai.beta.threads.runs.retrieve)(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == 'requires_action':
                print("🛠️ Processing function call...")
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = [{"tool_call_id": tc.id, "output": tc.function.arguments} for tc in tool_calls]
                
                run = await sync_to_async(openai.beta.threads.runs.submit_tool_outputs)(
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
        messages = await sync_to_async(openai.beta.threads.messages.list)(thread_id=thread.id)
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
        thread = await sync_to_async(openai.beta.threads.create)()
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
        await sync_to_async(openai.beta.threads.messages.create)(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        print("✅ Feedback prompt sent")

        # Start a run with the same assistant
        print("🚀 Starting assistant run for feedback...")
        run = await sync_to_async(openai.beta.threads.runs.create)(
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
            run_status = await sync_to_async(openai.beta.threads.runs.retrieve)(
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
                    run = await sync_to_async(openai.beta.threads.runs.submit_tool_outputs)(
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
        messages = await sync_to_async(openai.beta.threads.messages.list)(thread_id=thread.id)
        
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
        thread = await sync_to_async(openai.beta.threads.create)()
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
        await sync_to_async(openai.beta.threads.messages.create)(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        print("✅ Summary prompt sent")

        # Start a run with the same assistant
        print("🚀 Starting assistant run for summary...")
        run = await sync_to_async(openai.beta.threads.runs.create)(
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
            run_status = await sync_to_async(openai.beta.threads.runs.retrieve)(
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
                    run = await sync_to_async(openai.beta.threads.runs.submit_tool_outputs)(
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
        messages = await sync_to_async(openai.beta.threads.messages.list)(thread_id=thread.id)
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

        thread = await sync_to_async(openai.beta.threads.create)()
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
        
        await sync_to_async(openai.beta.threads.messages.create)(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        
        run = await sync_to_async(openai.beta.threads.runs.create)(
            thread_id=thread.id,
            assistant_id=assistant_id,
            tool_choice={"type": "function", "function": {"name": "generate_model_sentences"}}
        )

        start_time = datetime.now()
        timeout_seconds = 180
        tool_response_json = None  # 🆕 storage

        while True:
            run_status = await sync_to_async(openai.beta.threads.runs.retrieve)(
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

                run = await sync_to_async(openai.beta.threads.runs.submit_tool_outputs)(
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

