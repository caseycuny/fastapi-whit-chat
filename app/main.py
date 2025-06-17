from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os, time
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse
import logging
import httpx
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import json
from pydantic import ValidationError
import asyncio
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()

# CORS for local dev or frontend calls
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:8000".split(","))

if "http://localhost:8000" not in frontend_origins:
    frontend_origins.append("http://localhost:8000")

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

        # Create message
        client.beta.threads.messages.create(
            thread_id=input.thread_id,
            role="user",
            content=input.message
        )

        # Create and monitor run
        run = client.beta.threads.runs.create(
            thread_id=input.thread_id,
            assistant_id=input.assistant_id
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
            time.sleep(1)

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

        # Create thread
        thread = client.beta.threads.create()
        thread_id = thread.id
        logger.info(f"Created thread with ID: {thread_id}")

        if input.assignment_id:
            # Get trend data from Django API
            trend_data = await get_trend_data(input.assignment_id)
            context_text = f"Here is the class trend data:\n{trend_data}"
        else:
            context_text = "No assignment context available."

        # Initialize thread with context
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"Using this class trend data, collaborate as a thought partner and assistant with the teacher.\n{context_text}"
        )

        return {
            "thread_id": thread_id,
            "history": []  # Empty history for new thread
        }

    except Exception as e:
        logger.error(f"Error in initialize_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
