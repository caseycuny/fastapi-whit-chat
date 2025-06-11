from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os, time
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse
import logging
import httpx

app = FastAPI()

# CORS for local dev or frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
