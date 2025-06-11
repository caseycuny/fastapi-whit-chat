from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    teacher_chat_buddy_id: str
    django_api_url: str = "http://localhost:8000"  # Your Django API URL
    
    class Config:
        env_file = ".env"

settings = Settings()
