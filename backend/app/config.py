import os
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    MCP_API_KEY: str = os.getenv("MCP_API_KEY", "")
    MCP_ENDPOINT: str = os.getenv("MCP_ENDPOINT", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    ALLOWED_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000"
    ]

    # Vector DB
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME: str = "math_knowledge_base"

    # Model Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Guardrails Configuration
    BLOCKED_KEYWORDS: List[str] = [
        "hack", "exploit", "bypass", "illegal", "malicious",
        "porn", "gambling", "drugs", "weapon", "violence"
    ]
    ALLOWED_DOMAINS: List[str] = [
        "mathematics", "math", "algebra", "geometry", "calculus",
        "statistics", "trigonometry", "arithmetic", "number theory",
        "discrete mathematics", "linear algebra", "differential equations"
    ]

    # Feedback Configuration
    FEEDBACK_THRESHOLD: float = 0.7
    MIN_FEEDBACK_COUNT: int = 5

    # Human-in-the-loop Configuration
    ENABLE_HUMAN_FEEDBACK: bool = True
    FEEDBACK_LEARNING_RATE: float = 0.1

    class Config:
        # Use absolute path to .env file
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

settings = Settings()
