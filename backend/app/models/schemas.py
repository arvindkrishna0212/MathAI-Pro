from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class QuestionType(str, Enum):
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    STATISTICS = "statistics"
    TRIGONOMETRY = "trigonometry"
    ARITHMETIC = "arithmetic"
    OTHER = "other"

class MathQuestion(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    context: Optional[str] = None
    difficulty: Optional[str] = "medium"
    
class Step(BaseModel):
    step_number: int
    explanation: str
    formula: Optional[str] = None
    result: Optional[str] = None
    
class MathSolution(BaseModel):
    id: Optional[str] = None
    question: str
    steps: List[Step]
    final_answer: str
    explanation: str
    category: QuestionType
    confidence: float = Field(ge=0.0, le=1.0)
    
class FeedbackRequest(BaseModel):
    solution_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    improvements: Optional[List[str]] = None
    
class FeedbackResponse(BaseModel):
    feedback_id: str
    solution_id: str
    rating: int
    comment: Optional[str] = None
    improvements: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentResponse(BaseModel):
    solution: MathSolution
    feedback_prompt: str
    
class KnowledgeBaseItem(BaseModel):
    id: str
    question: str
    solution: Optional[MathSolution] = None
    embedding: Optional[List[float]] = None
    score: Optional[float] = None
    
class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    
class WebSearchRequest(BaseModel):
    query: str
    
class WebSearchResponse(BaseModel):
    results: List[WebSearchResult]
    
class FeedbackMechanism(BaseModel):
    agent_id: str
    user_id: str
    feedback_score: float
    feedback_text: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
