from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from .agents.math_agent import MathAgent
from .agents.router import Router
from .agents.guardrails import input_guardrail, output_guardrail, GuardrailResult
from .models.schemas import MathQuestion, AgentResponse, FeedbackRequest, FeedbackResponse, WebSearchResult
from .services.feedback import FeedbackService
from .services.knowledge_base import KnowledgeBaseService
from .services.web_search import WebSearchService
from fastapi.middleware.cors import CORSMiddleware 
from contextlib import asynccontextmanager

math_agent = None
router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global math_agent, router
    math_agent = MathAgent(feedback_service)
    router = Router()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Math Routing Agent API",
    description="API for an AI-enabled Math Tutor with RAG, routing, and feedback mechanisms with human-in-the-loop learning.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services and agents
feedback_service = FeedbackService()

# Initialize knowledge base service with error handling
try:
    kb_service = KnowledgeBaseService()
except Exception as e:
    print(f"Error initializing KnowledgeBaseService: {str(e)}")
    print("The application will start, but knowledge base functionality will be limited.")
    kb_service = None
web_search_service = WebSearchService()

# Initialize agents with required services
# Handle the case where kb_service is None due to connection errors
if kb_service is None:
    # Create a dummy kb_service that provides minimal functionality
    from .services.knowledge_base import KnowledgeBaseServiceDummy
    kb_service = KnowledgeBaseServiceDummy()
    print("Using dummy knowledge base service with limited functionality")

@app.post("/solve", response_model=AgentResponse)
async def solve_math_question(question: MathQuestion):
    """Solve a mathematical question using the Math Routing Agent"""
    # Input Guardrail
    input_check = input_guardrail(question.question)
    if not input_check.is_safe:
        raise HTTPException(status_code=400, detail=input_check.message)
        
    # Route the question
    datasource = await router.route(question.question)
    print(f"Router decided to use {datasource} for question: {question.question}")
    
    # Solve based on routing decision
    # The enhanced implementation now supports explicit routing
    if datasource == "knowledge_base":
        # Attempt KB retrieval directly, if fails, then fall back to MathAgent's default flow
        kb_solution = await math_agent.kb_service.retrieve_solution(question.question)
        if kb_solution:
            print(f"Found solution in knowledge base with confidence {kb_solution.solution.confidence}")
            response = AgentResponse(
                solution=kb_solution.solution, 
                feedback_prompt="This solution was retrieved from our knowledge base. Was it helpful?"
            )
        else:
            print("No solution found in knowledge base, falling back to full agent flow")
            response = await math_agent.solve_question(question) # Fallback to full agent flow
    elif datasource == "web_search":
        # Force web search first by setting a flag
        print("Explicitly routing to web search")
        question.context = "FORCE_WEB_SEARCH"  # Signal to prioritize web search
        response = await math_agent.solve_question(question)
    else: # llm_only or default
        print("Using default agent flow")
        response = await math_agent.solve_question(question)
    
    # Output Guardrail
    output_check = output_guardrail(response.solution.explanation + " " + response.solution.final_answer)
    if not output_check.is_safe:
        raise HTTPException(status_code=500, detail=output_check.message)
        
    return response
    
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for a solution to enable human-in-the-loop learning"""
    print(f"Received feedback for solution {feedback.solution_id}: Rating {feedback.rating}")
    
    # Store feedback and trigger learning process
    stored_feedback = await feedback_service.record_feedback(feedback)
    
    return stored_feedback

@app.get("/feedback/stats")
async def get_feedback_statistics():
    """Get statistics about collected feedback for human-in-the-loop analysis"""
    stats = await feedback_service.get_feedback_statistics()
    return stats

@app.get("/search", response_model=List[WebSearchResult])
async def search(query: str = Query(..., min_length=3)):
    """Perform a web search for mathematical information"""
    # Input Guardrail
    input_check = input_guardrail(query)
    if not input_check.is_safe:
        raise HTTPException(status_code=400, detail=input_check.message)
    
    results = await web_search_service.search(query)
    return results

@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/guardrail/check")
async def check_guardrail(text: str = Query(..., min_length=1)):
    """Check if text passes the input guardrail"""
    input_check = input_guardrail(text)
    return {
        "is_safe": input_check.is_safe,
        "message": input_check.message,
        "category": input_check.category,
        "confidence": input_check.confidence
    }

@app.get("/kb/status")
async def knowledge_base_status():
    """Get status of the knowledge base"""
    # In a real implementation, you would query the vector DB for stats
    # For now, we'll return a placeholder
    return {
        "status": "active",
        "collection": settings.QDRANT_COLLECTION_NAME,
        "url": settings.QDRANT_URL
    }
