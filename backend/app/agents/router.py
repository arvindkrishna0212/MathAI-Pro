from typing import Literal, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from ..config import settings
from ..services.knowledge_base import KnowledgeBaseService
from ..services.feedback import FeedbackService
import re
import json
import os
from datetime import datetime

class RouteQuery(BaseModel):
    datasource: Literal["knowledge_base", "web_search", "llm_only"] = Field(description="Determines the primary data source for answering the question.")
    reason: str = Field(description="Explanation for the chosen data source.")
    confidence: float = Field(description="Confidence level in the routing decision (0.0 to 1.0)", ge=0.0, le=1.0)
    math_category: str = Field(description="The mathematical category of the question (e.g., ALGEBRA, GEOMETRY, CALCULUS, etc.)")
    complexity: Literal["basic", "intermediate", "advanced"] = Field(description="The estimated complexity level of the question")

class Router:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.1
        )

        self.parser = JsonOutputParser(pydantic_object=RouteQuery)
        self.kb_service = KnowledgeBaseService()
        self.feedback_service = FeedbackService()
        self.routing_history_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                               "data", "routing_history.json")
        self.routing_history = self._load_routing_history()
        
        # Enhanced prompt with more detailed instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an intelligent mathematical routing agent. Your task is to determine the best data source 
            to answer a mathematical question. Analyze the question carefully to determine:
            
            1. The mathematical category (ALGEBRA, GEOMETRY, CALCULUS, STATISTICS, NUMBER_THEORY, etc.)
            2. The complexity level (basic, intermediate, advanced)
            3. Whether it's likely to be in a standard knowledge base (common textbook problems)
            4. Whether it requires real-time or specialized information better found through web search
            5. Whether it's a novel or complex problem requiring direct LLM reasoning
            
            Choose between "knowledge_base" (for standard, well-known problems), "web_search" (for problems 
            requiring external information or verification), or "llm_only" (for novel reasoning tasks).
            
            Provide a confidence score (0.0 to 1.0) indicating your certainty in this routing decision.
            The output MUST be in JSON format according to the provided schema.
            """),
            ("human", "Question: {question}\n\nAdditional context: {context}\n\n{format_instructions}")
        ])
        
        self.router_chain = self.prompt | self.llm | self.parser

    def _load_routing_history(self) -> Dict[str, Any]:
        """Load routing history from file or initialize if not exists"""
        if os.path.exists(self.routing_history_file):
            try:
                with open(self.routing_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading routing history: {e}")
        
        # Initialize with empty history
        return {
            "routes": [],
            "stats": {
                "knowledge_base": {"count": 0, "success": 0},
                "web_search": {"count": 0, "success": 0},
                "llm_only": {"count": 0, "success": 0}
            }
        }
    
    def _save_routing_history(self):
        """Save routing history to file"""
        os.makedirs(os.path.dirname(self.routing_history_file), exist_ok=True)
        try:
            with open(self.routing_history_file, 'w') as f:
                json.dump(self.routing_history, f, indent=2)
        except Exception as e:
            print(f"Error saving routing history: {e}")
    
    def _update_routing_stats(self, datasource: str, success: bool = True):
        """Update routing statistics"""
        if datasource in self.routing_history["stats"]:
            self.routing_history["stats"][datasource]["count"] += 1
            if success:
                self.routing_history["stats"][datasource]["success"] += 1
            self._save_routing_history()
    
    async def _check_knowledge_base_similarity(self, question: str) -> float:
        """Check if the question is similar to anything in the knowledge base"""
        try:
            # Use the KB service to check for similar questions without retrieving full solutions
            similar_items = await self.kb_service.find_similar_questions(question)
            if similar_items and len(similar_items) > 0:
                # Return the highest similarity score
                return max([item.score for item in similar_items])
            return 0.0
        except Exception as e:
            print(f"Error checking KB similarity: {e}")
            return 0.0
    
    def _extract_math_keywords(self, question: str) -> list:
        """Extract mathematical keywords from the question"""
        # List of common mathematical terms to look for
        math_keywords = [
            "equation", "solve", "calculate", "integral", "derivative", "function",
            "graph", "plot", "area", "volume", "perimeter", "circumference", "radius",
            "diameter", "angle", "triangle", "square", "circle", "rectangle", "polygon",
            "algebra", "geometry", "calculus", "statistics", "probability", "theorem",
            "proof", "formula", "expression", "simplify", "factor", "expand", "matrix",
            "vector", "scalar", "limit", "sequence", "series", "differential", "integration"
        ]
        
        found_keywords = []
        for keyword in math_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', question.lower()):
                found_keywords.append(keyword)
        
        return found_keywords
    
    async def route(self, question: str) -> str:
        """Route a question to the appropriate data source based on content analysis"""
        print(f"Routing question: {question}")
        
        try:
            # Check for similar questions in knowledge base
            kb_similarity = await self._check_knowledge_base_similarity(question)
            math_keywords = self._extract_math_keywords(question)
            
            # Prepare context for the router
            context = {
                "kb_similarity": kb_similarity,
                "math_keywords": math_keywords,
                "routing_stats": self.routing_history["stats"]
            }
            
            format_instructions = self.parser.get_format_instructions()
            routing_decision = await self.router_chain.ainvoke({
                "question": question,
                "context": json.dumps(context),
                "format_instructions": format_instructions
            })
            
            # Log the routing decision
            self.routing_history["routes"].append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "decision": routing_decision['datasource'],
                "confidence": routing_decision['confidence'],
                "category": routing_decision['math_category'],
                "complexity": routing_decision['complexity'],
                "reason": routing_decision['reason']
            })

            # Update stats
            self._update_routing_stats(routing_decision['datasource'])
            self._save_routing_history()
            
            print(f"Routing decision: {routing_decision['datasource']} (confidence: {routing_decision['confidence']})")
            print(f"Math category: {routing_decision['math_category']}, Complexity: {routing_decision['complexity']}")
            print(f"Reason: {routing_decision['reason']}")

            # If KB similarity is very high, override to knowledge_base
            if kb_similarity > 0.9 and routing_decision['datasource'] != "knowledge_base":
                print(f"Overriding routing decision to knowledge_base due to high similarity: {kb_similarity}")
                return "knowledge_base"

            return routing_decision['datasource']
        except Exception as e:
            print(f"Error during routing: {e}")
            return "llm_only" # Fallback to LLM if routing fails
    
    async def update_routing_feedback(self, question: str, datasource: str, success: bool):
        """Update routing statistics based on feedback"""
        self._update_routing_stats(datasource, success)
        print(f"Updated routing stats for {datasource}: success={success}")
