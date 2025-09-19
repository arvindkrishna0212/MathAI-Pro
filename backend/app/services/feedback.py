from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os
from ..models.schemas import FeedbackRequest, FeedbackResponse, MathSolution
from ..config import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class FeedbackService:
    def __init__(self):
        self.feedbacks: List[FeedbackResponse] = [] # In-memory storage for demonstration
        self.feedback_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'feedback_data.json')
        self.load_feedback_data()
        
        # Initialize LLM for feedback processing
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.2
        )

        
        # Feedback learning prompt
        self.feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that helps improve mathematical solutions based on user feedback. Your goal is to analyze the feedback and suggest specific improvements to the solution."),
            ("human", "Original solution: {original_solution}\n\nUser feedback: {feedback}\n\nPlease analyze this feedback and suggest specific improvements to the solution.")
        ])
        
        self.feedback_chain = self.feedback_prompt | self.llm
        
    def load_feedback_data(self):
        """Load feedback data from file if it exists"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.feedbacks.append(FeedbackResponse(**item))
                print(f"Loaded {len(self.feedbacks)} feedback items from file")
        except Exception as e:
            print(f"Error loading feedback data: {e}")
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            
            # Convert to dict for JSON serialization
            data = [fb.model_dump() for fb in self.feedbacks]
            
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, default=str)
            print(f"Saved {len(self.feedbacks)} feedback items to file")
        except Exception as e:
            print(f"Error saving feedback data: {e}")
        
    async def record_feedback(self, feedback_request: FeedbackRequest) -> FeedbackResponse:
        # In a real application, this would save to a database (SQL, NoSQL, etc.)
        # For simplicity, we're storing in memory and a JSON file.
        feedback_id = f"fb_{len(self.feedbacks) + 1}"
        new_feedback = FeedbackResponse(
            feedback_id=feedback_id,
            solution_id=feedback_request.solution_id,
            rating=feedback_request.rating,
            comment=feedback_request.comment,
            improvements=feedback_request.improvements,
            timestamp=datetime.utcnow()
        )
        self.feedbacks.append(new_feedback)
        print(f"Feedback recorded: {new_feedback.feedback_id}")
        
        # Save to file
        self.save_feedback_data()
        
        # If human-in-the-loop is enabled, process the feedback
        if settings.ENABLE_HUMAN_FEEDBACK and feedback_request.rating < 4:
            await self.process_feedback(new_feedback)
        
        return new_feedback
        
    async def get_feedback_by_solution_id(self, solution_id: str) -> List[FeedbackResponse]:
        return [fb for fb in self.feedbacks if fb.solution_id == solution_id]
    
    async def process_feedback(self, feedback: FeedbackResponse) -> None:
        """Process feedback using human-in-the-loop mechanism"""
        print(f"Processing feedback: {feedback.feedback_id}")
        
        # In a real application, you would retrieve the original solution from a database
        # For this example, we'll use a placeholder
        original_solution = f"Solution ID: {feedback.solution_id}"
        
        # Prepare feedback text
        feedback_text = f"Rating: {feedback.rating}/5\n"
        if feedback.comment:
            feedback_text += f"Comment: {feedback.comment}\n"
        if feedback.improvements:
            feedback_text += f"Areas for improvement: {', '.join(feedback.improvements)}"
        
        try:
            # Use LLM to analyze feedback and suggest improvements
            response = await self.feedback_chain.ainvoke({
                "original_solution": original_solution,
                "feedback": feedback_text
            })
            
            print(f"Feedback analysis: {response.content}")
            
            # In a real application, you would use this analysis to update your model or knowledge base
            # For example, you could store common patterns of feedback and use them to improve future responses
            
            # This is where you would implement DSPy for feedback-based learning
            # Example pseudocode:
            # dspy_module = MathSolverModule()
            # dspy_module.update_from_feedback(original_solution, feedback_text, response.content)
            # dspy_module.save_to_disk()
            
        except Exception as e:
            print(f"Error processing feedback: {e}")
    
    async def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        if not self.feedbacks:
            return {"total": 0, "average_rating": 0}
        
        total = len(self.feedbacks)
        avg_rating = sum(fb.rating for fb in self.feedbacks) / total
        
        # Count by improvement areas
        improvement_areas = {}
        for fb in self.feedbacks:
            if fb.improvements:
                for area in fb.improvements:
                    if area in improvement_areas:
                        improvement_areas[area] += 1
                    else:
                        improvement_areas[area] = 1
        
        return {
            "total": total,
            "average_rating": round(avg_rating, 2),
            "improvement_areas": improvement_areas
        }
        
    async def record_improvement(self, solution_id: str, original_confidence: float, 
                               improved_confidence: float, improvement_type: str) -> None:
        """Record when a solution has been automatically improved
        
        Args:
            solution_id: The ID of the solution that was improved
            original_confidence: The confidence score of the original solution
            improved_confidence: The confidence score of the improved solution
            improvement_type: The type of improvement (e.g., 'automated_feedback_based', 'human_feedback')
        """
        # In a real application, this would be stored in a database
        # For now, we'll just log it and add a special feedback entry
        print(f"Solution improvement recorded: {solution_id} - {improvement_type}")
        print(f"Confidence change: {original_confidence} -> {improved_confidence}")
        
        # Create a system feedback entry to track this improvement
        feedback_id = f"sys_imp_{len(self.feedbacks) + 1}"
        system_feedback = FeedbackResponse(
            feedback_id=feedback_id,
            solution_id=solution_id,
            rating=5,  # Assume system improvements are good
            comment=f"Automatic improvement via {improvement_type}",
            improvements=[improvement_type],
            timestamp=datetime.utcnow(),
            metadata={
                "original_confidence": original_confidence,
                "improved_confidence": improved_confidence,
                "confidence_delta": improved_confidence - original_confidence
            }
        )
        
        self.feedbacks.append(system_feedback)
        self.save_feedback_data()