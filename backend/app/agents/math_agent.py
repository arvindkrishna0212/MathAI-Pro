from typing import List, Dict, Any, Optional, Union
import os
import json
import uuid
from datetime import datetime
from ..models.schemas import MathQuestion, MathSolution, Step, QuestionType, AgentResponse
from ..services.knowledge_base import KnowledgeBaseService
from ..services.web_search import WebSearchService
from ..services.feedback import FeedbackService
from ..utils.math_parser import MathParser
from ..config import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Try to import DSPy if available
try:
    import dspy
    import uuid
    from datetime import datetime
    DSPY_AVAILABLE = True

    # Configure DSPy with Groq LM using OpenAI-compatible interface
    try:
        # Use a currently supported model - llama-3.3-70b-versatile is production-ready
        # If that's not available, fallback to llama-3.1-8b-instant
        supported_models = [
            'llama-3.3-70b-versatile',  # Production model with good capabilities
            'llama-3.1-8b-instant',     # Backup production model
            'gemma2-9b-it'              # Another backup option
        ]
        
        # Try each model until one works
        lm = None
        for model_name in supported_models:
            try:
                print(f"Attempting to configure DSPy with model: {model_name}")
                lm = dspy.LM(
                    model=model_name,
                    api_key=settings.GROQ_API_KEY,
                    api_base='https://api.groq.com/openai/v1'  # Groq's OpenAI-compatible endpoint
                )
                # Configure DSPy with this model
                dspy.configure(lm=lm)
                print(f"DSPy successfully configured with {model_name} model")
                break
            except Exception as e:
                print(f"Failed to configure DSPy with {model_name}: {e}")
                lm = None
                continue
        
        if lm is None:
            print("Failed to configure DSPy with any supported model")
            DSPY_AVAILABLE = False
            
    except Exception as e:
        print(f"Failed to configure DSPy with Groq: {e}")
        DSPY_AVAILABLE = False

except ImportError:
    DSPY_AVAILABLE = False
    print("DSPy not available. Feedback-based learning will be limited.")

# Define the Pydantic output schema for the LLM
class MathSolutionSchema(BaseModel):
    question: str = Field(description="The original math question.")
    steps: List[Dict[str, Any]] = Field(description="A list of detailed, step-by-step explanations to solve the math question.")
    final_answer: str = Field(description="The final numerical or symbolic answer to the math question.")
    explanation: str = Field(description="A concise summary of the solution process.")
    category: str = Field(description="The category of the math question (e.g., ALGEBRA, CALCULUS).")
    confidence: float = Field(description="A confidence score (0.0-1.0) in the correctness of the solution.")

# Define DSPy module if available
if DSPY_AVAILABLE:
    class MathSolverModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_solution = dspy.ChainOfThought("question, context -> solution, steps, final_answer, explanation, category, confidence")
            
            # Load feedback data if available
            self.feedback_data = []
            self.feedback_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'dspy_feedback.json')
            self.load_feedback_data()
            
            # Initialize optimization metrics for tracking improvement
            self.optimization_metrics = {
                "total_optimizations": 0,
                "avg_confidence_improvement": 0.0,
                "learning_rate": 0.1  # Default learning rate
            }
        
        def load_feedback_data(self, external_feedback=None):
            try:
                if external_feedback:
                    # Convert external feedback to DSPy format
                    for fb in external_feedback:
                        if hasattr(fb, 'model_dump'):
                            fb_dict = fb.model_dump()
                        else:
                            fb_dict = fb
                            
                        dspy_feedback = {
                            "solution_id": fb_dict.get("solution_id", str(uuid.uuid4())),
                            "rating": fb_dict.get("rating", 3),
                            "comment": fb_dict.get("comment", ""),
                            "improvements": fb_dict.get("improvements", []),
                            "timestamp": str(fb_dict.get("timestamp", datetime.utcnow()))
                        }
                        
                        # Only add if it's not already in the feedback data
                        if not any(item.get("solution_id") == dspy_feedback["solution_id"] for item in self.feedback_data):
                            self.feedback_data.append(dspy_feedback)
                    
                    print(f"Loaded {len(external_feedback)} external feedback items into DSPy module")
                    self.save_feedback_data()
                elif os.path.exists(self.feedback_file):
                    with open(self.feedback_file, 'r') as f:
                        self.feedback_data = json.load(f)
                    print(f"Loaded {len(self.feedback_data)} DSPy feedback examples")
            except Exception as e:
                print(f"Error loading DSPy feedback data: {e}")
        
        def save_feedback_data(self):
            try:
                os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
                with open(self.feedback_file, 'w') as f:
                    json.dump(self.feedback_data, f, default=str)
                print(f"Saved {len(self.feedback_data)} DSPy feedback examples")
            except Exception as e:
                print(f"Error saving DSPy feedback data: {e}")
        
        def forward(self, question: str, context: str = ""):
            # Check if we have relevant feedback to incorporate
            feedback_insights = ""
            if self.feedback_data:
                # Extract common patterns from feedback
                improvement_areas = {}
                for fb in self.feedback_data:
                    if fb.get("improvements"):
                        for area in fb["improvements"]:
                            if area in improvement_areas:
                                improvement_areas[area] += 1
                            else:
                                improvement_areas[area] = 1
                
                # Include top improvement areas in the solution generation
                if improvement_areas:
                    sorted_areas = sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)
                    for area, _ in sorted_areas[:3]:  # Use top 3 areas to guide solution
                        if "detailed" in area.lower():
                            context += "\nProvide detailed step-by-step explanations."
                        elif "formula" in area.lower():
                            context += "\nInclude relevant mathematical formulas."
                        elif "visual" in area.lower():
                            context += "\nDescribe visual representations where helpful."
            
            try:
                result = self.generate_solution(question=question, context=context)
                return result
            except Exception as e:
                print(f"Error in DSPy forward pass: {e}")
                # Return a fallback result
                return {
                    "solution": "Unable to generate solution using DSPy",
                    "steps": ["Error occurred during solution generation"],
                    "final_answer": "Error",
                    "explanation": f"DSPy module error: {str(e)}",
                    "category": "OTHER",
                    "confidence": 0.0
                }
        
        def update_from_feedback(self, question: str, solution: Dict[str, Any], feedback: Dict[str, Any]):
            # Add to feedback examples
            self.feedback_data.append({
                "question": question,
                "solution": solution,
                "feedback": feedback,
                "timestamp": str(feedback.get("timestamp", datetime.utcnow()))
            })
            
            # Save updated feedback data
            self.save_feedback_data()
            
            # Update optimization metrics
            self.optimization_metrics["total_optimizations"] += 1
            
            # In a real implementation, you would use DSPy's teleprompter or optimizer here
            # to update the model based on feedback
            # Example:
            # teleprompter = dspy.Teleprompter(self.generate_solution)
            # teleprompter.update_from_feedback(question, solution, feedback)
            
        def improve_solution(self, question: str, original_solution: MathSolution, feedback_patterns: List[str]):
            """Generate an improved solution based on feedback patterns"""
            # Create a prompt that includes the original solution and feedback patterns
            context = f"Improve this solution based on these feedback patterns: {', '.join(feedback_patterns)}"
            
            try:
                # Generate improved solution
                result = self.generate_solution(question=question, context=context)
                
                # Track improvement metrics
                if hasattr(result, 'confidence') and hasattr(original_solution, 'confidence'):
                    try:
                        # Parse confidence safely
                        result_confidence = result.confidence
                        if isinstance(result_confidence, str):
                            confidence_str = result_confidence.lower().strip()
                            confidence_mapping = {
                                'very high': 0.95, 'high': 0.85, 'medium': 0.7,
                                'moderate': 0.7, 'low': 0.5, 'very low': 0.3
                            }
                            if confidence_str in confidence_mapping:
                                result_confidence = confidence_mapping[confidence_str]
                            else:
                                import re
                                numeric_match = re.search(r'(\d*\.?\d+)', confidence_str)
                                if numeric_match:
                                    result_confidence = float(numeric_match.group(1))
                                    if result_confidence > 1.0:
                                        result_confidence = result_confidence / 100.0
                                    result_confidence = max(0.0, min(1.0, result_confidence))
                                else:
                                    result_confidence = 0.8
                        
                        confidence_improvement = result_confidence - original_solution.confidence
                        prev_avg = self.optimization_metrics["avg_confidence_improvement"]
                        prev_count = self.optimization_metrics["total_optimizations"]
                        
                        if prev_count > 0:
                            new_avg = (prev_avg * prev_count + confidence_improvement) / (prev_count + 1)
                            self.optimization_metrics["avg_confidence_improvement"] = new_avg
                    except Exception as e:
                        print(f"Error calculating confidence improvement: {e}")
                
                return result
            except Exception as e:
                print(f"Error improving solution with DSPy: {e}")
                return None

class MathAgent:
    def __init__(self, feedback_service=None):
        self.kb_service = KnowledgeBaseService()
        self.web_search_service = WebSearchService()
        self.feedback_service = feedback_service
        self.math_parser = MathParser()
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.2
        )

        self.solution_parser = JsonOutputParser(pydantic_object=MathSolutionSchema)
        
        # Initialize solution cache
        self.solution_cache = {}  # question -> AgentResponse
        
        # Initialize DSPy module if available
        self.dspy_module = None
        if DSPY_AVAILABLE:
            try:
                self.dspy_module = MathSolverModule()
                print("DSPy module initialized for feedback-based learning")
            except Exception as e:
                print(f"Failed to initialize DSPy module: {e}")
                self.dspy_module = None
        
        # Enhanced prompt with improved instructions
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert math professor with years of teaching experience. Your goal is to provide clear, step-by-step solutions to mathematical questions that students can easily understand and learn from. Break down complex problems into simple, understandable steps. Always provide a final answer and a confidence score. The output MUST be in JSON format according to the provided schema.\n\nWhen solving problems:\n1. Identify the mathematical concepts involved\n2. Outline a clear solution strategy\n3. Show each step with explanations\n4. Highlight key formulas and techniques\n5. Verify the answer when possible"),
            ("human", "Solve the following math question: {question}\n\nContext: {context}\n\n{format_instructions}")
        ])
        
        self.solution_chain = self.prompt_template | self.llm | self.solution_parser
        
        # Solution improvement prompt for refining solutions based on feedback patterns
        self.improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert math professor who specializes in improving mathematical solutions based on student feedback. Your goal is to enhance the given solution to make it clearer, more accurate, and more educational."),
            ("human", "Original solution: {solution}\n\nCommon feedback patterns: {feedback_patterns}\n\nSolution-specific feedback: {solution_specific_feedback}\n\nPlease improve this solution to address the feedback patterns while maintaining accuracy.")
        ])
        
        self.improvement_chain = self.improvement_prompt | self.llm
        
        # Feedback analysis prompt for extracting patterns from feedback
        self.feedback_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing student feedback on mathematical solutions. Your task is to identify common patterns and specific improvement areas from the feedback provided."),
            ("human", "Feedback data: {feedback_data}\n\nPlease analyze this feedback and identify: 1) Common patterns, 2) Specific improvement areas, 3) Priority issues to address")
        ])
        
        self.feedback_analysis_chain = self.feedback_analysis_prompt | self.llm

        # Relevance evaluation prompt for web search results
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at evaluating the relevance of web search results to mathematical questions. Your task is to determine if the provided search results contain information that can help answer the given mathematical question."),
            ("human", "Question: {question}\n\nWeb Search Results:\n{results}\n\nDoes this search result contain information that can help answer the mathematical question? Answer with only 'YES' or 'NO' and a brief explanation.")
        ])

        self.relevance_chain = self.relevance_prompt | self.llm

    async def solve_question(self, math_question: MathQuestion) -> AgentResponse:
        """Solve a mathematical question using the appropriate method based on routing"""
        # Generate a unique solution ID for tracking
        solution_id = str(uuid.uuid4())
        
        # Check if we have this question in cache
        if math_question.question in self.solution_cache:
            print(f"Retrieved from cache: {math_question.question}")
            cached_response = self.solution_cache[math_question.question]
            # Update the solution ID for tracking
            if not hasattr(cached_response.solution, 'id') or not cached_response.solution.id:
                cached_response.solution.id = solution_id
            return cached_response
        
        # Check if we should force web search based on context
        force_web_search = False
        if math_question.context and "FORCE_WEB_SEARCH" in math_question.context:
            force_web_search = True
            print("Forcing web search based on context flag")
        
        # 1. Check Knowledge Base (unless forced to web search)
        kb_solution = None
        if not force_web_search:
            kb_solution = await self.kb_service.retrieve_solution(math_question.question)
        
        if kb_solution:
            print(f"Retrieved from KB: {kb_solution.question}")
            
            # Check if we should improve the solution based on feedback
            improved_solution = await self._check_for_improvements(kb_solution.solution)
            if improved_solution:
                print("Returning improved solution based on feedback patterns")
                # Ensure the solution has an ID
                if not hasattr(improved_solution, 'id') or not improved_solution.id:
                    improved_solution.id = solution_id
                response = AgentResponse(
                    solution=improved_solution,
                    feedback_prompt="This solution was improved based on previous feedback. Was it helpful?"
                )
                self.solution_cache[math_question.question] = response
                return response
            
            # Ensure the solution has an ID
            if not hasattr(kb_solution.solution, 'id') or not kb_solution.solution.id:
                kb_solution.solution.id = solution_id
            response = AgentResponse(
                solution=kb_solution.solution,
                feedback_prompt="Was this solution helpful? Please provide feedback to help us improve."
            )
            self.solution_cache[math_question.question] = response
            return response

        # 2. Perform Web Search if not in KB or forced
        print(f"Performing web search for: {math_question.question}")
        search_results = await self.web_search_service.search(math_question.question)

        # Prepare context for solution generation
        context = math_question.context if math_question.context else ""
        if "FORCE_WEB_SEARCH" in context:
            context = context.replace("FORCE_WEB_SEARCH", "").strip()

        # Check if web search results are relevant to the question
        web_results_relevant = False
        if search_results:
            web_results_relevant = await self._evaluate_web_results_relevance(math_question.question, search_results)

        if search_results and web_results_relevant:
            context += "\n\nWeb Search Results:\n"
            for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
                context += f"{i+1}. {result.title}: {result.snippet}\n"
                if hasattr(result, 'url') and result.url:
                    context += f"   Source: {result.url}\n"
        elif search_results and not web_results_relevant:
            # Web results not relevant, will generate from LLM only
            context += "\n\nNote: Web search results were found but determined not relevant to the question. Generating solution using mathematical knowledge only."
        
        # 3. Generate solution using LLM or DSPy
        print("Generating solution...")
        try:
            solution = None
            datasource = "llm_only"
            if search_results and web_results_relevant:
                datasource = "web_search"
            elif search_results and not web_results_relevant:
                datasource = "llm_fallback_web_irrelevant"
            
            # Try using DSPy if available and human feedback is enabled
            if self.dspy_module and DSPY_AVAILABLE and settings.ENABLE_HUMAN_FEEDBACK:
                try:
                    print("Using DSPy module for solution generation")
                    dspy_result = self.dspy_module(question=math_question.question, context=context)
                    
                    # Convert DSPy result to MathSolution
                    steps = []
                    if hasattr(dspy_result, 'steps') and dspy_result.steps:
                        step_list = dspy_result.steps
                        if isinstance(step_list, str):
                            # Parse steps from string if needed
                            import re
                            step_matches = re.findall(r'Step (\d+): (.+)', step_list)
                            if step_matches:
                                for i, (_, step_text) in enumerate(step_matches):
                                    steps.append(Step(step_number=i+1, explanation=step_text))
                            else:
                                # Try parsing as JSON or list
                                try:
                                    parsed_steps = json.loads(step_list)
                                    if isinstance(parsed_steps, list):
                                        for i, step_text in enumerate(parsed_steps):
                                            steps.append(Step(step_number=i+1, explanation=str(step_text)))
                                except:
                                    # Just split by newlines as last resort
                                    step_lines = step_list.split('\n')
                                    for i, line in enumerate(step_lines):
                                        if line.strip():
                                            steps.append(Step(step_number=i+1, explanation=line.strip()))
                        elif isinstance(step_list, list):
                            for i, step in enumerate(step_list):
                                steps.append(Step(step_number=i+1, explanation=str(step)))
                    
                    if not steps:  # Fallback if steps parsing failed
                        steps = [Step(step_number=1, explanation="Solved using mathematical principles.")]
                    
                    # Determine category
                    category = getattr(dspy_result, 'category', QuestionType.OTHER)
                    if isinstance(category, str) and hasattr(QuestionType, category.upper()):
                        category = getattr(QuestionType, category.upper())
                    else:
                        category = QuestionType.OTHER
                    
                    # Parse confidence value safely
                    confidence_raw = getattr(dspy_result, 'confidence', 0.8)
                    confidence = 0.8  # default value
                    
                    if isinstance(confidence_raw, (int, float)):
                        confidence = float(confidence_raw)
                    elif isinstance(confidence_raw, str):
                        # Handle string confidence values
                        confidence_str = confidence_raw.lower().strip()
                        confidence_mapping = {
                            'very high': 0.95,
                            'high': 0.85,
                            'medium': 0.7,
                            'moderate': 0.7,
                            'low': 0.5,
                            'very low': 0.3
                        }
                        
                        if confidence_str in confidence_mapping:
                            confidence = confidence_mapping[confidence_str]
                        else:
                            # Try to extract numeric value from string
                            import re
                            numeric_match = re.search(r'(\d*\.?\d+)', confidence_str)
                            if numeric_match:
                                try:
                                    confidence = float(numeric_match.group(1))
                                    # Ensure it's in the valid range
                                    if confidence > 1.0:
                                        confidence = confidence / 100.0  # Convert percentage
                                    confidence = max(0.0, min(1.0, confidence))
                                except ValueError:
                                    confidence = 0.8
                    
                    # Ensure confidence is in valid range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    explanation = getattr(dspy_result, 'explanation', 'Solution generated using DSPy')
                    if datasource == "llm_fallback_web_irrelevant":
                        explanation = f"Note: Web search results were found but determined not relevant to the question. This solution was generated using mathematical knowledge only.\n\n{explanation}"

                    solution = MathSolution(
                        id=solution_id,
                        question=math_question.question,
                        steps=steps,
                        final_answer=getattr(dspy_result, 'final_answer', 'See explanation'),
                        explanation=explanation,
                        category=category,
                        confidence=confidence
                    )
                except Exception as e:
                    print(f"Error using DSPy module: {e}. Falling back to standard LLM.")
            
            # Fallback to standard LLM if DSPy failed or isn't available
            if not solution:
                # Ensure format_instructions are passed correctly
                format_instructions = self.solution_parser.get_format_instructions()
                solution_data = await self.solution_chain.ainvoke({
                    "question": math_question.question,
                    "context": context,
                    "format_instructions": format_instructions
                })
                
                # Convert steps to proper Step objects
                steps = []
                for i, step_data in enumerate(solution_data.get('steps', [])):
                    if isinstance(step_data, dict):
                        # Handle different possible formats from LLM
                        if 'step_number' in step_data and 'explanation' in step_data:
                            steps.append(Step(**step_data))
                        elif 'step' in step_data and 'explanation' in step_data:
                            # Handle format like {"step": "1", "explanation": "..."}
                            steps.append(Step(
                                step_number=int(step_data.get('step', i+1)),
                                explanation=step_data['explanation']
                            ))
                        elif 'description' in step_data:
                            # Handle format with 'description' instead of 'explanation'
                            steps.append(Step(
                                step_number=i+1,
                                explanation=step_data['description']
                            ))
                        else:
                            # Fallback: use the first available text field
                            text_field = None
                            for key in ['explanation', 'description', 'text', 'content']:
                                if key in step_data:
                                    text_field = step_data[key]
                                    break
                            if text_field:
                                steps.append(Step(step_number=i+1, explanation=str(text_field)))
                            else:
                                # Last resort: convert the whole dict to string
                                steps.append(Step(step_number=i+1, explanation=str(step_data)))
                    else:
                        # Handle string or other formats
                        steps.append(Step(step_number=i+1, explanation=str(step_data)))
                
                # Determine category
                category = solution_data.get('category', QuestionType.OTHER)
                if isinstance(category, str) and hasattr(QuestionType, category.upper()):
                    category = getattr(QuestionType, category.upper())
                else:
                    category = QuestionType.OTHER
                
                # Create MathSolution with proper Step objects
                explanation = solution_data.get('explanation', 'Solution generated')
                if datasource == "llm_fallback_web_irrelevant":
                    explanation = f"Note: Web search results were found but determined not relevant to the question. This solution was generated using mathematical knowledge only.\n\n{explanation}"

                solution = MathSolution(
                    id=solution_id,
                    question=solution_data.get('question', math_question.question),
                    steps=steps,
                    final_answer=solution_data.get('final_answer', 'See explanation'),
                    explanation=explanation,
                    category=category,
                    confidence=float(solution_data.get('confidence', 0.8))
                )
            
            # Store new solution in KB for future use
            await self.kb_service.add_solution(math_question.question, solution)
            
            # Create response with appropriate feedback prompt
            response = AgentResponse(
                solution=solution,
                feedback_prompt="Please rate the quality of this solution and provide specific feedback to help us improve."
            )
            
            # Cache the response
            self.solution_cache[math_question.question] = response
            
            return response
        except Exception as e:
            print(f"Error generating solution: {e}")
            # Fallback or error handling
            return AgentResponse(
                solution=MathSolution(
                    id=solution_id,
                    question=math_question.question,
                    steps=[Step(step_number=1, explanation="Could not generate a solution at this time.")],
                    final_answer="Error",
                    explanation=f"An error occurred while processing the request: {str(e)}",
                    category=QuestionType.OTHER,
                    confidence=0.0
                ),
                feedback_prompt="We apologize, an error occurred. Please try again later."
            )
    
    async def _check_for_improvements(self, solution: MathSolution) -> Optional[MathSolution]:
        """Check if a solution should be improved based on feedback patterns"""
        # Skip improvement check for high confidence solutions with detailed steps
        # unless we have specific feedback for this solution
        has_specific_feedback = False
        feedback_data = None
        
        if hasattr(solution, 'id') and solution.id and hasattr(self, 'feedback_service'):
            feedback_data = await self.feedback_service.get_feedback_by_solution_id(solution.id)
            has_specific_feedback = bool(feedback_data)
        
        if solution.confidence > 0.9 and len(solution.steps) >= 3 and not has_specific_feedback:
            return None
        
        # Use the improvement chain to check for potential improvements
        try:
            # Get feedback patterns from feedback service
            feedback_patterns = "No specific feedback patterns available."
            solution_specific_feedback = "No specific feedback for this solution."
            
            if hasattr(self, 'feedback_service'):
                # Get general feedback patterns
                stats = await self.feedback_service.get_feedback_statistics()
                if stats and stats.get('improvement_areas'):
                    patterns = []
                    for area, count in stats['improvement_areas'].items():
                        if count >= settings.FEEDBACK_MIN_COUNT:
                            patterns.append(f"{area} (mentioned {count} times)")
                    
                    if patterns:
                        feedback_patterns = "Common improvement areas:\n" + "\n".join([f"- {p}" for p in patterns])
                
                # Get solution-specific feedback if available
                if feedback_data:
                    specific_feedback = []
                    for fb in feedback_data:
                        if fb.rating < settings.FEEDBACK_THRESHOLD and fb.comment:
                            specific_feedback.append(f"Rating {fb.rating}/5: {fb.comment}")
                    
                    if specific_feedback:
                        solution_specific_feedback = "Specific feedback for this solution:\n" + "\n".join([f"- {fb}" for fb in specific_feedback])
            
            # If we have DSPy module and human feedback is enabled, try to use it for improvement
            if self.dspy_module and DSPY_AVAILABLE and settings.ENABLE_HUMAN_FEEDBACK:
                try:
                    # Prepare context with feedback
                    context = f"Original solution confidence: {solution.confidence}\n"
                    context += f"Number of steps: {len(solution.steps)}\n"
                    context += f"\n{feedback_patterns}\n"
                    context += f"\n{solution_specific_feedback}"
                    
                    # Use DSPy to improve the solution
                    dspy_result = self.dspy_module.improve_solution(
                        question=solution.question,
                        original_solution=solution,
                        feedback_patterns=[feedback_patterns, solution_specific_feedback]
                    )
                    
                    if dspy_result and hasattr(dspy_result, 'improved_solution'):
                        return dspy_result.improved_solution
                except Exception as e:
                    print(f"Error using DSPy for improvement: {e}. Falling back to standard improvement chain.")
            
            # Fallback to standard improvement chain
            result = await self.improvement_chain.ainvoke({
                "solution": solution,
                "feedback_patterns": feedback_patterns,
                "solution_specific_feedback": solution_specific_feedback
            })
            
            if result.get("should_improve", False):
                # Create improved solution
                improved_steps = []
                for i, step_data in enumerate(result.get('improved_steps', [])):
                    if isinstance(step_data, dict):
                        improved_steps.append(Step(**step_data))
                    else:
                        improved_steps.append(Step(step_number=i+1, explanation=str(step_data)))
                
                if not improved_steps:  # Fallback if steps parsing failed
                    return None
                
                # Preserve the original solution ID for tracking
                solution_id = getattr(solution, 'id', str(uuid.uuid4()))
                
                improved_solution = MathSolution(
                    id=solution_id,
                    question=solution.question,
                    steps=improved_steps,
                    final_answer=result.get('improved_final_answer', solution.final_answer),
                    explanation=result.get('improved_explanation', solution.explanation),
                    category=solution.category,
                    confidence=float(result.get('improved_confidence', solution.confidence))
                )
                
                # If we have a feedback service, record this improvement
                if hasattr(self, 'feedback_service'):
                    await self.feedback_service.record_improvement(
                        solution_id=solution_id,
                        original_confidence=solution.confidence,
                        improved_confidence=improved_solution.confidence,
                        improvement_type="automated_feedback_based"
                    )
                
                return improved_solution
        except Exception as e:
            print(f"Error checking for improvements: {e}")
        
        return None
    
    async def process_feedback(self, solution_id: str, feedback_data: dict) -> bool:
        """Process feedback for a solution and update the learning model"""
        if not settings.ENABLE_HUMAN_FEEDBACK:
            return False
            
        try:
            # Get the solution from the feedback service
            solution_feedback = await self.feedback_service.get_feedback_by_solution_id(solution_id)
            
            # If we have DSPy available, update the module with the feedback
            if self.dspy_module and DSPY_AVAILABLE:
                # Find the solution in the cache
                solution = None
                for question, response in self.solution_cache.items():
                    if response.solution.id == solution_id:
                        solution = response.solution
                        break
                
                if solution:
                    # Update the DSPy module with the feedback
                    self.dspy_module.update_from_feedback(
                        question=solution.question,
                        solution=json.loads(solution.model_dump_json()),
                        feedback=feedback_data
                    )
                    return True
            
            return False
        except Exception as e:
            print(f"Error processing feedback: {e}")
            return False

    async def _evaluate_web_results_relevance(self, question: str, search_results: List) -> bool:
        """Evaluate if web search results are relevant to the mathematical question"""
        try:
            # Format search results for evaluation
            results_text = ""
            for i, result in enumerate(search_results[:3]):  # Evaluate top 3 results
                results_text += f"{i+1}. {result.title}: {result.snippet}\n"

            # Use LLM to evaluate relevance
            relevance_result = await self.relevance_chain.ainvoke({
                "question": question,
                "results": results_text
            })

            # Parse the response - expect YES/NO format
            response_text = str(relevance_result).strip().upper()
            is_relevant = "YES" in response_text

            print(f"Web search results relevance evaluation: {is_relevant}")
            return is_relevant

        except Exception as e:
            print(f"Error evaluating web results relevance: {e}")
            # Default to relevant if evaluation fails
            return True
