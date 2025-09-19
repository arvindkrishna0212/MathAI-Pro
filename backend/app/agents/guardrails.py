from typing import List, Dict, Any
import re
import json
from pydantic import BaseModel
from ..config import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class GuardrailResult(BaseModel):
    is_safe: bool
    message: str = ""
    category: str = ""
    confidence: float = 1.0

class AIGateway:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.1
        )

        
        # Input guardrail prompt with stronger emphasis on JSON format
        self.input_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI safety guardrail system. Your task is to analyze user input and determine if it's safe and appropriate for a mathematics education system. The system should only allow mathematical questions and related educational content.\n\nYou MUST respond with a valid JSON object containing ONLY these fields:\n{\"is_safe\": boolean, \"message\": string, \"category\": string, \"confidence\": float}\n\nDo not include any text outside of this JSON object."),
            ("human", "User input: {text}\n\nIs this input safe and appropriate for a mathematics education system? Respond with ONLY the JSON object.")
        ])
        
        # Output guardrail prompt with stronger emphasis on JSON format
        self.output_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI safety guardrail system. Your task is to analyze AI-generated output and determine if it's safe, appropriate, and relevant for a mathematics education system. The system should only provide mathematical solutions and related educational content.\n\nYou MUST respond with a valid JSON object containing ONLY these fields:\n{\"is_safe\": boolean, \"message\": string, \"category\": string, \"confidence\": float}\n\nDo not include any text outside of this JSON object."),
            ("human", "AI output: {text}\n\nIs this output safe, appropriate, and relevant for a mathematics education system? Respond with ONLY the JSON object.")
        ])
        
        self.input_chain = self.input_prompt | self.llm
        self.output_chain = self.output_prompt | self.llm
        
    def parse_response(self, response_content):
        """Parse the LLM response to extract the JSON object"""
        try:
            # Try to parse the entire response as JSON
            return json.loads(response_content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            try:
                # Look for JSON-like patterns
                import re
                json_match = re.search(r'\{[^}]*\"is_safe\"[^}]*\}', response_content)
                if json_match:
                    return json.loads(json_match.group(0))
                    
                # If no match, create a default response based on keywords
                is_safe = "not safe" not in response_content.lower() and "unsafe" not in response_content.lower()
                return {
                    "is_safe": is_safe,
                    "message": "Parsed from non-JSON response",
                    "category": "parsing_fallback",
                    "confidence": 0.6
                }
            except Exception as e:
                print(f"Error in JSON extraction: {e}")
                # Default fallback
                return {"is_safe": True, "message": "Failed to parse response", "category": "parsing_error", "confidence": 0.5}

# Initialize the AI Gateway
ai_gateway = AIGateway()

def input_guardrail(text: str) -> GuardrailResult:
    """Enhanced input guardrail with multiple layers of protection"""
    # Layer 1: Basic keyword filtering
    for keyword in settings.BLOCKED_KEYWORDS:
        if keyword.lower() in text.lower():
            return GuardrailResult(
                is_safe=False, 
                message=f"Input contains inappropriate content", 
                category="blocked_keyword",
                confidence=1.0
            )
    
    # Layer 2: Pattern matching for sensitive information
    # Check for potential PII (email, phone, SSN, etc.)
    pii_patterns = {
        "email": r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-]?){3}\d{4}\b'
    }
    
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, text):
            return GuardrailResult(
                is_safe=False, 
                message=f"Input contains sensitive personal information", 
                category="pii_detected",
                confidence=0.9
            )
    
    # Layer 3: Content relevance check
    # Ensure the input is related to mathematics
    math_keywords = [
        # Basic math terms
        "math", "equation", "formula", "solve", "calculate", "problem",
        "algebra", "geometry", "calculus", "arithmetic", "number",
        "function", "graph", "variable", "theorem", "proof",

        # Calculus terms
        "integration", "integral", "integrate", "derivative", "differentiate",
        "limit", "series", "sequence", "convergence", "divergence",

        # Trigonometry
        "sin", "cos", "tan", "trig", "trigonometry", "angle", "radian", "degree",

        # Advanced math
        "matrix", "vector", "linear", "quadratic", "polynomial", "logarithm", "log",
        "exponential", "power", "root", "square", "cube", "factor", "expand",

        # Math operations and symbols
        "plus", "minus", "times", "divide", "equals", "sum", "product", "difference",
        "quotient", "modulo", "absolute", "factorial",

        # Math concepts
        "area", "volume", "perimeter", "circumference", "radius", "diameter",
        "slope", "intercept", "coordinate", "plane", "axis", "origin"
    ]

    # Check for mathematical expressions and patterns
    has_math_expressions = (
        any(char in text for char in ['+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}', '∫', '∑', '∏', '√', '∂', 'Δ', '∞']) or  # Math symbols
        bool(re.search(r'\d+', text)) or  # Contains numbers
        bool(re.search(r'x\s*[+\-*/=]', text)) or  # Variables with operations
        bool(re.search(r'\b\d+\s*[+\-*/]\s*\d+\b', text))  # Number operations
    )

    # Simple heuristic: if the text contains math keywords or expressions, it's likely math-related
    is_math_related = any(keyword.lower() in text.lower() for keyword in math_keywords) or has_math_expressions

    if not is_math_related and len(text.split()) > 3:  # Only check longer queries
        # This is a simple heuristic; in a real system, you'd use a more sophisticated approach
        return GuardrailResult(
            is_safe=False,
            message="Input does not appear to be related to mathematics",
            category="off_topic",
            confidence=0.7
        )
    
    # Layer 4: AI-based content analysis (disabled for now to avoid parsing issues)
    # TODO: Re-enable once JSON parsing is more robust
    pass
    
    # Default: Allow the input
    return GuardrailResult(is_safe=True, category="safe_input", confidence=0.95)

def output_guardrail(text: str) -> GuardrailResult:
    """Enhanced output guardrail with multiple layers of protection"""
    # Layer 1: Check for allowed domains/topics (more permissive for math solutions)
    is_math_related = any(domain in text.lower() for domain in settings.ALLOWED_DOMAINS)

    # For math solutions, also check for mathematical content patterns
    has_math_content = (
        any(char in text for char in ['+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}', '√', '∫', '∑', '∏', 'Δ', '∂']) or  # Math operators/symbols
        any(word in text.lower() for word in ['step', 'answer', 'solution', 'calculate', 'compute', 'result', 'therefore', 'thus', 'hence']) or  # Solution keywords
        bool(re.search(r'\d+', text))  # Contains numbers
    )

    if not (is_math_related or has_math_content) and len(text) > 200:  # Only check very long outputs that don't seem math-related
        return GuardrailResult(
            is_safe=False,
            message="Output is not related to mathematics topics",
            category="off_topic",
            confidence=0.7
        )
    
    # Layer 2: Check for sensitive information leakage
    pii_patterns = {
        "email": r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-]?){3}\d{4}\b'
    }
    
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, text):
            return GuardrailResult(
                is_safe=False, 
                message=f"Output contains sensitive personal information", 
                category="pii_leaked",
                confidence=0.9
            )
    
    # Layer 3: Check for blocked keywords in output
    for keyword in settings.BLOCKED_KEYWORDS:
        if keyword.lower() in text.lower():
            return GuardrailResult(
                is_safe=False, 
                message=f"Output contains inappropriate content", 
                category="blocked_content",
                confidence=1.0
            )
    
    # Layer 4: AI-based content analysis (disabled for now to avoid parsing issues)
    # TODO: Re-enable once JSON parsing is more robust
    pass
    
    # Default: Allow the output
    return GuardrailResult(is_safe=True, category="safe_output", confidence=0.95)
