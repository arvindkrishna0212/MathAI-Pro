import asyncio
import os
import sys
from datasets import load_dataset
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.models.schemas import MathSolution, QuestionType, Step
from backend.app.services.knowledge_base import KnowledgeBaseService

# Initialize knowledge base service
kb_service = KnowledgeBaseService()

# Load GSM8K dataset and select 100 random samples
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = random.sample(list(dataset), 100)  # Randomly select 100 items

# Function to parse answer into steps (split on sentences or << >> markers)
def parse_steps(answer: str) -> tuple[list[Step], str]:
    # Extract final answer (after ####) and explanation
    parts = answer.split("####")
    explanation = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) > 1 else ""
    # Split explanation into steps (basic split on periods; improve with regex if needed)
    steps_text = explanation.replace("<<", ".").replace(">>", ".").split(". ")
    steps = [Step(step_number=i+1, explanation=step.strip()) for i, step in enumerate(steps_text) if step.strip()]
    return steps, final_answer

# Function to infer question category based on keywords
def infer_category(question: str) -> QuestionType:
    question = question.lower()
    if any(word in question for word in ["area", "triangle", "circle", "perimeter"]):
        return QuestionType.GEOMETRY
    elif any(word in question for word in ["x =", "equation", "solve for"]):
        return QuestionType.ALGEBRA
    elif any(word in question for word in ["probability", "chance", "likely"]):
        return QuestionType.STATISTICS
    return QuestionType.ARITHMETIC  # Default for GSM8K

# Add 100 samples to Qdrant
async def add_gsm8k_subset():
    added_count = 0
    for item in dataset:
        question = item["question"]
        answer = item["answer"]

        steps, final_answer = parse_steps(answer)
        explanation = answer.split("####")[0].strip()

        # Create MathSolution
        solution = MathSolution(
            question=question,
            explanation=explanation,
            final_answer=final_answer,
            category=infer_category(question),
            steps=steps,
            confidence=1.0
        )

        try:
            await kb_service.add_solution(question, solution)
            added_count += 1
            print(f"Added: {question[:50]}...")
        except Exception as e:
            print(f"Error adding {question[:50]}: {e}")

    print(f"Successfully added {added_count} items to Qdrant.")
    # Print stats
    stats = await kb_service.get_kb_statistics()
    print(f"Knowledge base stats: {stats}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(add_gsm8k_subset())
