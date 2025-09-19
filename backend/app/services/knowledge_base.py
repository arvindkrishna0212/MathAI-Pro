from typing import Optional, List
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from ..config import settings
from ..models.schemas import MathSolution, KnowledgeBaseItem, QuestionType, Step
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os
from datetime import datetime

# Dummy class for when Qdrant is not available
class KnowledgeBaseServiceDummy:
    """A dummy implementation of KnowledgeBaseService that provides minimal functionality
    when Qdrant is not available."""
    
    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        print("Initialized dummy knowledge base service with limited functionality")
    
    async def retrieve_solution(self, question: str) -> Optional[KnowledgeBaseItem]:
        """Always returns None as if no solution was found"""
        print("Dummy knowledge base service: retrieve_solution called but returning None")
        return None
        
    async def store_solution(self, solution: MathSolution) -> bool:
        """Pretends to store a solution but actually does nothing"""
        print("Dummy knowledge base service: store_solution called but not storing anything")
        return False
        
    async def search_similar(self, query: str, limit: int = 5) -> List[KnowledgeBaseItem]:
        """Returns an empty list as if no similar items were found"""
        print("Dummy knowledge base service: search_similar called but returning empty list")
        return []

class KnowledgeBaseService:
    def __init__(self):
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL)
            self.collection_name = settings.QDRANT_COLLECTION_NAME
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            self._create_collection_if_not_exists()
        except Exception as e:
            print(f"Error connecting to Qdrant at {settings.QDRANT_URL}: {str(e)}")
            print("Please ensure Qdrant server is running and accessible.")
            # Re-raise to allow proper error handling at the API level
            raise
        
        # Path for KB stats
        self.kb_stats_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         "data", "kb_stats.json")
        self.kb_stats = self._load_kb_stats()

    def _create_collection_if_not_exists(self):
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            # Check if the vector dimension matches our embedding model
            if collection_info.config.params.vectors.size != 384:
                print(f"Collection \'{self.collection_name}\' has incorrect vector dimension ({collection_info.config.params.vectors.size}), recreating...")
                self.client.delete_collection(collection_name=self.collection_name)
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE), # all-MiniLM-L6-v2 embeddings size
                )
                print(f"Collection \'{self.collection_name}\' recreated with correct dimension (384).")
            else:
                print(f"Collection \'{self.collection_name}\' already exists with correct dimension.")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE), # all-MiniLM-L6-v2 embeddings size
            )
            print(f"Collection \'{self.collection_name}\' created.")
            
    def _load_kb_stats(self):
        """Load knowledge base statistics from file or initialize if not exists"""
        if os.path.exists(self.kb_stats_file):
            try:
                with open(self.kb_stats_file, 'r') as f:
                    stats = json.load(f)
                    # Ensure total_items is always an integer
                    if stats.get("total_items") is None:
                        stats["total_items"] = 0
                    return stats
            except Exception as e:
                print(f"Error loading KB stats: {e}")

        # Initialize with empty stats
        return {
            "total_items": 0,
            "categories": {},
            "access_history": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_kb_stats(self):
        """Save knowledge base statistics to file"""
        os.makedirs(os.path.dirname(self.kb_stats_file), exist_ok=True)
        try:
            with open(self.kb_stats_file, 'w') as f:
                json.dump(self.kb_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving KB stats: {e}")
    
    def _update_kb_stats(self, action: str, question: str, category: str = None, success: bool = True):
        """Update knowledge base statistics"""
        self.kb_stats["access_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "question": question,
            "success": success
        })
        
        # Limit history size
        if len(self.kb_stats["access_history"]) > 100:
            self.kb_stats["access_history"] = self.kb_stats["access_history"][-100:]
        
        if action == "add" and success:
            self.kb_stats["total_items"] += 1
            if category:
                if category not in self.kb_stats["categories"]:
                    self.kb_stats["categories"][category] = 1
                else:
                    self.kb_stats["categories"][category] += 1
        
        self.kb_stats["last_updated"] = datetime.now().isoformat()
        self._save_kb_stats()

    async def _get_embedding(self, text: str) -> List[float]:
        # GoogleGenerativeAIEmbeddings expects a list of texts
        embeddings = await self.embeddings_model.aembed_documents([text])
        return embeddings[0]

    async def add_solution(self, question: str, solution: MathSolution):
        embedding = await self._get_embedding(question + " " + solution.explanation)
        
        # Create a unique ID for the point (ensure it's positive for Qdrant)
        import uuid
        point_id = str(uuid.uuid4())
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question": question,
                        "solution": solution.model_dump_json(), # Store solution as JSON string
                        "category": solution.category.value,
                        "timestamp": datetime.now().isoformat()
                    },
                )
            ],
        )
        
        # Update stats
        self._update_kb_stats("add", question, solution.category.value)
        print(f"Added solution for \'{question}\' to KB.")

    async def retrieve_solution(self, query: str, top_k: int = 1) -> Optional[KnowledgeBaseItem]:
        query_embedding = await self._get_embedding(query)
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )
        
        if search_result and search_result[0].score > 0.8: # Threshold for relevance
            payload = search_result[0].payload
            solution_data = payload["solution"]
            # Reconstruct MathSolution from JSON string
            solution = MathSolution.model_validate_json(solution_data)
            
            # Update stats
            self._update_kb_stats("retrieve", query, payload.get("category"))
            
            return KnowledgeBaseItem(
                id=str(search_result[0].id),
                question=payload["question"],
                solution=solution,
                embedding=search_result[0].vector # Optionally return embedding
            )
            
        # Update stats for unsuccessful retrieval
        self._update_kb_stats("retrieve", query, success=False)
        return None
        
    async def find_similar_questions(self, question: str, limit: int = 3) -> List[KnowledgeBaseItem]:
        """Find similar questions in the knowledge base without retrieving full solutions."""
        try:
            # Generate embedding for the question
            query_embedding = await self._get_embedding(question)
            
            # Search for similar questions in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,  # Get top N similar results
            )
            
            # Convert results to KnowledgeBaseItems
            similar_items = []
            for result in results:
                payload = result.payload
                similar_items.append(KnowledgeBaseItem(
                    id=str(result.id),
                    question=payload["question"],
                    solution=None,  # Don't include full solution for similarity check
                    score=result.score,
                ))
            
            # Update stats
            self._update_kb_stats("similarity_check", question, 
                                 success=(len(similar_items) > 0))
            
            return similar_items
        except Exception as e:
            print(f"Error finding similar questions: {e}")
            self._update_kb_stats("similarity_check", question, success=False)
            return []
    
    async def get_kb_statistics(self):
        """Get knowledge base statistics"""
        try:
            # Get collection info from Qdrant
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            # Update stats with current vector count (ensure it's not None)
            vector_count = collection_info.vectors_count
            if vector_count is not None:
                self.kb_stats["total_items"] = vector_count
            else:
                self.kb_stats["total_items"] = 0
            self._save_kb_stats()

            return self.kb_stats
        except Exception as e:
            print(f"Error getting KB statistics: {e}")
            return self.kb_stats
