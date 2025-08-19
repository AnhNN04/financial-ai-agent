from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from .base import BaseEmbeddings
from ..base import HealthCheckable


from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)

class HfEmbeddings(BaseEmbeddings, HealthCheckable):
    """Local embeddings implementation using SentenceTransformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            logger.info(f"Loading local embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            embedding = self.model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise


    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise


    async def check_health(self) -> Dict[str, Any]:
        """Check the health of Hugging Face embedding provider."""
        try:
            # Test with a simple embedding request
            test_text = ["Health check"]
            embeddings = self.model.encode(test_text).tolist()
            embedding_length = len(embeddings[0]) if embeddings else 0
            return {
                "status": "healthy",
                "details": {
                    "message": "Hugging Face model loaded successfully",
                    "model": self.model_name,
                    "embedding_length": embedding_length
                }
            }
        except Exception as e:
            logger.error(f"Hugging Face health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e),
                    "message": "Failed to load or use Hugging Face model"
                }
            }