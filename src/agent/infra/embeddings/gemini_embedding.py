# src/stock_assistant/infra/providers/gemini_embeddings.py
import google.generativeai as genai
from typing import List, Optional, Dict, Any
from .base import BaseEmbeddings
from ..base import HealthCheckable
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)

class GeminiEmbeddings(BaseEmbeddings, HealthCheckable):
    """Google Gemini embeddings implementation"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini embeddings
        
        Args:
            model: Gemini embedding model name (optional, will use settings if not provided)
            api_key: Gemini API key (optional, will use settings if not provided)
        """
        # Use settings values as defaults if not provided
        self.model = model or settings.embeddings.model
        self.api_key = api_key or settings.app.gemini_api_key

        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            logger.info(f"Initialized Gemini embeddings with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise

    def _invoke_model(self, texts: List[str]) -> List[List[float]]:
        """Invoke Gemini model to get embeddings"""
        try:
            # Call Gemini embedding API
            response = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document"
            )
            
            # Extract embeddings
            embeddings = response.get("embedding", [])
            if isinstance(embeddings[0], list):
                return embeddings
            else:
                # If single text, wrap in list for consistency
                return [embeddings]
            
        except Exception as e:
            logger.error(f"Failed to invoke Gemini model: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            logger.debug(f"Generating embedding for single text of length: {len(text)}")
            embeddings = self._invoke_model([text])
            
            if not embeddings:
                raise ValueError("No embeddings returned from model")
                
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not texts:
                return []
                
            logger.debug(f"Generating embeddings for {len(texts)} documents")
            
            # Gemini API supports batching, but we'll process in chunks to avoid limits
            max_batch_size = getattr(settings.embeddings, 'cohere_max_batch_size', 96)
            all_embeddings = []
            
            for i in range(0, len(texts), max_batch_size):
                batch_texts = texts[i:i + max_batch_size]
                batch_embeddings = self._invoke_model(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processed batch {i//max_batch_size + 1}, texts: {len(batch_texts)}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for documents: {e}")
            raise

    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""
        return {
            "model": self.model,
            "max_batch_size": getattr(settings.embeddings, 'cohere_max_batch_size', 96),
            "max_input_length": 2048  # Gemini embedding models typically support up to 2048 tokens
        }

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of Gemini embedding provider."""
        try:
            # Test with a simple embedding request
            test_text = ["Health check"]
            embeddings = self._invoke_model(test_text)
            embedding_length = len(embeddings[0]) if embeddings else 0
            return {
                "status": "healthy",
                "details": {
                    "message": "Gemini embedding API connection successful",
                    "model": self.model,
                    "embedding_length": embedding_length
                }
            }
        except Exception as e:
            logger.error(f"Gemini health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e),
                    "message": "Failed to connect to Gemini embedding API"
                }
            }