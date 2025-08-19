# src/stock_assistant/infra/providers/rag_provider.py
from typing import Dict, Any, List

from ...domain.tools.rag_tool import RagRetriever
from ..vector_stores.qdrant_vector_store import QdrantVectorStore
from ..base import HealthCheckable
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class QdrantRagRetriever(RagRetriever, HealthCheckable):
    """Infrastructure-specific implementation of RagRetriever using QdrantVectorStore.
    
    This provider isolates the concrete vector store implementation from the domain layer,
    adhering to Domain-Driven Design by depending on domain abstractions.
    """
    
    def __init__(self):
        self.vector_store = QdrantVectorStore()
    
    async def retrieve(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Retrieve similar documents using Qdrant vector store."""
        try:
            return await self.vector_store.similarity_search(
                query=query,
                k=max_results
            )
        except Exception as e:
            logger.error(f"RAG retrieval failed: {str(e)}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the RAG provider."""
        try:
            # Check health of underlying Qdrant vector store
            vector_store_health = await self.vector_store.check_health()
            
            if vector_store_health["status"] == "healthy":
                return {
                    "status": "healthy",
                    "details": {
                        "message": "RAG provider is operational",
                        "vector_store_status": vector_store_health["details"]
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "details": {
                        "message": "RAG provider is not operational due to vector store issues",
                        "vector_store_status": vector_store_health["details"]
                    }
                }
        except Exception as e:
            logger.error(f"RAG health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e),
                    "message": "Failed to check RAG provider health"
                }
            }