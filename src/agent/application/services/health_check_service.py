# src/stock_assistant/application/services/health_check_service.py
from typing import Dict, Any

from ...infra.vector_stores.qdrant_vector_store import QdrantVectorStore
from ...infra.chats.gemini_chat import GeminiChatProvider
from ...infra.providers.tavily_search_provider import TavilySearchProvider
from ...infra.providers.rag_provider import QdrantRagRetriever
from ...infra.embeddings.cohere_multilang_v3_embedding import CohereV3Embeddings
# from ...infra.embeddings.gemini_embedding import GeminiEmbeddings
# from ...infra.embeddings.hf_embedding import HfEmbeddings
from ...infra.document_loaders.s3_loader import S3DocumentLoader

from ...shared.settings.settings import settings
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class HealthCheckService:
    """Application service for checking the health of system components."""
    
    def __init__(self):
        self.services = {
            "vector_store": QdrantVectorStore(),
            "llm": GeminiChatProvider(),
            "search": TavilySearchProvider(),
            "rag_provider": QdrantRagRetriever(),
            "cohere_embedding": CohereV3Embeddings(),
            # "gemini_embedding": GeminiEmbeddings(),
            # "hf_embedding": HfEmbeddings(),
            "s3_loader": S3DocumentLoader()
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of all system components."""
        try:
            results = {}
            overall_status = "healthy"
            
            for service_name, service in self.services.items():
                try:
                    print("="*70)
                    print(service_name)
                    result = await service.check_health()
                    print(result)
                    print("")
                    results[service_name] = result
                    if result["status"] == "unhealthy":
                        overall_status = "unhealthy"
                except Exception as e:
                    logger.error(f"Health check for {service_name} failed: {str(e)}")
                    results[service_name] = {
                        "status": "unhealthy",
                        "details": {
                            "error": str(e),
                            "message": f"Failed to check health of {service_name}"
                        }
                    }
                    overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "version": settings.app.version,
                "services": results
            }
        except Exception as e:
            logger.error(f"Health check service failed: {str(e)}")
            return {
                "status": "unhealthy",
                "version": settings.app.version,
                "services": {
                    name: {
                        "status": "unknown",
                        "details": {"error": str(e), "message": f"Failed to check health of {name}"}
                    }
                    for name in self.services
                }
            }