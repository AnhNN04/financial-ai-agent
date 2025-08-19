# src/stock_assistant/api/routers/health.py
from fastapi import APIRouter
from ..models.responses import HealthResponse
from ...shared.settings.settings import settings
from ...application.services.health_check_service import HealthCheckService
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_service = HealthCheckService()
        result = await health_service.check_health()
        return HealthResponse(**result)
    except Exception as e:
        logger.error(f"Health check endpoint failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app.version,
            services={
                "vector_store": f"Status: Unknown - Detailed Error: {str(e)}",
                "llm": f"Status: Unknown - Detailed Error: {str(e)}",
                "search": f"Status: Unknown - Detailed Error: {str(e)}",
                "rag_provider": f"Status: Unknown - Detailed Error: {str(e)}",
                "cohere_embedding": f"Status: Unknown - Detailed Error: {str(e)}",
                "gemini_embedding": f"Status: Unknown - Detailed Error: {str(e)}",
                "hf_embedding": f"Status: Unknown - Detailed Error: {str(e)}",
                "s3_loader": f"Status: Unknown - Detailed Error: {str(e)}",
            }
        )