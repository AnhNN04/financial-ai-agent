# src/stock_assistant/api/routers/agent.py
from fastapi import APIRouter, HTTPException, Depends

from ..models.requests import QueryRequest, DocumentUploadRequest
from ..models.responses import AgentResponse, DocumentProcessResponse

from ..dependencies.service import get_stock_analysis_service
from ..dependencies.database import get_document_processing_service

from ...application.services.health_check_service import HealthCheckService
from ...application.services.document_processing_service import DocumentProcessingService
from ...application.services.stock_analysis_service import StockAnalysisService

from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings
logger = Logger.get_logger(__name__)


router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentResponse)
async def query_agent(
    request: QueryRequest,
    stock_analysis_service: StockAnalysisService = Depends(get_stock_analysis_service),
    health_service: HealthCheckService = Depends(HealthCheckService)
):
    """Process user query using StockAnalysisService."""
    try:
        # Check health of required services
        health_status = await health_service.check_health()
        if health_status["status"] == "unhealthy":
            unhealthy_services = [name for name, status in health_status["services"].items() if status["status"] == "unhealthy"]
            raise HTTPException(
                status_code=503,
                detail=f"Cannot process query: unhealthy services: {unhealthy_services}"
            )

        result = await stock_analysis_service.analyze(
            query=request.query,
            session_id=request.session_id
        )

        return AgentResponse(
            answer=result["answer"],
            success=result["metadata"]["success"],
            steps=result["metadata"]["steps"],
            tools_used=result["metadata"]["tools_used"],
            intermediate_results=result["metadata"]["intermediate_results"],
            session_id=result["metadata"]["session_id"]
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/load-documents", response_model=DocumentProcessResponse)
async def load_documents(
    request: DocumentUploadRequest,
    doc_service: DocumentProcessingService = Depends(get_document_processing_service),
    health_service: HealthCheckService = Depends(HealthCheckService)
):
    """Load and process documents from S3 into vector store."""
    try:
        # Check health of required services
        health_status = await health_service.check_health()
        if health_status["status"] == "unhealthy":
            unhealthy_services = [name for name, status in health_status["services"].items() if status["status"] == "unhealthy"]
            raise HTTPException(
                status_code=503,
                detail=f"Cannot process documents: unhealthy services: {unhealthy_services}"
            )

        result = await doc_service.process_documents(s3_keys=request.s3_keys)

        return DocumentProcessResponse(
            success=result["success"],
            processed_documents=result["processed_documents"],
            total_chunks=result["total_chunks"],
            failed_documents=result["failed_documents"],
            processing_time=result["processing_time"]
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@router.get("/tools")
async def list_tools(
    stock_analysis_service: StockAnalysisService = Depends(get_stock_analysis_service)
):
    """List available tools from StockAnalysisService."""
    try:
        tools = stock_analysis_service.coordinator.tools
        return {
            "tools": [
                {
                    "name": tool_name,
                    "description": tool.description
                }
                for tool_name, tool in tools.items()
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")