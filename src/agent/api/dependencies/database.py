# src/stock_assistant/api/dependencies/database.py
from fastapi import Depends
from typing import Generator
from ...application.services.document_processing_service import DocumentProcessingService
from ...infra.vector_stores.qdrant_vector_store import QdrantVectorStore
from ...infra.document_loaders.s3_loader import S3DocumentLoader
from ...infra.vector_stores.qdrant_vector_store import QdrantVectorStore

from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)

def get_vector_store() -> Generator[QdrantVectorStore, None, None]:
    """Dependency to get vector store instance"""
    try:
        vector_store = QdrantVectorStore()
        yield vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

def get_document_processing_service(vector_store: QdrantVectorStore = Depends(get_vector_store)):
    """Dependency for injecting DocumentProcessingService."""
    loader = S3DocumentLoader()
    return DocumentProcessingService(loader, vector_store)