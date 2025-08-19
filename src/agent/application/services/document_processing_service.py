# src/stock_assistant/application/services/document_processing_service.py
from typing import List, Dict, Any
import time
from ...infra.document_loaders.s3_loader import S3DocumentLoader
from ...infra.vector_stores.qdrant_vector_store import QdrantVectorStore
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)

class DocumentProcessingService:
    """Application service for processing documents from S3 to vector store."""

    def __init__(self, loader: S3DocumentLoader, vector_store: QdrantVectorStore):
        self.loader = loader
        self.vector_store = vector_store

    async def process_documents(self, s3_keys: List[str] = None) -> Dict[str, Any]:
        """Process documents from S3 and add to vector store."""
        try:
            start_time = time.time()
            total_chunks = 0
            processed_docs = 0
            failed_docs = []

            if not s3_keys:
                # Load all documents if no specific keys provided
                all_chunks = await self.loader.load_all_documents()
                total_chunks = len(all_chunks)
                processed_docs = len(set(chunk.metadata.source for chunk in all_chunks))

                if all_chunks:
                    success = await self.vector_store.add_documents(all_chunks)
                    if not success:
                        raise ValueError("Failed to add documents to vector store")
            else:
                # Process specific documents
                for s3_key in s3_keys:
                    try:
                        chunks = await self.loader.load_and_chunk_document(s3_key)
                        if chunks:
                            success = await self.vector_store.add_documents(chunks)
                            if success:
                                total_chunks += len(chunks)
                                processed_docs += 1
                            else:
                                failed_docs.append(s3_key)
                        else:
                            failed_docs.append(s3_key)
                    except Exception as e:
                        logger.error(f"Failed to process document {s3_key}: {str(e)}")
                        failed_docs.append(s3_key)

            processing_time = time.time() - start_time
            logger.info(f"Document processing completed: {processed_docs} docs, {total_chunks} chunks")

            return {
                "success": True,
                "processed_documents": processed_docs,
                "total_chunks": total_chunks,
                "failed_documents": failed_docs,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                "success": False,
                "processed_documents": 0,
                "total_chunks": 0,
                "failed_documents": s3_keys or [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }