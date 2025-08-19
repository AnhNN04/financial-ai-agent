# src/stock_assistant/infrastructure/vector_stores/qdrant_client.py
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition

from .base import BaseVectorStore
from ..base import HealthCheckable
from ...domain.entities.document import DocumentChunk
from ...infra.embeddings.cohere_multilang_v3_embedding import CohereV3Embeddings
from ...shared.exceptions.domain_exceptions import VectorStoreError
from ...shared.settings.settings import settings
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class QdrantVectorStore(BaseVectorStore, HealthCheckable):
    """Qdrant implementation of vector store"""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key
        )
        self.collection_name = settings.qdrant.collection_name
        self.embeddings_client = CohereV3Embeddings(
            model_id=settings.embeddings.cohere_model_id,
            region_name=settings.s3.aws_region,
            aws_access_key_id=settings.s3.aws_access_key_id,
            aws_secret_access_key=settings.s3.aws_secret_access_key
        )
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if not"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.qdrant.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
        except Exception as e:
            raise VectorStoreError(f"Failed to ensure collection exists: {str(e)}")
    
    
    async def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """Add document chunks to Qdrant"""
        try:
            
            points = []
            for doc in documents:
                # Generate embedding for document content
                embedding = await self.embeddings_client.embed_text(doc.content)
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": doc.content,
                        "source": doc.metadata.source,
                        "title": doc.metadata.title,
                        "document_type": doc.metadata.document_type,
                        "chunk_index": doc.chunk_index,
                        "tags": doc.metadata.tags,
                        "language": doc.metadata.language
                    }
                )
                points.append(point)
            
            # Upload points to Qdrant
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return operation_info.status == "completed"
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}")
    

    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try: 
            # Generate embedding for query
            query_embedding = await self.embeddings_client.embed_text(query)
            
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(FieldCondition(key=field, match={"value": value}))
                query_filter = Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=k
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "content": hit.payload["content"],
                    "metadata": {
                        "source": hit.payload.get("source", ""),
                        "title": hit.payload.get("title", ""),
                        "document_type": hit.payload.get("document_type", ""),
                        "tags": hit.payload.get("tags", [])
                    },
                    "score": hit.score
                })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise VectorStoreError(f"Similarity search failed: {str(e)}")
    

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=document_ids
            )
            
            logger.info(f"Deleted {len(document_ids)} documents from vector store")
            return operation_info.status == "completed"
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise VectorStoreError(f"Failed to delete documents: {str(e)}")
        

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of Qdrant vector store."""
        try:
            collections = self.client.get_collections()
            return {
                "status": "healthy",
                "details": {
                    "collections": len(collections.collections),
                    "message": "Qdrant connection successful"
                }
            }
        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e)
                }
            }