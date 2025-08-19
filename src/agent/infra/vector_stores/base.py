# src/stock_assistant/infrastructure/vector_stores/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ...domain.entities.document import DocumentChunk

class BaseVectorStore(ABC):
    """Base class for vector store implementations"""
    
    @abstractmethod
    async def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass