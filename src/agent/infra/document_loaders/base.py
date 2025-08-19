# src/stock_assistant/infrastructure/document_loaders/base.py
from abc import ABC, abstractmethod
from typing import List
from ...domain.entities.document import DocumentChunk

class BaseDocumentLoader(ABC):
    """Base class for document loaders"""
    
    @abstractmethod
    async def load_and_chunk_document(self, source: str) -> List[DocumentChunk]:
        """Load and chunk a single document"""
        pass
    
    @abstractmethod
    async def load_all_documents(self) -> List[DocumentChunk]:
        """Load and chunk all available documents"""
        pass