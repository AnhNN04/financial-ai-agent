# src/stock_assistant/domain/entities/document.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Metadata for documents"""
    source: str  # S3 path or URL
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    document_type: str  # pdf, docx, txt
    tags: List[str] = []
    language: str = "vi"  # Vietnamese by default


class DocumentChunk(BaseModel):
    """Document chunk for vector storage"""
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    start_char: int
    end_char: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata.dict(),
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char
        }