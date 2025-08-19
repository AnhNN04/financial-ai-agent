# src/stock_assistant/api/models/responses.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class ToolResultResponse(BaseModel):
    """Response model for tool execution results"""
    tool_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None

class AgentResponse(BaseModel):
    """Response model for agent queries"""
    answer: str
    success: bool
    steps: int
    tools_used: List[str]
    intermediate_results: List[ToolResultResponse]
    session_id: Optional[str] = None
    timestamp: datetime = datetime.now()

class DocumentProcessResponse(BaseModel):
    """Response model for document processing"""
    success: bool
    processed_documents: int
    total_chunks: int
    failed_documents: List[str]
    processing_time: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = datetime.now()
    version: str
    services: Dict[str, str]