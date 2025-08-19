# src/stock_assistant/domain/entities/query_context.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime


class QueryContext(BaseModel):
    """Context for user queries"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class ToolResult(BaseModel):
    """Result from tool execution"""
    tool_name: str
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class AgentState(BaseModel):
    """State maintained during agent execution"""
    messages: List[Dict[str, str]] = []
    current_step: int = 0
    max_steps: int = 10
    tools_used: List[str] = []
    intermediate_results: List[ToolResult] = []
    final_answer: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True