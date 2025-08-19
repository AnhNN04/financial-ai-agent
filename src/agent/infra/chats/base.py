"""
Base classes for chat infrastructure layer

This module provides the foundational abstractions for chat implementations
that will be used by domain/tools layer for higher-level operations.
"""
# src/stock_assistant/infra/chats/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ChatProvider(ABC):
    """Abstract interface for chat operations with LLMs in infrastructure layer."""
    
    @abstractmethod
    def chat(self, prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 2000, 
             streaming: bool = False, callbacks: Optional[List] = None, **kwargs) -> Dict[str, Any]:
        """Send a prompt to the LLM and return the response."""
        pass