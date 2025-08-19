# src/stock_assistant/domain/tools/chat_tool.py
from typing import Dict, Any
from .base import BaseTool
from ..entities.query_context import QueryContext
from ...infra.chats.base import ChatProvider
from ...shared.settings.settings import Settings
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class ChatTool(BaseTool):
    """Tool for interacting with an LLM to answer queries about the Vietnamese stock market.
    
    This tool represents the domain service for LLM chat operations, depending on an abstracted
    ChatProvider to isolate infrastructure concerns as per Domain-Driven Design principles.
    """
    
    def __init__(self, chat_provider: ChatProvider):
        super().__init__(
            name="chat_llm",
            description="Interact with an LLM to answer queries about Vietnamese stocks, market news, and financial data"
        )
        self.chat_provider = chat_provider
        self.settings = Settings()
    
    async def _execute_impl(self, context: QueryContext, **kwargs) -> Dict[str, Any]:
        """Execute LLM chat using the injected provider."""
        query = kwargs.get("query", context.query)
        provider = kwargs.get("provider", self.settings.llm.default_provider)
        model = kwargs.get("model", None)
        temperature = kwargs.get("temperature", self.settings.llm.temperature)
        max_tokens = kwargs.get("max_tokens", self.settings.llm.max_tokens)
        streaming = kwargs.get("streaming", self.settings.llm.streaming)
        callbacks = kwargs.get("callbacks", [] if self.settings.llm.enable_callbacks else None)
        
        # Enhance query for Vietnamese stock market context
        enhanced_query = self._enhance_query_for_vietnamese_market(query)
        
        # Apply content filter if enabled
        if self.settings.llm.enable_content_filter:
            enhanced_query = self._apply_content_filter(enhanced_query)
        
        try:
            # Call the chat provider with settings
            chat_response = self.chat_provider.chat(
                prompt=enhanced_query,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                callbacks=callbacks,
                top_p=kwargs.get("top_p", getattr(self.settings.app, f"{provider}_top_p", 1.0)),
                frequency_penalty=kwargs.get("frequency_penalty", getattr(self.settings.app, f"{provider}_frequency_penalty", 0.0)),
                presence_penalty=kwargs.get("presence_penalty", getattr(self.settings.app, f"{provider}_presence_penalty", 0.0)),
                top_k=kwargs.get("top_k", getattr(self.settings.app, f"{provider}_top_k", 40)) if provider == "gemini" else None
            )
            
            # Apply content filter to response if enabled
            response = chat_response["response"]
            if self.settings.llm.enable_content_filter:
                response = self._apply_content_filter(response)
            
            return {
                "query": enhanced_query,
                "response": response,
                "model": chat_response["model"],
                "usage": chat_response["usage"],
                "total_tokens": chat_response["usage"]["total_tokens"],
                "provider": provider
            }
            
        except Exception as e:
            logger.error(f"Chat LLM failed: {str(e)}")
            raise
    
    def _enhance_query_for_vietnamese_market(self, query: str) -> str:
        """Enhance query for Vietnamese stock market context (domain-specific logic)."""
        vietnamese_market_terms = [
            "Vietnam stock", "Vietnamese stock market", "HOSE", "HNX", "UPCoM",
            "chứng khoán Việt Nam", "thị trường chứng khoán"
        ]
        
        query_lower = query.lower()
        has_vn_context = any(term.lower() in query_lower for term in vietnamese_market_terms)
        
        if not has_vn_context:
            import re
            if re.search(r'\b[A-Z]{2,4}\b', query):
                enhanced_query = f"{query} Vietnam stock HOSE HNX"
            else:
                enhanced_query = f"{query} Vietnamese stock market"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _apply_content_filter(self, text: str) -> str:
        """Apply basic content filtering to ensure safe and appropriate content."""
        # Basic filtering: remove sensitive keywords or phrases
        sensitive_terms = ["inappropriate", "offensive", "harmful"]  # Add more as needed
        filtered_text = text
        for term in sensitive_terms:
            filtered_text = filtered_text.replace(term, "[FILTERED]")
        
        # If using LangChain's moderation chain, you can integrate it here
        # Example: from langchain.chains.moderation import ModerationChain
        return filtered_text