# src/stock_assistant/infra/chats/gemini_chat.py
from typing import Dict, Any, Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from ...infra.chats.base import ChatProvider
from ...shared.settings.settings import settings
from ..base import HealthCheckable


from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)

class GeminiChatProvider(ChatProvider, HealthCheckable):
    """Infrastructure-specific implementation of ChatProvider using LangChain's ChatGoogleGenerativeAI."""
    
    def __init__(self):
        self.client = ChatGoogleGenerativeAI(
            google_api_key=settings.app.gemini_api_key,
            model=settings.llm.gemini_model,
            temperature=settings.llm.gemini_temperature,
            max_output_tokens=settings.llm.gemini_max_tokens,
            top_p=settings.llm.gemini_top_p,
            top_k=settings.llm.gemini_top_k,
            timeout=settings.llm.gemini_timeout,
            max_retries=settings.llm.gemini_max_retries,
            verbose=settings.llm.verbose
        )
    
    def chat(self, prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 2000, 
             streaming: bool = False, callbacks: Optional[List] = None, **kwargs) -> Dict[str, Any]:
        """Send a prompt to Gemini via LangChain and return the response."""
        try:
            messages = [
                SystemMessage(content="You are a helpful assistant for Vietnamese stock market queries."),
                HumanMessage(content=prompt)
            ]
            
            # Update client with overridden parameters if provided
            client = self.client
            if model or temperature != settings.llm.gemini_temperature or max_tokens != settings.llm.gemini_max_tokens:
                client = client.with_config({
                    "model": model or settings.llm.gemini_model,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": kwargs.get("top_p", settings.llm.gemini_top_p),
                    "top_k": kwargs.get("top_k", settings.llm.gemini_top_k)
                })
            
            # Handle streaming if enabled
            if streaming:
                response_text = ""
                for chunk in client.stream(messages, config={"callbacks": callbacks or []}):
                    response_text += chunk.content
                response = {"content": response_text}
                usage = {}  # Streaming mode may not provide usage metadata
            else:
                response = client.invoke(messages, config={"callbacks": callbacks or []})
                usage = response.response_metadata.get("token_usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
            
            return {
                "response": response.content,
                "model": model or settings.llm.gemini_model,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            }
        except Exception as e:
            logger.error(f"Gemini chat failed: {str(e)}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of OpenAI LLM."""
        try:
            # Test with a simple prompt
            messages = [
                SystemMessage(content="Health check"),
                HumanMessage(content="Ping")
            ]
            response = await self.client.ainvoke(messages)
            return {
                "status": "healthy",
                "details": {
                    "model": response.response_metadata.get("model_name", settings.app.openai_model),
                    "message": "OpenAI API connection successful"
                }
            }
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e)
                }
            }