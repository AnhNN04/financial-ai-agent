
# src/stock_assistant/infra/providers/tavily_search_provider.py
from typing import Dict, Any, List

from tavily import TavilyClient
from ...domain.tools.tavily_search_tool import WebSearchRetriever
from ...shared.settings.settings import settings
from ..base import HealthCheckable
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class TavilySearchProvider(WebSearchRetriever, HealthCheckable):
    """Infrastructure-specific implementation of WebSearchRetriever using TavilyClient.
    
    This provider isolates the concrete Tavily API implementation from the domain layer,
    adhering to Domain-Driven Design by depending on domain abstractions.
    """
    
    def __init__(self):
        
        self.client = TavilyClient(api_key=settings.app.tavily_api_key)
    
    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform a web search using Tavily API."""
        search_response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=["cafef.vn", "vneconomy.vn", "vietstock.vn", "investing.com"],
            include_raw_content=True
        )
        
        results = []
        if search_response.get("results"):
            for result in search_response["results"]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "published_date": result.get("published_date", "")
                })
        return results
    

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of Tavily API."""
        try:
            # Test with a simple search
            response = self.client.search(
                query="health check",
                max_results=1
            )
            return {
                "status": "healthy",
                "details": {
                    "message": "Tavily API connection successful",
                    "results_count": len(response.get("results", []))
                }
            }
        except Exception as e:
            logger.error(f"Tavily health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e)
                }
            }
