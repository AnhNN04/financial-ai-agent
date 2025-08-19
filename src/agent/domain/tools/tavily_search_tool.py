# src/stock_assistant/domain/tools/tavily_search_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .base import BaseTool
from ..entities.query_context import QueryContext
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)

class WebSearchRetriever(ABC):
    """Abstract interface for web search operations in domain layer."""
    
    @abstractmethod
    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform a web search and return results."""
        pass

class TavilySearchTool(BaseTool):
    """Web search tool for retrieving current information about Vietnamese stocks, market news, and financial data.
    
    This tool represents the domain service for web search operations, depending on an abstracted retriever
    to isolate infrastructure concerns as per Domain-Driven Design principles.
    """
    
    def __init__(self, web_search_retriever: WebSearchRetriever):
        super().__init__(
            name="tavily_search",
            description="Search the web for current information about Vietnamese stocks, market news, and financial data"
        )
        self.web_search_retriever = web_search_retriever
    
    async def _execute_impl(self, context: QueryContext, **kwargs) -> Dict[str, Any]:
        """Execute web search using the injected retriever."""
        query = kwargs.get("query", context.query)
        max_results = kwargs.get("max_results", 5)
        
        # Enhance query for Vietnamese stock market context (domain logic)
        enhanced_query = self._enhance_query_for_vietnamese_market(query)
        
        try:
            # Retrieve search results using the abstracted retriever
            search_results = self.web_search_retriever.search(
                query=enhanced_query,
                max_results=max_results
            )
            
            # Format results (domain formatting logic)
            formatted_results = self._format_results(search_results)
            
            return {
                "query": enhanced_query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            raise
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw search results into domain-specific structure."""
        formatted = []
        for result in results:
            formatted.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", "")
            })
        return formatted
    
    def _enhance_query_for_vietnamese_market(self, query: str) -> str:
        """Enhance search query for Vietnamese stock market context (domain-specific logic)."""
        vietnamese_market_terms = [
            "Vietnam stock", "Vietnamese stock market", "HOSE", "HNX", "UPCoM",
            "chứng khoán Việt Nam", "thị trường chứng khoán"
        ]
        
        # Add Vietnamese market context if not already present
        query_lower = query.lower()
        has_vn_context = any(term.lower() in query_lower for term in vietnamese_market_terms)
        
        if not has_vn_context:
            # Check if query contains stock symbols (2-4 uppercase letters)
            import re
            if re.search(r'\b[A-Z]{2,4}\b', query):
                enhanced_query = f"{query} Vietnam stock HOSE HNX"
            else:
                enhanced_query = f"{query} Vietnamese stock market"
        else:
            enhanced_query = query
        
        return enhanced_query