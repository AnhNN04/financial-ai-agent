# src/stock_assistant/domain/tools/rag_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .base import BaseTool
from ..entities.query_context import QueryContext
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)

class RagRetriever(ABC):
    """Abstract interface for RAG retrieval operations in domain layer."""
    
    @abstractmethod
    async def retrieve(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Retrieve similar documents from the knowledge base."""
        pass

class RAGTool(BaseTool):
    """RAG (Retrieval Augmented Generation) tool for knowledge retrieval.
    
    This tool represents the domain service for RAG operations, depending on an abstracted retriever
    to isolate infrastructure concerns as per Domain-Driven Design principles.
    """
    
    def __init__(self, rag_retriever: RagRetriever):
        super().__init__(
            name="rag_knowledge",
            description="Search internal knowledge base for Vietnamese stock market information from uploaded documents"
        )
        self.rag_retriever = rag_retriever
    
    async def _execute_impl(self, context: QueryContext, **kwargs) -> Dict[str, Any]:
        """Execute RAG search using the injected retriever."""
        query = kwargs.get("query", context.query)
        max_results = kwargs.get("max_results", 5)
        min_score = kwargs.get("min_score", 0.7)
        
        try:
            # Retrieve similar documents using the abstracted retriever
            search_results = await self.rag_retriever.retrieve(
                query=query,
                max_results=max_results
            )
            
            # Filter results by minimum score (domain logic)
            filtered_results = [
                result for result in search_results 
                if result.get("score", 0) >= min_score
            ]
            
            # Format results for context (domain formatting)
            knowledge_context = self._format_knowledge_context(filtered_results)
            
            return {
                "query": query,
                "knowledge_context": knowledge_context,
                "sources": self._extract_sources(filtered_results),
                "total_results": len(filtered_results),
                "confidence_scores": [r.get("score", 0) for r in filtered_results]
            }
            
        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            raise
    
    def _format_knowledge_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into knowledge context (domain-specific formatting)."""
        if not results:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu nội bộ."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            source = result.get("metadata", {}).get("title", "Unknown")
            score = result.get("score", 0)
            
            context_parts.append(
                f"[Nguồn {i}: {source} (độ tin cậy: {score:.2f})]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from results (domain extraction logic)."""
        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            sources.append({
                "title": metadata.get("title", "Unknown"),
                "source": metadata.get("source", ""),
                "document_type": metadata.get("document_type", ""),
                "score": result.get("score", 0)
            })
        return sources