# src/stock_assistant/api/dependencies/services.py
from typing import Dict
from fastapi import Depends

from ...domain.agents.react_agent import StockReActAgent
from ...domain.coordinators.react_coordinator import ReActCoordinator
from ...domain.tools.base import BaseTool
from ...domain.tools import rag_tool, tavily_search_tool, chat_tool

from ...infra.chats.base import ChatProvider
from ...infra.chats.openai_chat import OpenAIChatProvider

from ...application.services.stock_analysis_service import StockAnalysisService


def get_tools() -> Dict[str, BaseTool]:
    # Tạo tools dict - dựa trên prompt ở react_agent.py
    return {
        "rag_knowledge": rag_tool,  
        "tavily_search": tavily_search_tool,
        "chat_llm": chat_tool
    }

def get_chat_provider() -> ChatProvider:
    return OpenAIChatProvider()

def get_react_agent(tools: Dict[str, BaseTool] = Depends(get_tools)) -> StockReActAgent:
    return StockReActAgent(tools=tools)  # Instantiate StockReActAgent

def get_react_coordinator(
    agent: StockReActAgent = Depends(get_react_agent),
    chat_provider: ChatProvider = Depends(get_chat_provider),
    tools: Dict[str, BaseTool] = Depends(get_tools)
) -> ReActCoordinator:
    return ReActCoordinator(agent=agent, chat_provider=chat_provider, tools=tools)

def get_stock_analysis_service(
    coordinator: ReActCoordinator = Depends(get_react_coordinator)
) -> StockAnalysisService:
    """Dependency for injecting StockAnalysisService."""
    return StockAnalysisService(coordinator=coordinator)
