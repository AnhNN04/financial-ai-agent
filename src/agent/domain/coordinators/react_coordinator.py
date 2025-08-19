# src/stock_assistant/domain/coordinators/react_coordinator.py
from typing import Dict, Any
from ..entities.query_context import QueryContext, AgentState
from ..agents.react_agent import ReActAgent
from ..tools.base import BaseTool
from ...infra.chats.base import ChatProvider
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class ReActCoordinator:
    """Coordinator for orchestrating ReAct Agent workflow."""
    
    def __init__(self, agent: ReActAgent, chat_provider: ChatProvider, tools: Dict[str, BaseTool]):
        self.agent = agent
        self.chat_provider = chat_provider
        self.tools = tools
    
    # async def execute(self, query: str, max_steps: int = 10) -> Dict[str, Any]:
    #     """Execute the ReAct workflow for a given query."""
    #     try:
    #         state = AgentState(
    #             messages=[{"role": "user", "content": query}],
    #             current_step=0,
    #             max_steps=max_steps,
    #             tools_used=[],
    #             intermediate_results=[]
    #         )
            
    #         while state["current_step"] < state["max_steps"] and not state.get("final_answer"):
    #             state = await self._step(state)
            
    #         return {
    #             "answer": state.get("final_answer", "Không thể tìm ra câu trả lời."),
    #             "steps": state["current_step"],
    #             "tools_used": state.get("tools_used", []),
    #             "intermediate_results": state.get("intermediate_results", []),
    #             "success": bool(state.get("final_answer"))
    #         }
    #     except Exception as e:
    #         logger.error(f"Coordinator execution failed: {str(e)}")
    #         return {
    #             "answer": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
    #             "success": False
    #         }
    
    async def _step(self, state: AgentState) -> AgentState:
        """Execute a single step in the ReAct workflow."""
        # Reason step
        reasoning_result = await self.agent.reason(state)
        conversation = reasoning_result["messages"]
        current_step = reasoning_result["current_step"]
        
        if reasoning_result.get("final_answer"):
            return {
                **state,
                "messages": conversation,
                "current_step": current_step + 1,
                "final_answer": reasoning_result["final_answer"]
            }
        
        # Get LLM response
        response = await self.chat_provider.chat(
            prompt="\n".join([msg["content"] for msg in conversation]),
            model=None,
            temperature=0.1,
            max_tokens=2000
        )
        response_content = response["response"]
        
        # Update messages
        messages = conversation + [{"role": "assistant", "content": response_content}]
        
        if "Final Answer:" in response_content:
            final_answer = response_content.split("Final Answer:")[-1].strip()
            return {
                **state,
                "messages": messages,
                "current_step": current_step + 1,
                "final_answer": final_answer
            }
        
        # Parse tool usage
        tool_name, tool_input = self.agent.parse_tool_usage(response_content)
        if tool_name and tool_name in self.tools:
            context = QueryContext(query=tool_input)
            tool_result = await self.tools[tool_name].execute(context, query=tool_input)
            observation = self.agent.format_tool_result(tool_name, tool_result)
            
            return {
                **state,
                "messages": messages + [{"role": "user", "content": observation}],
                "current_step": current_step + 1,
                "tools_used": state.get("tools_used", []) + [tool_name],
                "intermediate_results": state.get("intermediate_results", []) + [tool_result]
            }
        
        return {
            **state,
            "messages": messages,
            "current_step": current_step + 1
        }