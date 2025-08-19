# src/stock_assistant/infra/agents/langgraph_workflow.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ...domain.entities.query_context import AgentState
from ...domain.coordinators.react_coordinator import ReActCoordinator
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class ReActWorkflow:
    """Infrastructure layer for building and running LangGraph workflow."""
    
    def __init__(self, coordinator: ReActCoordinator):
        self.coordinator = coordinator
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        def should_continue(state: AgentState) -> str:
            """Decide whether to continue or end the workflow."""
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if state.get("final_answer"):
                return "end"
            
            if state["current_step"] >= state["max_steps"]:
                return "end"
            
            if last_message and "Action:" in last_message.get("content", ""):
                return "continue"
            
            return "end"
        
        async def agent_node(state: AgentState) -> Dict[str, Any]:
            """Agent node calling coordinator's step method."""
            return await self.coordinator._step(state)
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "agent", "end": END}
        )
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the LangGraph workflow."""
        try:
            initial_state = AgentState(
                messages=[{"role": "user", "content": query}],
                current_step=0,
                max_steps=10,
                tools_used=[],
                intermediate_results=[]
            )
            config = {"configurable": {"thread_id": session_id or "default"}}
            
            final_state = None
            async for state in self.workflow.astream(initial_state, config=config):
                final_state = state
            
            return final_state["agent"] if final_state else {
                "messages": [{"role": "assistant", "content": "Không thể xử lý câu hỏi."}],
                "current_step": 0,
                "final_answer": None
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise