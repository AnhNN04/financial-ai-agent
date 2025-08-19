# src/stock_assistant/application/services/stock_analysis_service.py

from typing import Dict, Any, Optional
from ...domain.coordinators.react_coordinator import ReActCoordinator
from ...infra.agents.langgraph_workflow import ReActWorkflow
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class StockAnalysisService:
    """Application service for stock market analysis using ReAct workflow."""
    
    def __init__(self, coordinator: ReActCoordinator):
        self.coordinator = coordinator
        # Khởi tạo workflow LangGraph với coordinator
        self.workflow_manager = ReActWorkflow(coordinator=self.coordinator)

    async def analyze(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Phân tích truy vấn chứng khoán bằng cách sử dụng agent ReAct.
        """
        try:
            logger.info(f"Bắt đầu phân tích cho truy vấn: '{query}'")
            # Sử dụng phương thức run của ReActWorkflow để thực thi luồng
            final_state = await self.workflow_manager.run(query=query, session_id=session_id)
            
            # Trích xuất dữ liệu cần thiết từ trạng thái cuối cùng của workflow
            final_answer = final_state.get("final_answer")
            messages = final_state.get("messages", [])
            tools_used = final_state.get("tools_used", [])
            intermediate_results = final_state.get("intermediate_results", [])
            
            # Tạo một câu trả lời tổng hợp từ các bước trung gian
            combined_answer = ""
            for msg in messages:
                if msg.get("role") == "user":
                    continue
                combined_answer += msg.get("content", "") + "\n"

            # Trả về kết quả
            return {
                "answer": final_answer or combined_answer.strip(),
                "metadata": {
                    "success": True,
                    "session_id": session_id,
                    "steps": len(messages),
                    "tools_used": tools_used,
                    "intermediate_results": intermediate_results
                }
            }
        except Exception as e:
            logger.error(f"Phân tích thất bại cho truy vấn '{query}': {e}", exc_info=True)
            return {
                "answer": f"Xin lỗi, đã xảy ra lỗi trong quá trình phân tích: {str(e)}",
                "metadata": {
                    "success": False,
                    "session_id": session_id,
                    "steps": 0,
                    "tools_used": [],
                    "intermediate_results": []
                }
            }