from typing import Optional, List, Dict, Any

from src.models.dataset import Dataset
from src.models.validation_result import ValidationReport
from src.models.analysis_result import EDAResult


class ConversationState:
    """Manages conversation history and session state."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.dataset: Optional[Dataset] = None
        self.validation_report: Optional[ValidationReport] = None
        self.analysis_history: List[EDAResult] = []
        self.charts: List[Any] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, charts: List = None) -> None:
        msg = {"role": "assistant", "content": content}
        if charts:
            msg["charts"] = charts
        self.messages.append(msg)

    def add_tool_use(self, tool_name: str, tool_input: Dict, tool_result: str) -> None:
        """Record a tool call in history (for context tracking)."""
        self.messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"tool_{len(self.messages)}",
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
        })
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"tool_{len(self.messages) - 1}",
                    "content": tool_result,
                }
            ],
        })

    def get_api_messages(self, max_turns: int = 10) -> List[Dict]:
        """Get recent messages in Anthropic API format."""
        # Filter to actual user/assistant text messages for context
        api_messages = []
        for msg in self.messages[-(max_turns * 2):]:
            api_messages.append(msg)
        return api_messages

    def get_display_messages(self) -> List[Dict]:
        """Get messages for UI display (text only, no tool internals)."""
        display = []
        for msg in self.messages:
            if isinstance(msg.get("content"), str):
                display.append(msg)
        return display
