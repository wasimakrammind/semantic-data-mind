from typing import List, Dict, Optional

from src.models.dataset import Dataset
from src.agent.conversation import ConversationState
from src.utils.token_counter import estimate_tokens, truncate_to_budget


class ContextManager:
    """Manages what goes into the LLM context window."""

    MAX_CONTEXT_TOKENS = 150_000
    SYSTEM_PROMPT_BUDGET = 5_000
    TOOL_DEFS_BUDGET = 5_000
    HISTORY_BUDGET = 20_000
    RESPONSE_BUDGET = 4_096

    def build_system_prompt(self, session: ConversationState) -> str:
        """Build the system prompt with dataset context."""
        parts = [self._base_system_prompt()]

        if session.dataset:
            parts.append("\n\n" + session.dataset.profile_summary)

            if session.validation_report:
                parts.append("\n\n" + session.validation_report.summary)

        return "\n".join(parts)

    def get_messages_for_api(
        self, session: ConversationState, new_message: str
    ) -> List[Dict]:
        """Build the messages array for the API call.

        Ensures proper alternating user/assistant format required by the API.
        """
        raw_messages = session.get_api_messages(max_turns=10)

        # Filter to only valid text messages (skip tool internals from past turns)
        messages = []
        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content")
            # Only include simple text messages
            if isinstance(content, str) and role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        # Ensure conversation starts with user and alternates properly
        cleaned = []
        for msg in messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                cleaned[-1]["content"] += "\n" + msg["content"]
            else:
                cleaned.append(msg.copy())

        # Ensure starts with user
        if not cleaned or cleaned[0]["role"] != "user":
            cleaned = [{"role": "user", "content": new_message}]
        elif cleaned[-1]["role"] == "user" and cleaned[-1]["content"] == new_message:
            # Already has the current message
            pass
        elif cleaned[-1]["role"] == "user":
            # Last message is user but different — append to it
            cleaned[-1]["content"] += "\n" + new_message
        else:
            # Last message is assistant — add user message
            cleaned.append({"role": "user", "content": new_message})

        return cleaned

    def _base_system_prompt(self) -> str:
        return """You are a Data Intelligence Agent — an expert data analyst powered by AI.

Your role is to help non-technical users understand their data through natural language conversation. You have access to tools for data validation, exploratory analysis, visualization, and statistical computing.

Guidelines:
- Always explain results in plain English that a business executive can understand
- When discussing numbers, provide context (e.g., "revenue increased 23%, from $1.2M to $1.5M")
- Create visualizations whenever they would help illustrate a point
- Proactively surface interesting patterns or potential data quality issues
- If you notice data quality problems, mention them before proceeding with analysis
- When uncertain, explain your reasoning and the limitations of the analysis
- Suggest follow-up questions the user might want to ask

When you first receive data, briefly describe what you see (shape, key columns, initial observations) and ask how you can help."""
