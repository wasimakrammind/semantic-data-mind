import json
from typing import Optional

import anthropic
from loguru import logger

from config.settings import Settings
from src.agent.tool_definitions import get_all_tool_definitions
from src.agent.tool_registry import ToolRegistry, create_tool_registry
from src.agent.context_manager import ContextManager
from src.agent.conversation import ConversationState
from src.models.analysis_result import AgentResponse


class AgentOrchestrator:
    """Main agent loop implementing a ReAct pattern using Claude's tool_use API."""

    MAX_TOOL_ITERATIONS = 10

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.client = client
        self.settings = settings
        self.tool_registry = tool_registry or create_tool_registry(client)
        self.context_manager = ContextManager()
        self.tool_definitions = get_all_tool_definitions()

    def process_message(
        self, user_message: str, session: ConversationState
    ) -> AgentResponse:
        """Process a user message through the agent loop.

        1. Build system prompt with dataset context
        2. Send to Claude with tools
        3. If Claude calls a tool, execute it and loop
        4. If Claude returns text, return as response
        """
        session.add_user_message(user_message)
        system_prompt = self.context_manager.build_system_prompt(session)

        # Build initial messages for API
        messages = self.context_manager.get_messages_for_api(session, user_message)

        all_charts = []
        tool_calls_made = []

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            logger.info(f"Agent iteration {iteration + 1}")

            try:
                response = self.client.messages.create(
                    model=self.settings.anthropic_model,
                    max_tokens=self.settings.anthropic_max_tokens,
                    system=system_prompt,
                    tools=self.tool_definitions,
                    messages=messages,
                )
            except anthropic.AuthenticationError as e:
                logger.error(f"Anthropic auth error: {e}")
                error_msg = (
                    "**Authentication failed.** Your Anthropic API key is invalid or expired.\n\n"
                    "Please check your API key and try again. You can update it in the sidebar "
                    "or in your `.env` file."
                )
                session.add_assistant_message(error_msg)
                return AgentResponse(text=error_msg)
            except anthropic.RateLimitError as e:
                logger.error(f"Rate limit error: {e}")
                error_msg = (
                    "**Rate limit reached.** The API is temporarily throttled. "
                    "Please wait a moment and try again."
                )
                session.add_assistant_message(error_msg)
                return AgentResponse(text=error_msg)
            except anthropic.APIConnectionError as e:
                logger.error(f"API connection error: {e}")
                error_msg = (
                    "**Connection error.** Could not reach the Anthropic API. "
                    "Please check your internet connection and try again."
                )
                session.add_assistant_message(error_msg)
                return AgentResponse(text=error_msg)
            except anthropic.APIError as e:
                logger.error(f"Anthropic API error: {e}")
                error_msg = f"**API Error:** {e}\n\nPlease try again or rephrase your question."
                session.add_assistant_message(error_msg)
                return AgentResponse(text=error_msg)
            except Exception as e:
                logger.error(f"Unexpected error in agent loop: {e}")
                error_msg = f"**Unexpected error:** {e}\n\nPlease try again."
                session.add_assistant_message(error_msg)
                return AgentResponse(text=error_msg)

            # Process the response
            assistant_content = response.content
            stop_reason = response.stop_reason

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": assistant_content})

            if stop_reason == "end_turn":
                # Extract text from response
                text_parts = []
                for block in assistant_content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)

                final_text = "\n".join(text_parts)

                # Collect any charts generated during this response
                if session.charts:
                    all_charts.extend(session.charts)
                    session.charts = []

                session.add_assistant_message(final_text, all_charts)
                return AgentResponse(
                    text=final_text,
                    charts=all_charts,
                    tool_calls_made=tool_calls_made,
                )

            elif stop_reason == "tool_use":
                # Execute all tool calls in the response
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        logger.info(f"Executing tool: {tool_name}")
                        tool_calls_made.append(tool_name)

                        result = self.tool_registry.execute(
                            tool_name, tool_input, session
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                        })

                # Add tool results as user message (API format)
                messages.append({"role": "user", "content": tool_results})

            else:
                logger.warning(f"Unexpected stop_reason: {stop_reason}")
                break

        # If we hit max iterations
        fallback_msg = "I've completed my analysis. Let me know if you'd like me to look into anything specific."
        session.add_assistant_message(fallback_msg, all_charts)
        return AgentResponse(
            text=fallback_msg,
            charts=all_charts,
            tool_calls_made=tool_calls_made,
        )
