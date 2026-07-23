"""Shared base class for every agent node in the graph."""

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from anthropic import Anthropic

from orchestra.orchestration.state import OrchestraState

if TYPE_CHECKING:
    from orchestra.orchestration.schemas import Domain
    from orchestra.tools.registry import ToolRegistry, ToolSpec

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def extract_json(text: str) -> str:
    """Pull the first top-level JSON object out of a model response."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return text[start : end + 1]


class BaseAgent(ABC):
    """A LangGraph node: reads OrchestraState, returns a partial state update."""

    name: str
    model: str = "claude-sonnet-5"

    def __init__(self, model: str | None = None):
        if model:
            self.model = model

    def _call_llm(self, system: str, user: str, max_tokens: int = 1536) -> str:
        response = get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def _call_structured(
        self,
        system: str,
        user: str,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        max_tokens: int = 2048,
    ) -> dict:
        """Force the model to respond via a single tool call, returning its input dict.

        This is more reliable than asking the model to emit JSON in prose and regexing
        it back out — the API enforces the schema's shape at the message level.
        """
        response = get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=[
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": input_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        tool_use = next(block for block in response.content if block.type == "tool_use")
        return tool_use.input

    def _call_with_tools(
        self,
        system: str,
        user: str,
        tools: list["ToolSpec"],
        registry: "ToolRegistry",
        domain: "Domain",
        max_iterations: int = 5,
        max_tokens: int = 1536,
    ) -> tuple[str, list[str]]:
        """Run an agentic tool-use loop: let the model call registered tools until it's done.

        Every tool call is routed through `registry.invoke()`, so authorization, rate
        limiting, and invocation logging all apply exactly as they would to any other caller.
        Returns (final_text, names_of_tools_called).
        """
        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        messages: list[dict] = [{"role": "user", "content": user}]
        tools_called: list[str] = []
        response = None

        for _ in range(max_iterations):
            response = get_client().messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                tools=anthropic_tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tools_called.append(block.name)
                try:
                    output = registry.invoke(block.name, domain=domain, **block.input)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
                    )
                except Exception as exc:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
        else:
            # Ran out of iterations still asking for tools — return whatever text is there.
            pass

        final_text = "".join(b.text for b in response.content if b.type == "text") if response else ""
        return final_text, tools_called

    @abstractmethod
    def __call__(self, state: OrchestraState) -> dict:
        ...
