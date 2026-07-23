"""Shared base class for every agent node in the graph."""

import os
from abc import ABC, abstractmethod

from anthropic import Anthropic

from orchestra.orchestration.state import OrchestraState

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

    @abstractmethod
    def __call__(self, state: OrchestraState) -> dict:
        ...
