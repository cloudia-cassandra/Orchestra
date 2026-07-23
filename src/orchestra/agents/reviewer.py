"""Reviewer Agent: validates a specialist's output before it reaches the supervisor."""

import json

from orchestra.agents.base import BaseAgent, extract_json
from orchestra.orchestration.schemas import ReviewVerdict
from orchestra.orchestration.state import OrchestraState

MAX_RETRIES = 2

SYSTEM_PROMPT = """You are the Reviewer Agent. You receive a plan step's instruction and a \
specialist's output for it. Judge whether the output actually satisfies the instruction: \
correct, complete, and usable as-is. Be strict but fair — do not reject for style alone.

Respond with ONLY a JSON object of this shape, no other text:
{"approved": true|false, "confidence": 0.0-1.0, "feedback": "<required if rejected, else null>"}"""


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def __call__(self, state: OrchestraState) -> dict:
        plan = state["plan"]
        step = plan.steps[state["current_step_index"]]
        result = state["pending_result"]

        raw = self._call_llm(
            SYSTEM_PROMPT,
            f"Step instruction: {step.instruction}\n\nSpecialist output:\n{result.output}",
        )
        payload = json.loads(extract_json(raw))
        verdict = ReviewVerdict(step_id=step.step_id, **payload)

        if verdict.approved:
            return {
                "specialist_results": [result],
                "review_history": [verdict],
                "pending_result": None,
                "status": "delegating",
            }

        retry_count = state.get("retry_count", 0) + 1
        if retry_count > MAX_RETRIES:
            # Specialist can't satisfy the reviewer after MAX_RETRIES attempts.
            # Phase 3 (Human-in-the-Loop) hooks into this status to escalate.
            return {"review_history": [verdict], "status": "needs_escalation"}

        return {
            "review_history": [verdict],
            "retry_count": retry_count,
            "status": "retrying",
        }
