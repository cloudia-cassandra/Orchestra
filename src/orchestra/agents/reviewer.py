"""Reviewer Agent: validates every specialist output produced this wave before the supervisor
decides what happens next.

Because steps can run in parallel, a single invocation may need to judge several pending
results at once (one per step that finished this wave) — LangGraph fans multiple specialist
branches back into one reviewer call, so this agent processes them as a batch rather than one
at a time.
"""

import json

from orchestra.agents.base import BaseAgent, extract_json
from orchestra.orchestration.schemas import PlanStep, ReviewVerdict, SpecialistResult
from orchestra.orchestration.state import OrchestraState

MAX_ATTEMPTS = 3
LOW_CONFIDENCE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are the Reviewer Agent. You receive a plan step's description, its \
expected output format, and a specialist's output. Judge whether the output actually satisfies \
the description AND matches the expected format: correct, complete, and usable as-is. Be strict \
but fair — do not reject for style alone.

Respond with ONLY a JSON object of this shape, no other text:
{"approved": true|false, "confidence": 0.0-1.0, "feedback": "<required if rejected, else null>"}"""


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def __call__(self, state: OrchestraState) -> dict:
        plan = state["plan"]
        steps_by_id = {s.step_id: s for s in plan.steps}
        already_reviewed = {(v.step_id, v.attempt) for v in state.get("review_history", [])}

        to_review = [
            result
            for result in state.get("pending_results", [])
            if (result.step_id, result.attempt) not in already_reviewed
        ]

        review_history: list[ReviewVerdict] = []
        specialist_results: list[SpecialistResult] = []
        completed_step_ids: list[str] = []
        escalations: list[dict] = []
        step_progress: dict[str, dict] = {}

        for result in to_review:
            step = steps_by_id[result.step_id]
            verdict = self._review_one(step, result)
            review_history.append(verdict)

            low_confidence = verdict.confidence < LOW_CONFIDENCE_THRESHOLD
            exhausted = not verdict.approved and result.attempt >= MAX_ATTEMPTS

            if low_confidence or exhausted:
                reason = "low_confidence" if low_confidence else "max_attempts_exceeded"
                step_progress[step.step_id] = {"escalated": True}
                escalations.append(
                    {
                        "step_id": step.step_id,
                        "attempt": result.attempt,
                        "reason": reason,
                        "feedback": verdict.feedback,
                        "confidence": verdict.confidence,
                    }
                )
            elif verdict.approved:
                specialist_results.append(result)
                completed_step_ids.append(step.step_id)

        return {
            "review_history": review_history,
            "specialist_results": specialist_results,
            "completed_step_ids": completed_step_ids,
            "escalations": escalations,
            "step_progress": step_progress,
            "status": "needs_escalation" if escalations else state.get("status", "executing"),
        }

    def _review_one(self, step: PlanStep, result: SpecialistResult) -> ReviewVerdict:
        raw = self._call_llm(
            SYSTEM_PROMPT,
            f"Step description: {step.description}\n"
            f"Expected output format: {step.expected_output_format}\n\n"
            f"Specialist output:\n{result.output}",
        )
        payload = json.loads(extract_json(raw))
        return ReviewVerdict(step_id=step.step_id, attempt=result.attempt, **payload)
