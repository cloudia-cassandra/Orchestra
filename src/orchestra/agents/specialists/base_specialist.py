"""Generic specialist node: executes the current plan step within its domain."""

from orchestra.agents.base import BaseAgent
from orchestra.orchestration.schemas import Domain, SpecialistResult
from orchestra.orchestration.state import OrchestraState


class SpecialistAgent(BaseAgent):
    domain: Domain
    system_prompt: str
    # Domain-specific tools get bound here once the tool framework (MCP + custom) lands;
    # specialists are LLM-only for now.
    tools: list[str] = []

    def __call__(self, state: OrchestraState) -> dict:
        plan = state["plan"]
        step = plan.steps[state["current_step_index"]]

        user_prompt = step.instruction
        feedback = self._latest_rejection_feedback(state, step.step_id)
        if feedback:
            user_prompt += (
                f"\n\nA reviewer rejected your previous attempt at this step. "
                f"Address this feedback:\n{feedback}"
            )

        output = self._call_llm(self.system_prompt, user_prompt)
        result = SpecialistResult(
            step_id=step.step_id,
            domain=self.domain,
            output=output,
            confidence=0.8,
        )
        return {"pending_result": result, "status": "reviewing"}

    @staticmethod
    def _latest_rejection_feedback(state: OrchestraState, step_id: str) -> str | None:
        for verdict in reversed(state.get("review_history", [])):
            if verdict.step_id == step_id and not verdict.approved:
                return verdict.feedback
        return None
