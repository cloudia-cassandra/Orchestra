"""Generic specialist node: executes the current plan step within its domain."""

from orchestra.agents.base import BaseAgent
from orchestra.memory.working_memory import WorkingMemory
from orchestra.orchestration.schemas import Domain, PlanStep, SpecialistResult
from orchestra.orchestration.state import OrchestraState
from orchestra.tools.builtin import registry as default_registry


class SpecialistAgent(BaseAgent):
    domain: Domain
    system_prompt: str
    # Names of tools (from the ToolRegistry) this specialist is allowed to call. Empty means
    # LLM-only, no tool loop.
    tools: list[str] = []

    def __call__(self, state: OrchestraState) -> dict:
        plan = state["plan"]
        step = next(s for s in plan.steps if s.step_id == state["active_step_id"])
        attempt = state.get("step_progress", {}).get(step.step_id, {}).get("attempts", 0) + 1

        user_prompt = self._build_prompt(state, step)
        feedback = self._latest_rejection_feedback(state, step.step_id)
        if feedback:
            user_prompt += (
                f"\n\nA reviewer rejected your previous attempt at this step. "
                f"Address this feedback — try a different approach, not the same one again:\n"
                f"{feedback}"
            )

        memory = WorkingMemory(state["task_id"])

        tool_calls: list[str] = []
        try:
            if self.tools:
                tool_specs = [default_registry.get(name) for name in self.tools]
                output, tool_calls = self._call_with_tools(
                    self.system_prompt, user_prompt, tool_specs, default_registry, self.domain
                )
            else:
                output = self._call_llm(self.system_prompt, user_prompt)
        except Exception as exc:
            memory.append_error_log(step.step_id, str(exc), attempt=attempt, domain=self.domain)
            raise

        result = SpecialistResult(
            step_id=step.step_id,
            domain=self.domain,
            attempt=attempt,
            output=output,
            confidence=0.8,
            tool_calls=tool_calls,
        )
        memory.add_intermediate_result(result)
        return {
            "pending_results": [result],
            "step_progress": {step.step_id: {"attempts": attempt}},
        }

    def _build_prompt(self, state: OrchestraState, step: PlanStep) -> str:
        parts = [f"Task: {step.description}"]

        if step.depends_on:
            dependency_outputs = self._dependency_outputs(state, step.depends_on)
            parts.append("Outputs from prior steps this depends on:\n" + dependency_outputs)

        if step.required_inputs:
            parts.append("Required inputs:\n- " + "\n- ".join(step.required_inputs))

        parts.append(f"Expected output format: {step.expected_output_format}")
        return "\n\n".join(parts)

    @staticmethod
    def _dependency_outputs(state: OrchestraState, depends_on: list[str]) -> str:
        results_by_id = {r.step_id: r for r in state.get("specialist_results", [])}
        blocks = []
        for step_id in depends_on:
            result = results_by_id.get(step_id)
            if result is not None:
                blocks.append(f"[{step_id}] {result.output}")
        return "\n\n".join(blocks)

    @staticmethod
    def _latest_rejection_feedback(state: OrchestraState, step_id: str) -> str | None:
        for verdict in reversed(state.get("review_history", [])):
            if verdict.step_id == step_id and not verdict.approved:
                return verdict.feedback
        return None
