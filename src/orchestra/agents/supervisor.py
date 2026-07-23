"""Supervisor Agent: plans the task, delegates one step at a time, synthesizes the result."""

from orchestra.agents.base import BaseAgent, extract_json
from orchestra.orchestration.schemas import ExecutionPlan
from orchestra.orchestration.state import OrchestraState

PLANNING_PROMPT = """You are the Supervisor Agent in a multi-agent system. Break the user's task \
into an ordered list of steps. Each step must be assigned to exactly one specialist domain: \
research, data_analysis, writing, or code_execution. Keep steps minimal but complete.

Respond with ONLY a JSON object of this shape, no other text:
{"reasoning": "<why you split it this way>", \
"steps": [{"step_id": "s1", "domain": "research", "instruction": "...", "depends_on": []}]}"""

SYNTHESIS_PROMPT = """You are the Supervisor Agent. Combine the specialist results below into a \
single, coherent final answer to the original task. Do not mention the internal step process."""


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __call__(self, state: OrchestraState) -> dict:
        if state.get("plan") is None:
            return self._plan(state)
        return self._advance(state)

    def _plan(self, state: OrchestraState) -> dict:
        raw = self._call_llm(PLANNING_PROMPT, state["task"])
        plan = ExecutionPlan.model_validate_json(extract_json(raw))
        return {
            "plan": plan,
            "current_step_index": 0,
            "retry_count": 0,
            "specialist_results": [],
            "review_history": [],
            "status": "delegating",
        }

    def _advance(self, state: OrchestraState) -> dict:
        plan = state["plan"]
        next_index = state["current_step_index"] + 1
        if next_index >= len(plan.steps):
            return {"final_output": self._synthesize(state), "status": "complete"}
        return {"current_step_index": next_index, "retry_count": 0, "status": "delegating"}

    def _synthesize(self, state: OrchestraState) -> str:
        results = "\n\n".join(
            f"[{r.domain} / {r.step_id}] {r.output}" for r in state["specialist_results"]
        )
        return self._call_llm(
            SYNTHESIS_PROMPT,
            f"Original task: {state['task']}\n\nSpecialist results:\n{results}",
        )
