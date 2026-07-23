"""Supervisor Agent: decomposes the task, delegates one step at a time, synthesizes the result."""

from pydantic import ValidationError

from orchestra.agents.base import BaseAgent
from orchestra.orchestration.schemas import ExecutionPlan
from orchestra.orchestration.state import OrchestraState

PLANNING_PROMPT = """You are the Supervisor Agent in a multi-agent system. Decompose the user's \
task into an ordered list of subtasks and submit it via the submit_execution_plan tool.

Rules for a valid plan:
- Each step is assigned to exactly one specialist domain: research, data_analysis, writing, \
or code_execution.
- List steps in dependency order: if step B needs step A's output, A must appear before B in \
the list, and B must list A's step_id in depends_on.
- required_inputs should name the concrete inputs the step needs, calling out which prior \
step_id each one comes from (e.g. "s1.output: extracted statistics").
- expected_output_format should describe the shape of the output precisely enough that the \
next step (or the final synthesis) knows what to expect.
- estimated_complexity is your honest estimate of how hard the step is: low, medium, or high.
- Keep the plan minimal — no step that isn't necessary to answer the task."""

SYNTHESIS_PROMPT = """You are the Supervisor Agent. Combine the specialist results below into a \
single, coherent final answer to the original task. Do not mention the internal step process."""

_PLAN_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Why the task was split this way.",
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "domain": {
                        "type": "string",
                        "enum": ["research", "data_analysis", "writing", "code_execution"],
                    },
                    "description": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "required_inputs": {"type": "array", "items": {"type": "string"}},
                    "expected_output_format": {"type": "string"},
                    "estimated_complexity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": [
                    "step_id",
                    "domain",
                    "description",
                    "depends_on",
                    "required_inputs",
                    "expected_output_format",
                    "estimated_complexity",
                ],
            },
        },
    },
    "required": ["reasoning", "steps"],
}

_MAX_PLANNING_ATTEMPTS = 3


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __call__(self, state: OrchestraState) -> dict:
        if state.get("plan") is None:
            return self._plan(state)
        return self._advance(state)

    def _plan(self, state: OrchestraState) -> dict:
        user_prompt = f"Task: {state['task']}"
        last_error: ValidationError | None = None

        for _ in range(_MAX_PLANNING_ATTEMPTS):
            if last_error is not None:
                user_prompt = (
                    f"Task: {state['task']}\n\n"
                    f"Your previous plan was invalid: {last_error}\n"
                    f"Submit a corrected plan."
                )
            raw_plan = self._call_structured(
                system=PLANNING_PROMPT,
                user=user_prompt,
                tool_name="submit_execution_plan",
                tool_description="Submit the ordered execution plan for the task.",
                input_schema=_PLAN_TOOL_SCHEMA,
            )
            try:
                plan = ExecutionPlan.model_validate(raw_plan)
                break
            except ValidationError as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Supervisor could not produce a valid execution plan after "
                f"{_MAX_PLANNING_ATTEMPTS} attempts: {last_error}"
            )

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
