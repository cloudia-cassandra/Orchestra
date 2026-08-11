"""Supervisor Agent: decomposes the task, tracks wave progress, synthesizes the final result.

Dispatch itself — fanning out to whichever steps are ready this wave — lives in the graph's
conditional edge (orchestration/graph.py, using orchestration/waves.py), since LangGraph's
`Send` mechanism only works from a conditional edge function, not from inside a node. This
agent's job is everything else: producing the plan, and deciding when the plan is done.
"""

from pydantic import ValidationError

from orchestra.agents.base import BaseAgent
from orchestra.hitl.triggers import PLAN_CONFIDENCE_THRESHOLD, classify_sensitivity
from orchestra.memory.long_term_memory import LongTermMemory
from orchestra.memory.working_memory import WorkingMemory
from orchestra.orchestration.schemas import ExecutionPlan
from orchestra.orchestration.state import OrchestraState
from orchestra.orchestration.waves import is_plan_complete

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
- confidence is your honest 0.0-1.0 estimate that this plan, executed as written, will actually \
satisfy the task. Don't inflate it — a low-confidence plan is routed to a human before any work \
starts instead of being run anyway, which is cheaper than a confidently wrong one.
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
        "confidence": {
            "type": "number",
            "description": "Honest 0.0-1.0 confidence that this plan will satisfy the task.",
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
    "required": ["reasoning", "confidence", "steps"],
}

_MAX_PLANNING_ATTEMPTS = 3


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self, model: str | None = None, memory: LongTermMemory | None = None):
        super().__init__(model)
        self.memory = memory or LongTermMemory()

    def __call__(self, state: OrchestraState) -> dict:
        if state.get("plan") is None:
            return self._plan(state)
        if state.get("status") in ("needs_escalation", "rejected"):
            # Resuming a task (hitl/resume.py) always re-enters here via intake -> supervisor.
            # A rejected plan-level escalation has nothing left to run — no-op and let
            # route_after_supervisor send it straight to end.
            return {}
        if is_plan_complete(state):
            return {"final_output": self._synthesize(state), "status": "complete"}
        return {"status": "executing"}

    def _plan(self, state: OrchestraState) -> dict:
        memory_section = self._relevant_memory_section(state["task"], state.get("user_id"))
        user_prompt = f"Task: {state['task']}{memory_section}"
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

        working_memory = WorkingMemory(state["task_id"])
        working_memory.set_plan(plan)

        base_update = {
            "plan": plan,
            "step_progress": {},
            "completed_step_ids": [],
            "pending_results": [],
            "specialist_results": [],
            "review_history": [],
        }

        escalation = self._check_plan_triggers(state, plan)
        if escalation is not None:
            working_memory.append_error_log(
                escalation["step_id"], f"escalated: {escalation['reason']}", feedback=escalation["feedback"]
            )
            return {**base_update, "status": "needs_escalation", "escalations": [escalation]}

        return {**base_update, "status": "executing", "escalations": []}

    def _check_plan_triggers(self, state: OrchestraState, plan: ExecutionPlan) -> dict | None:
        """Phase 3.1's plan-level escalation triggers, checked once before any specialist runs:
        an explicit human-review request always wins, a sensitive step is checked next (a
        confident plan to delete data should still escalate), and low plan confidence last."""
        if state.get("user_requested_review"):
            return {
                "step_id": "plan",
                "attempt": 0,
                "reason": "user_requested",
                "feedback": "Human review was requested for this task.",
                "confidence": plan.confidence,
            }

        for step in plan.steps:
            category = classify_sensitivity(step)
            if category is not None:
                return {
                    "step_id": step.step_id,
                    "attempt": 0,
                    "reason": "sensitive_operation",
                    "feedback": f"Step {step.step_id!r} looks like it involves {category}.",
                    "confidence": plan.confidence,
                }

        if plan.confidence < PLAN_CONFIDENCE_THRESHOLD:
            return {
                "step_id": "plan",
                "attempt": 0,
                "reason": "low_plan_confidence",
                "feedback": plan.reasoning,
                "confidence": plan.confidence,
            }

        return None

    def _relevant_memory_section(self, task: str, user_id: str | None) -> str:
        # Long-term memory (Phase 2.2/2.3) — similar past tasks, whether their approach worked or
        # led to an escalation, the plans they used, and any facts/preferences observed. Best-
        # effort: an empty or unreachable memory store just means no context, not a planning
        # failure — and a memory record is never binding, just a prior to weigh. Scoped to this
        # user (Phase 2.4) so one person's history doesn't leak into another's plans.
        records = self.memory.query(task, user_id=user_id)
        if not records:
            return ""
        entries = "\n\n".join(self._format_memory_record(r) for r in records)
        return (
            f"\n\nRelevant past experience, for context (not necessarily binding — weigh "
            f"'succeeded' approaches as a starting point, and 'escalated' ones as something to "
            f"do differently):\n{entries}"
        )

    @staticmethod
    def _format_memory_record(record) -> str:
        lines = [f"- Past task ({record.outcome}): {record.task!r}"]
        lines.append(f"  approach: {record.approach_summary}")
        if record.plan_steps:
            lines.append(f"  plan used: {'; '.join(record.plan_steps)}")
        if record.domain_facts:
            lines.append(f"  facts discovered: {'; '.join(record.domain_facts)}")
        if record.user_preferences:
            lines.append(f"  preferences observed: {'; '.join(record.user_preferences)}")
        return "\n".join(lines)

    def _synthesize(self, state: OrchestraState) -> str:
        results = "\n\n".join(
            f"[{r.domain} / {r.step_id}] {r.output}" for r in state["specialist_results"]
        )
        return self._call_llm(
            SYNTHESIS_PROMPT,
            f"Original task: {state['task']}\n\nSpecialist results:\n{results}",
        )
