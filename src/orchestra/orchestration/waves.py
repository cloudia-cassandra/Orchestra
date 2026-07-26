"""Wave computation: which plan steps are eligible to run right now.

A step is ready when it hasn't been completed or escalated yet, and every step it depends on
has been completed. Steps rejected by the reviewer (but not yet escalated) are ready again on
the next wave automatically — there is no separate "retry" bookkeeping, since not-completed
and not-escalated already means "still needs a successful attempt".
"""

from orchestra.orchestration.schemas import ExecutionPlan, PlanStep
from orchestra.orchestration.state import OrchestraState


def ready_steps(state: OrchestraState) -> list[PlanStep]:
    plan: ExecutionPlan = state["plan"]
    completed = set(state.get("completed_step_ids", []))
    progress = state.get("step_progress", {})

    ready = []
    for step in plan.steps:
        if step.step_id in completed:
            continue
        if progress.get(step.step_id, {}).get("escalated"):
            continue
        if set(step.depends_on) <= completed:
            ready.append(step)
    return ready


def is_plan_complete(state: OrchestraState) -> bool:
    plan: ExecutionPlan = state["plan"]
    return len(set(state.get("completed_step_ids", []))) == len(plan.steps)
