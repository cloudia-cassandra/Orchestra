"""Shared graph state threaded through every agent node.

Execution is wave-based: each pass, the supervisor fans out (via LangGraph `Send`) to every
plan step whose dependencies are satisfied and which hasn't been completed or escalated —
independent steps run in parallel within a wave, dependent steps naturally land in later waves.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from orchestra.orchestration.schemas import ExecutionPlan, ReviewVerdict, SpecialistResult

Status = Literal["planning", "executing", "needs_escalation", "rejected", "complete"]


def _merge_step_progress(left: dict[str, dict], right: dict[str, dict]) -> dict[str, dict]:
    """Shallow-merge per-step progress dicts. Distinct steps never collide within a wave;
    when a step_id does reappear (e.g. reviewer updating a step a specialist just touched),
    the later write's fields win, layered on top of the earlier ones."""
    merged = dict(left)
    for step_id, delta in right.items():
        merged[step_id] = {**merged.get(step_id, {}), **delta}
    return merged


class OrchestraState(TypedDict, total=False):
    task: str
    task_id: str
    user_id: str
    # Set by the caller at invoke time (Phase 3.1) to force human review regardless of how
    # confident the supervisor or reviewer end up being — e.g. graph.invoke({..., "user_requested_review": True}).
    user_requested_review: bool
    plan: ExecutionPlan | None
    status: Status

    # Which step a specialist node is currently handling — set per-invocation via Send payload.
    active_step_id: str | None

    # step_id -> {"attempts": int, "escalated": bool}. Append-only per key via merge, so
    # concurrent specialists in the same wave (different step_ids) never collide.
    step_progress: Annotated[dict[str, dict[str, Any]], _merge_step_progress]

    completed_step_ids: Annotated[list[str], operator.add]
    pending_results: Annotated[list[SpecialistResult], operator.add]
    specialist_results: Annotated[list[SpecialistResult], operator.add]
    review_history: Annotated[list[ReviewVerdict], operator.add]
    escalations: Annotated[list[dict[str, Any]], operator.add]

    final_output: str | None
