"""Phase 3.2: apply a human reviewer's decision to a paused task and continue it from where it
left off.

Resuming doesn't need LangGraph's checkpointer machinery — the graph was already built (Phase
1.4) so that everything needed to keep going lives in `OrchestraState` itself, not in call-stack
position. So "continue" here just means: reconstruct that state from the approval request's
snapshot, fold the human's decision into it the same way an automated approval/rejection would
have, and call `graph.invoke()` again. `intake_node` and `SupervisorAgent` both already know not
to re-plan or reset progress when a plan is already present in the state they're handed.

Three decisions, two different shapes depending on whether the escalation was step-level
(specialist_failed_twice / low_quality_score — a specific step's output is in question) or
plan-level (low_plan_confidence / sensitive_operation / user_requested — nothing has run yet):

- **approved** (step-level): the proposed output is accepted as-is, step marked complete.
- **approved** (plan-level): run the plan as originally proposed.
- **rejected** (step-level): sent back for another attempt, seeded with the reviewer's own
  feedback — the same channel the automated reviewer's rejections already use.
- **rejected** (plan-level): the task ends outright. There's no partial work to retry, and
  re-planning from scratch isn't a feature this has — rejecting the plan rejects the task.
- **modified** (step-level): the reviewer's edited output is accepted in place of the proposed
  one, step marked complete.
- **modified** (plan-level): not supported — editing a whole plan isn't a feature this has
  either. Raises clearly rather than silently doing something plausible-looking instead.
"""

from orchestra.hitl.approval_queue import ApprovalQueue, ApprovalRequest
from orchestra.memory.working_memory import WorkingMemory
from orchestra.orchestration.graph import build_graph
from orchestra.orchestration.schemas import ExecutionPlan, ReviewVerdict, SpecialistResult
from orchestra.orchestration.state import OrchestraState


def _deserialize_state(snapshot: dict) -> OrchestraState:
    return {
        "task": snapshot["task"],
        "task_id": snapshot["task_id"],
        "user_id": snapshot["user_id"],
        "plan": ExecutionPlan.model_validate(snapshot["plan"]),
        "step_progress": dict(snapshot["step_progress"]),
        "completed_step_ids": list(snapshot["completed_step_ids"]),
        "pending_results": [SpecialistResult.model_validate(r) for r in snapshot["pending_results"]],
        "specialist_results": [SpecialistResult.model_validate(r) for r in snapshot["specialist_results"]],
        "review_history": [ReviewVerdict.model_validate(v) for v in snapshot["review_history"]],
        "escalations": list(snapshot["escalations"]),
    }


def resume_task(request_id: str, queue: ApprovalQueue | None = None) -> dict:
    """Apply the decision already recorded on `request_id` (call `ApprovalQueue.decide()`
    first) and, unless the task was terminated outright, run the graph forward from where it
    paused. Returns the graph's final state, same shape as a fresh `graph.invoke()`."""
    queue = queue or ApprovalQueue()
    request = queue.get(request_id)
    if request is None:
        raise ValueError(f"No approval request found for id={request_id!r}")
    if request.status == "pending":
        raise ValueError("Cannot resume a request that hasn't been decided yet — call decide() first.")

    state = _deserialize_state(request.state_snapshot)
    working_memory = WorkingMemory(state["task_id"])

    if request.current_step_id in (None, "plan"):
        state = _apply_plan_level_decision(state, request, working_memory)
    else:
        state = _apply_step_level_decision(state, request, working_memory)

    graph = build_graph()
    return graph.invoke(state, config={"recursion_limit": 50})


def _apply_plan_level_decision(
    state: OrchestraState, request: ApprovalRequest, working_memory: WorkingMemory
) -> OrchestraState:
    if request.decision == "modified":
        raise ValueError(
            "Plan-level escalations only support approve or reject — editing a whole plan "
            "isn't supported yet."
        )
    if request.decision == "approved":
        state["status"] = "executing"
    else:  # rejected
        state["status"] = "rejected"
        working_memory.append_error_log(
            None, "task rejected by human reviewer", notes=request.reviewer_notes
        )
    return state


def _apply_step_level_decision(
    state: OrchestraState, request: ApprovalRequest, working_memory: WorkingMemory
) -> OrchestraState:
    step_id = request.current_step_id
    attempt = request.escalation_detail.get("attempt")
    state["step_progress"] = {
        **state["step_progress"],
        step_id: {**state["step_progress"].get(step_id, {}), "escalated": False},
    }

    if request.decision == "rejected":
        verdict = ReviewVerdict(
            step_id=step_id,
            attempt=attempt,
            approved=False,
            confidence=0.0,
            feedback=request.reviewer_notes or "Rejected by human reviewer.",
        )
        state["review_history"] = [*state["review_history"], verdict]
        state["status"] = "executing"
        return state

    output = request.modified_output if request.decision == "modified" else request.proposed_action
    domain = next(s.domain for s in state["plan"].steps if s.step_id == step_id)
    result = SpecialistResult(
        step_id=step_id, domain=domain, attempt=attempt, output=output or "", confidence=1.0, tool_calls=[]
    )
    state["specialist_results"] = [*state["specialist_results"], result]
    state["completed_step_ids"] = [*state["completed_step_ids"], step_id]
    working_memory.add_completed_output(result)

    verdict = ReviewVerdict(
        step_id=step_id,
        attempt=attempt,
        approved=True,
        confidence=1.0,
        feedback=f"Approved by human reviewer ({request.decision}).",
    )
    state["review_history"] = [*state["review_history"], verdict]
    state["status"] = "executing"
    return state
