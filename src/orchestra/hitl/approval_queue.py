"""Phase 3.2: the approval queue. When an escalation trigger (Phase 3.1) fires, the graph
routes through the `approval_queue` node (orchestration/graph.py) before halting, which packages
everything a human needs to review — the original task, the plan, what's completed so far, the
step that needs a decision, and the agent's proposed action — into an `ApprovalRequest` row and
pushes it here. `hitl/resume.py` is the other half: it reads a decided request back out and
continues the paused task.

Backed by Postgres (see `hitl/db.py`) rather than Redis or Chroma — a review queue is genuinely
relational: "every pending request for this user," "has this exact escalation already been
queued," "update this row's decision" are queries and updates, not a key-value read or a
semantic search.
"""

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, Text, insert, select, update
from sqlalchemy.engine import Engine

from orchestra.hitl import db
from orchestra.orchestration.state import OrchestraState

Decision = Literal["approved", "rejected", "modified"]
RequestStatus = Literal["pending", "approved", "rejected", "modified"]

metadata = MetaData()

approval_requests_table = Table(
    "approval_requests",
    metadata,
    Column("id", String, primary_key=True),
    Column("task_id", String, nullable=False),
    Column("user_id", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("reason", String, nullable=False),
    Column("status", String, nullable=False, default="pending"),
    Column("original_task", Text, nullable=False),
    Column("current_step_id", String, nullable=True),
    Column("proposed_action", Text, nullable=True),
    Column("escalation_detail", JSON, nullable=False),
    Column("state_snapshot", JSON, nullable=False),
    Column("decision", String, nullable=True),
    Column("decided_by", String, nullable=True),
    Column("decided_at", DateTime(timezone=True), nullable=True),
    Column("modified_output", Text, nullable=True),
    Column("reviewer_notes", Text, nullable=True),
)


class ApprovalRequest(BaseModel):
    id: str
    task_id: str
    user_id: str
    created_at: str
    reason: str
    status: RequestStatus = "pending"
    original_task: str
    current_step_id: str | None
    proposed_action: str | None
    escalation_detail: dict
    state_snapshot: dict
    decision: str | None = None
    decided_by: str | None = None
    decided_at: str | None = None
    modified_output: str | None = None
    reviewer_notes: str | None = None

    @classmethod
    def _from_row(cls, row) -> "ApprovalRequest":
        return cls(
            id=row.id,
            task_id=row.task_id,
            user_id=row.user_id,
            created_at=row.created_at.isoformat(),
            reason=row.reason,
            status=row.status,
            original_task=row.original_task,
            current_step_id=row.current_step_id,
            proposed_action=row.proposed_action,
            escalation_detail=row.escalation_detail,
            state_snapshot=row.state_snapshot,
            decision=row.decision,
            decided_by=row.decided_by,
            decided_at=row.decided_at.isoformat() if row.decided_at else None,
            modified_output=row.modified_output,
            reviewer_notes=row.reviewer_notes,
        )


def _serialize_state(state: OrchestraState) -> dict:
    """Just enough of OrchestraState to resume the task later — everything a fresh
    `graph.invoke()` needs to pick up exactly where this one paused."""
    return {
        "task": state["task"],
        "task_id": state["task_id"],
        "user_id": state.get("user_id", "default_user"),
        "plan": state["plan"].model_dump(),
        "step_progress": state.get("step_progress", {}),
        "completed_step_ids": list(state.get("completed_step_ids", [])),
        "pending_results": [r.model_dump() for r in state.get("pending_results", [])],
        "specialist_results": [r.model_dump() for r in state.get("specialist_results", [])],
        "review_history": [v.model_dump() for v in state.get("review_history", [])],
        "escalations": list(state.get("escalations", [])),
    }


class ApprovalQueue:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or db.get_engine()
        metadata.create_all(self.engine, tables=[approval_requests_table], checkfirst=True)

    def push_all(self, state: OrchestraState) -> list[ApprovalRequest]:
        """Push one ApprovalRequest per escalation in `state["escalations"]` that isn't already
        queued. Almost always exactly one — but a single review wave can escalate more than one
        parallel step at once, and a resumed-then-re-escalated task's snapshot already carries
        its earlier escalation(s) — so this dedupes by (step_id, attempt, reason) rather than
        assuming there's only ever one."""
        existing = self._existing_keys(state["task_id"])
        created = []
        for escalation in state.get("escalations", []):
            key = (escalation.get("step_id"), escalation.get("attempt"), escalation["reason"])
            if key in existing:
                continue
            created.append(self._push_one(state, escalation))
            existing.add(key)
        return created

    def _existing_keys(self, task_id: str) -> set[tuple]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(approval_requests_table.c.escalation_detail).where(
                    approval_requests_table.c.task_id == task_id
                )
            ).all()
        return {
            (row.escalation_detail.get("step_id"), row.escalation_detail.get("attempt"), row.escalation_detail["reason"])
            for row in rows
        }

    def _push_one(self, state: OrchestraState, escalation: dict) -> ApprovalRequest:
        plan = state["plan"]
        step_id = escalation.get("step_id")
        current_step = next((s for s in plan.steps if s.step_id == step_id), None)

        request_id = uuid.uuid4().hex
        with self.engine.begin() as conn:
            conn.execute(
                insert(approval_requests_table).values(
                    id=request_id,
                    task_id=state["task_id"],
                    user_id=state.get("user_id", "default_user"),
                    created_at=datetime.now(UTC),
                    reason=escalation["reason"],
                    status="pending",
                    original_task=state["task"],
                    current_step_id=current_step.step_id if current_step else None,
                    proposed_action=self._find_proposed_action(state, escalation),
                    escalation_detail=escalation,
                    state_snapshot=_serialize_state(state),
                )
            )
        return self.get(request_id)

    @staticmethod
    def _find_proposed_action(state: OrchestraState, escalation: dict) -> str | None:
        step_id, attempt = escalation.get("step_id"), escalation.get("attempt")
        for result in reversed(state.get("pending_results", [])):
            if result.step_id == step_id and result.attempt == attempt:
                return result.output
        return None

    def get(self, request_id: str) -> ApprovalRequest | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(approval_requests_table).where(approval_requests_table.c.id == request_id)
            ).first()
        return ApprovalRequest._from_row(row) if row else None

    def list_pending(self, user_id: str | None = None) -> list[ApprovalRequest]:
        stmt = select(approval_requests_table).where(approval_requests_table.c.status == "pending")
        if user_id:
            stmt = stmt.where(approval_requests_table.c.user_id == user_id)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt.order_by(approval_requests_table.c.created_at)).all()
        return [ApprovalRequest._from_row(row) for row in rows]

    def decide(
        self,
        request_id: str,
        decision: Decision,
        decided_by: str,
        reviewer_notes: str | None = None,
        modified_output: str | None = None,
    ) -> ApprovalRequest:
        if decision == "modified" and not modified_output:
            raise ValueError("modified_output is required when decision='modified'.")

        with self.engine.begin() as conn:
            conn.execute(
                update(approval_requests_table)
                .where(approval_requests_table.c.id == request_id)
                .values(
                    status=decision,
                    decision=decision,
                    decided_by=decided_by,
                    decided_at=datetime.now(UTC),
                    reviewer_notes=reviewer_notes,
                    modified_output=modified_output,
                )
            )
        return self.get(request_id)
