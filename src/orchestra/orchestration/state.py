"""Shared graph state threaded through every agent node."""

import operator
from typing import Annotated, Literal, TypedDict

from orchestra.orchestration.schemas import ExecutionPlan, ReviewVerdict, SpecialistResult

Status = Literal["delegating", "reviewing", "retrying", "needs_escalation", "complete"]


class OrchestraState(TypedDict, total=False):
    task: str
    plan: ExecutionPlan | None
    current_step_index: int
    retry_count: int
    pending_result: SpecialistResult | None
    specialist_results: Annotated[list[SpecialistResult], operator.add]
    review_history: Annotated[list[ReviewVerdict], operator.add]
    final_output: str | None
    status: Status
