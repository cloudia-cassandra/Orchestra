"""Pydantic schemas exchanged between agent nodes."""

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["research", "data_analysis", "writing", "code_execution"]


class PlanStep(BaseModel):
    step_id: str
    domain: Domain
    instruction: str
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    reasoning: str
    steps: list[PlanStep]


class SpecialistResult(BaseModel):
    step_id: str
    domain: Domain
    output: str
    confidence: float = Field(ge=0.0, le=1.0)
    tool_calls: list[str] = Field(default_factory=list)


class ReviewVerdict(BaseModel):
    step_id: str
    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    feedback: str | None = None
