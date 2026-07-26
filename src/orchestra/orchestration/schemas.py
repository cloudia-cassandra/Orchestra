"""Pydantic schemas exchanged between agent nodes."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Domain = Literal["research", "data_analysis", "writing", "code_execution"]
Complexity = Literal["low", "medium", "high"]


class PlanStep(BaseModel):
    step_id: str
    domain: Domain
    description: str = Field(description="What this subtask must accomplish.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="step_ids whose output this step needs before it can run.",
    )
    required_inputs: list[str] = Field(
        default_factory=list,
        description="Specific pieces of information this step needs, e.g. "
        "'s1.output: raw survey data'. Should reference depends_on steps by id "
        "where the input comes from a prior step.",
    )
    expected_output_format: str = Field(
        description="The shape the specialist's output should take, e.g. "
        "'markdown summary with headings' or 'JSON list of {name, value}'."
    )
    estimated_complexity: Complexity


class ExecutionPlan(BaseModel):
    reasoning: str
    steps: list[PlanStep]

    @model_validator(mode="after")
    def _validate_dag(self) -> "ExecutionPlan":
        if not self.steps:
            raise ValueError("Execution plan must contain at least one step.")

        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"Duplicate step_id: {step.step_id!r}")
            seen.add(step.step_id)

        all_ids = {step.step_id for step in self.steps}
        satisfied: set[str] = set()
        for step in self.steps:
            unknown = set(step.depends_on) - all_ids
            if unknown:
                raise ValueError(
                    f"Step {step.step_id!r} depends on unknown step(s): {sorted(unknown)}"
                )
            if step.step_id in step.depends_on:
                raise ValueError(f"Step {step.step_id!r} cannot depend on itself.")
            unmet = set(step.depends_on) - satisfied
            if unmet:
                raise ValueError(
                    f"Step {step.step_id!r} depends on {sorted(unmet)}, which must appear "
                    f"earlier in the ordered plan (dependencies run before dependents)."
                )
            satisfied.add(step.step_id)

        return self


class SpecialistResult(BaseModel):
    step_id: str
    domain: Domain
    attempt: int = Field(ge=1, description="1-indexed attempt number for this step.")
    output: str
    confidence: float = Field(ge=0.0, le=1.0)
    tool_calls: list[str] = Field(default_factory=list)


class ReviewVerdict(BaseModel):
    step_id: str
    attempt: int = Field(ge=1, description="Which attempt of the step this verdict judges.")
    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    feedback: str | None = None
