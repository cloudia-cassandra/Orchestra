"""Shared test factories."""

from orchestra.orchestration.schemas import Domain, PlanStep


def make_step(
    step_id: str,
    domain: Domain = "research",
    description: str = "do the thing",
    depends_on: list[str] | None = None,
    required_inputs: list[str] | None = None,
    expected_output_format: str = "plain text",
    estimated_complexity: str = "low",
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        domain=domain,
        description=description,
        depends_on=depends_on or [],
        required_inputs=required_inputs or [],
        expected_output_format=expected_output_format,
        estimated_complexity=estimated_complexity,
    )
