"""Phase 3.1: the single definition of every condition that sends a task to a human instead of
letting the system decide alone. Every trigger produces an entry in `state["escalations"]`
tagged with one of these reasons and a `status` of `needs_escalation` — the actual halt-and-wait
mechanics are the existing Phase 2.3 escalation path (`orchestration/graph.py`): a task that
escalates routes through `memory_writer` (so what was tried gets remembered) but skips
`delivery` (so its working memory survives in Redis for a human to inspect). Phase 3.2+ is what
turns that halt into an actual paused, resumable wait instead of just ending the run.

Five triggers:

- **low_plan_confidence** — the supervisor's own confidence in its plan, checked once, before
  any specialist runs. Cheaper to catch "I'm not sure this plan will work" before spending a
  wave on it than after.
- **sensitive_operation** — a plan step looks like it involves a financial transaction, data
  deletion, or external communication. Checked by keyword match over the step's own text —
  deliberately not asked of the model, the same reasoning `tools_used` isn't derived from the
  model in Phase 2.2: asking "is this sensitive?" gives an omission a chance to slip through;
  matching the step's own words doesn't.
- **specialist_failed_twice** — a step's output has now been rejected on its 2nd attempt.
- **low_quality_score** — the reviewer's confidence in an (otherwise-approved) deliverable is
  below threshold. An uncertain "yes" still isn't good enough to ship without a human.
- **user_requested** — the caller flagged the task for mandatory human review up front,
  regardless of how confident anything downstream turns out to be.
"""

from typing import Literal

from orchestra.orchestration.schemas import PlanStep

EscalationReason = Literal[
    "low_plan_confidence",
    "sensitive_operation",
    "specialist_failed_twice",
    "low_quality_score",
    "user_requested",
]

PLAN_CONFIDENCE_THRESHOLD = 0.5
QUALITY_SCORE_THRESHOLD = 0.5
SPECIALIST_FAILURE_LIMIT = 2  # escalate once a step has been rejected this many times

# Keyword match, not an exhaustive NLP classifier — false negatives are possible (an oddly
# worded sensitive step could slip through), but false positives are the safer failure mode
# here, and a human reviewing an unnecessary escalation costs a lot less than the reverse.
FINANCIAL_KEYWORDS = (
    "payment",
    "pay ",
    "purchase",
    "buy ",
    "transfer funds",
    "wire transfer",
    "invoice",
    "refund",
    "charge the card",
    "billing",
    "checkout",
    "transaction",
    "bank account",
)
DELETION_KEYWORDS = (
    "delete",
    "erase",
    "permanently remove",
    "drop table",
    "wipe",
    "purge",
)
COMMUNICATION_KEYWORDS = (
    "send an email",
    "send email",
    "email the",
    "notify the",
    "message the",
    "text the",
    "post to",
    "publish to",
    "tweet",
    "dm ",
    "call the customer",
    "contact the",
)


def classify_sensitivity(step: PlanStep) -> str | None:
    """Return a short category if a plan step looks like it needs human sign-off before it
    runs (financial_transaction / data_deletion / external_communication), else None. Keyword
    match over the step's own description and required_inputs."""
    text = " ".join([step.description, *step.required_inputs]).lower()
    if any(keyword in text for keyword in FINANCIAL_KEYWORDS):
        return "financial_transaction"
    if any(keyword in text for keyword in DELETION_KEYWORDS):
        return "data_deletion"
    if any(keyword in text for keyword in COMMUNICATION_KEYWORDS):
        return "external_communication"
    return None
