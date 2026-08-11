"""Memory Writer: after a task finishes — successfully or by escalating to a human — extracts
the reusable parts of what just happened and stores them in long-term semantic memory.

Runs on *both* terminal paths (Phase 2.3): a successful task teaches a future planner what
worked, but an escalated task is just as valuable to remember — it teaches a future planner what
*didn't* work, so it isn't retried blindly. Only the outcome and how the extraction prompt frames
the question differ; the storage path is the same either way.

`tools_used` and `plan_steps` are *not* asked of the model — they're read straight out of state
(`specialist_results` and `plan`), since both are already known exactly and re-deriving them via
the LLM would just add a chance to hallucinate something that never happened.
"""

from orchestra.agents.base import BaseAgent
from orchestra.memory.long_term_memory import LongTermMemory, MemoryRecord
from orchestra.orchestration.state import OrchestraState

_EXTRACTION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "approach_summary": {"type": "string"},
        "domain_facts": {"type": "array", "items": {"type": "string"}},
        "user_preferences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approach_summary", "domain_facts", "user_preferences"],
}

_SUCCESS_PROMPT = """You are the Memory Writer in a multi-agent system. A task just completed \
successfully. Read the original task and the specialist results below, then extract what's worth \
remembering for future, similar tasks, and submit it via the record_memory tool.

- approach_summary: what approach worked, concrete enough that a future planner could reuse it \
(which domains were involved, in what order, and why).
- domain_facts: specific factual claims discovered while doing the task (not generic knowledge). \
Empty list if none were discovered.
- user_preferences: preferences the user expressed or implied about how they want work done \
(format, tone, scope, tools to prefer or avoid). Empty list if none were observed."""

_ESCALATION_PROMPT = """You are the Memory Writer in a multi-agent system. A task did NOT \
complete — it was escalated to a human after a specialist's output was repeatedly rejected or \
the reviewer's confidence stayed too low. Read the original task, the specialist results, and \
the escalation reasons below, then extract what's worth remembering so a future planner doesn't \
repeat the same mistake, and submit it via the record_memory tool.

- approach_summary: what approach was tried and specifically why it didn't work (what the \
reviewer objected to, or why confidence stayed low) — concrete enough that a future planner \
knows what to do differently.
- domain_facts: specific factual claims discovered along the way, even though the task stalled. \
Empty list if none were discovered.
- user_preferences: preferences the user expressed or implied about how they want work done. \
Empty list if none were observed."""


class MemoryWriterAgent(BaseAgent):
    name = "memory_writer"

    def __init__(self, model: str | None = None, memory: LongTermMemory | None = None):
        super().__init__(model)
        self.memory = memory or LongTermMemory()

    def __call__(self, state: OrchestraState) -> dict:
        escalated = state.get("status") == "needs_escalation"
        results = state.get("specialist_results", [])
        results_text = "\n\n".join(f"[{r.domain} / {r.step_id}] {r.output}" for r in results)
        tools_used = list(dict.fromkeys(tool for r in results for tool in r.tool_calls))
        plan_steps = [f"{s.domain}: {s.description}" for s in state["plan"].steps]

        user = f"Original task: {state['task']}\n\nSpecialist results:\n{results_text}\n\n"
        if escalated:
            escalation_text = "\n".join(
                f"- step {e['step_id']} (attempt {e['attempt']}): {e['reason']}"
                f" — {e.get('feedback') or 'no feedback given'}"
                for e in state.get("escalations", [])
            )
            user += f"Escalation reasons:\n{escalation_text}"
        else:
            user += f"Final output: {state.get('final_output')}"

        extracted = self._call_structured(
            system=_ESCALATION_PROMPT if escalated else _SUCCESS_PROMPT,
            user=user,
            tool_name="record_memory",
            tool_description="Record what's worth remembering from this task.",
            input_schema=_EXTRACTION_TOOL_SCHEMA,
        )

        record = MemoryRecord(
            task_id=state["task_id"],
            user_id=state.get("user_id", "default_user"),
            task=state["task"],
            outcome="escalated" if escalated else "succeeded",
            approach_summary=extracted["approach_summary"],
            plan_steps=plan_steps,
            tools_used=tools_used,
            domain_facts=extracted["domain_facts"],
            user_preferences=extracted["user_preferences"],
        )
        self.memory.store(record)
        return {}