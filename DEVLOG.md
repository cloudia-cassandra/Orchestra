# Devlog

Running log of what was tried, what happened, and what shipped — for showing teammates progress.

---

## 2026-07-22

**This is my cool thing:** Scaffolded the Orchestra repo — project structure, `pyproject.toml`
with the full stack (LangGraph, Anthropic SDK, MCP, Postgres/ChromaDB, Redis/Celery), and a
`docker-compose.yml` for Postgres + Redis + ChromaDB. No agent logic yet — this is the shell
Phase 1 gets built into.

## 2026-07-23

**Tried this — and this is what happened:** Built out the three-layer agent hierarchy for Phase
1.1 as LangGraph nodes:

- **Supervisor**: turns the raw task into a JSON execution plan (ordered steps, each tagged
  with a domain), delegates one step at a time, and synthesizes the final answer once every step
  is approved.
- **Specialists** (research, data_analysis, writing, code_execution): one `SpecialistAgent` base
  class with domain-specific system prompts. Tool binding (web search, SQL, sandboxed shell,
  etc.) is stubbed as a `tools` list for now — wiring real tools is a separate ticket.
- **Reviewer**: validates each specialist output against its step's instruction, approves or
  rejects with feedback. Rejections loop back to the *same* specialist with the feedback injected
  into the prompt, up to 2 retries — after that the graph flags `needs_escalation`, which is a
  placeholder hook for Phase 3's human-in-the-loop system.

All state flows through one shared `OrchestraState` (TypedDict), with Pydantic schemas
(`ExecutionPlan`, `SpecialistResult`, `ReviewVerdict`) defining what each node actually reads and
writes. Routing between nodes is two small conditional-edge functions that key off `status` and
the current plan step's domain.

Wrote 8 structural tests (graph compiles with the right nodes, routing logic, retry/escalation
behavior) — all pass without needing an API key, since the reviewer's LLM call is mocked. Haven't
done a live end-to-end run yet (no `ANTHROPIC_API_KEY` in this environment) — that's next, once a
key is in `.env`.

## 2026-07-24

**Tried this — and this is what happened:** Phase 1.2, the task decomposition engine — the part
that actually makes the supervisor's plans trustworthy instead of just plausible-looking JSON.

- **Structured output, not regex.** The old planner asked the model for JSON in prose and
  regex'd the first `{...}` out of the response — fragile if the model added any commentary.
  Swapped it for Anthropic's forced tool-calling (`tool_choice: {"type": "tool", ...}`), so the
  API itself enforces the shape. Added `_call_structured()` to `BaseAgent` for this.
- **Richer plan schema.** Each `PlanStep` now carries `required_inputs` (what data it needs, and
  which prior step it comes from), `expected_output_format` (what shape the output should take),
  and `estimated_complexity` (low/medium/high) — on top of the existing `domain` and
  `depends_on`. Renamed `instruction` → `description` to match.
- **The plan is a validated DAG, not just a list.** `ExecutionPlan` now has a Pydantic
  `model_validator` that rejects duplicate step_ids, dependencies on unknown or nonexistent steps,
  self-dependencies, and — importantly — forward references: if step B depends on step A, A must
  already appear earlier in the list. That ordering guarantee is what lets specialists safely
  assume a dependency's output already exists by the time they run.
- **Dependencies actually do something now.** Specialists read `depends_on`, pull the matching
  prior results out of `specialist_results`, and inject them into the prompt along with
  `required_inputs` and `expected_output_format` — so step B genuinely receives step A's output,
  not just a hint that it exists.
- **Retry loop on invalid plans.** If the model (rarely) still produces something that fails DAG
  validation, the supervisor feeds the validation error back and asks for a corrected plan, up to
  3 attempts, before giving up loudly instead of silently continuing with a broken plan.

9 new tests cover the DAG validator's rejection cases, the structured tool-call path (mocked),
the retry-then-succeed path, and that a specialist's prompt actually contains a dependency's
output. 17/17 passing, still no API key needed.
