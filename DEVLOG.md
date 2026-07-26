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

## 2026-07-25

**Tried this — and this is what happened:** Phase 1.3, the tool registry — this is what turns
"specialists" from prompt-only LLM calls into agents that can actually go do something.

- **A real `ToolRegistry`.** Every tool is registered with a name, description, input/output
  JSON schema, the list of specialist domains allowed to use it, and a rate limit
  (`max_calls` per `per_seconds`, sliding window). `registry.invoke()` is the single choke point
  everything goes through — it checks domain authorization, enforces the rate limit, times the
  call, and logs a structured record (inputs, output or error, latency, success/failure)
  regardless of who's calling or whether it succeeded.
- **Five tools, honestly scoped to what's actually wired up:**
  - `web_search` — pluggable provider seam (`set_search_provider`); raises clearly until a real
    provider is chosen, no fake results pretending to work.
  - `file_read_write` — sandboxed to a `workspace/` directory, blocks path traversal.
  - `code_execution` — runs Python in a subprocess with a timeout; process-isolated but
    explicitly *not* network/resource-sandboxed yet — flagged as a hardening TODO before it
    ever sees untrusted input.
  - `database_query` — read-only, rejects anything that isn't a `SELECT` before it even opens a
    connection.
  - `api_call` — outbound HTTP blocked by default; only allowlisted hosts (via
    `ORCHESTRA_API_ALLOWLIST`) are reachable, so a specialist can't be prompted into an SSRF.
- **Specialists can now actually call tools.** Added an agentic tool-use loop
  (`BaseAgent._call_with_tools`) — the model gets the specialist's registered tools, and if it
  asks for one, the loop invokes it through the registry (so authorization/rate-limiting/logging
  all still apply) and feeds the result back, up to 5 rounds. This was stubbed out back in 1.1 as
  "a later ticket" — this was that ticket.

13 new tests (registry authorization/rate-limiting/logging, each tool's guardrails, and one that
fakes the Anthropic response shape to prove the tool loop really calls the registry and logs it).
39/39 total, still zero API key or live services required.

## 2026-07-26

**Tried this — and this is what happened:** Phase 1.4, the LangGraph state machine — this is
where the graph stopped being "one step at a time" and became a real scheduler.

The old graph tracked a single `current_step_index` and ran the plan strictly in list order.
That's fine for a linear plan, but it meant two independent steps (say, a research lookup and a
data pull that don't depend on each other) would run one after another for no reason, and
there was no notion of confidence-based escalation at all. Rebuilt it properly:

- **Wave-based scheduling.** Every round, the supervisor's routing function
  (`orchestration/waves.ready_steps`) finds every step whose dependencies are already
  satisfied and fans out to all of them at once via LangGraph's `Send` — that's the "parallel"
  half. A step with an unmet dependency just isn't ready yet, so it naturally waits for a later
  wave — that's the "sequential" half, both from the same mechanism instead of two code paths.
  Proved this isn't just a claim with a test that tracks dispatch order: two independent steps
  land in the same wave, and the dependent step only dispatches after both finish.
- **Rejection routes back to the specialist, same as before — just implicitly now.** A step
  that's neither completed nor escalated is automatically "still ready" next wave, so the
  reviewer's feedback (already injected into the specialist's prompt since 1.2) is what drives
  the retry. No separate retry-edge needed.
- **Two escalation triggers, not one.** Previously only "rejected 3 times" escalated. Now the
  reviewer also escalates on low confidence (< 0.5) even if it approved the output — an
  uncertain "yes" still isn't good enough to ship without a human, and a rejection at the last
  allowed attempt escalates the same way. Both dead-end into `needs_escalation`, a real halt
  state (not silently continuing on a broken step) — Phase 3 turns that halt into an actual
  pause-for-human instead of just stopping.
- **Specialist-node crashes get LangGraph's own retry_policy** (3 attempts, backoff) — this is
  separate from reviewer-driven retries: it's for the node *crashing* (API hiccup, rate limit),
  not the node succeeding with output the reviewer doesn't like.
- **Explicit `intake` and `delivery` nodes** bookend the graph now, matching the requested
  pipeline shape (intake → planning → execution → review → synthesis → delivery) and giving
  Phase 4 a clean seam to hook tracing into later.

Batching turned out to matter: reviewing happens once per wave, not once per step, since
LangGraph fans multiple parallel specialist branches back into a single reviewer call. Had to
add `attempt` numbers to both `SpecialistResult` and `ReviewVerdict` so the reviewer can tell
which attempt of which step it's looking at across that batch, instead of guessing from list
order like the old code did.

8 new tests, including one that runs the whole compiled graph end-to-end (fully mocked at the
LLM boundary, no API key) and asserts the actual dispatch order proves parallel execution
happened, plus one proving a step that never gets approved correctly halts the whole run in
`needs_escalation`. 47/47 total passing.

## 2026-07-27

**Tried this — and this is what happened:** Phase 2.1, short-term working memory — the first
piece of the memory system, and the first thing in the project actually backed by Redis
instead of living purely inside one Python process.

Up to now, everything an agent needed (the plan, other steps' outputs, review feedback) only
existed inside the LangGraph state for the duration of one `graph.invoke()` call — reasonable
for running the pipeline, but it means the instant that call returns (or crashes), all of it is
gone. There was nowhere to look at an in-flight or escalated task from outside that one process.

- **`WorkingMemory`** (`src/orchestra/memory/working_memory.py`) is a thin Redis wrapper scoped
  entirely to one `task_id` (`orchestra:wm:{task_id}:*` keys), holding exactly what the ticket
  asked for: the execution plan, every subtask attempt as it happens ("intermediate results" —
  including rejected ones, not just the final approved output), the approved output per
  completed subtask, and an error log.
- **It's a mirror, not a replacement.** The graph's own in-process state is still what drives
  execution (Phase 1.4's wave scheduler depends on that being fast and synchronous) — Working
  Memory is written to alongside it at every meaningful point (plan produced, each specialist
  attempt, each approval, each error/escalation), so it's genuinely externally visible: any
  other process can attach to Redis mid-run and see what's happened so far.
- **Cleared on success, kept on escalation — on purpose.** The ticket says memory is "cleared
  when the task completes," but a task that halts in `needs_escalation` hasn't completed, it's
  waiting on a human — clearing its memory would erase exactly the context that human needs.
  So `delivery` (the success path) calls `clear()`; the escalation path deliberately skips it.
  Every key also carries a TTL as a backstop against a crash leaking keys forever.
- **`scripts/inspect_memory.py`** — a second, independent script that reads a task's memory back
  out of Redis by `task_id` alone. This is the actual proof the store is shared rather than just
  an audit log written and never read: run a task, then inspect it from a completely separate
  process.

9 new tests against a small fake Redis client (no live server needed) covering round-trips for
every field, TTLs actually being set, and that `clear()` only touches its own task's keys, plus
a fixture (`tests/conftest.py`) that swaps in that fake Redis for every existing test so the
whole suite — including the full end-to-end graph runs from 1.4 — still needs zero live
services. 56/56 passing.
