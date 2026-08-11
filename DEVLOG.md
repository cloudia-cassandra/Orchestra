# Devlog

Running log of what was tried, what happened, and what shipped — for showing teammates progress.

---

## 2026-07-22 — Scaffold

Set up the repo: `pyproject.toml` with the full stack (LangGraph, Anthropic SDK, MCP,
Postgres/ChromaDB, Redis/Celery) and `docker-compose.yml` for Postgres + Redis + ChromaDB. No
agent logic yet.

## 2026-07-23 — Phase 1.1: Agent architecture

Three-layer LangGraph pipeline: **Supervisor** (plans, delegates, synthesizes) → **Specialists**
(research, data_analysis, writing, code_execution — tools stubbed for now) → **Reviewer**
(approve/reject, retries up to 2x, then escalates). Shared `OrchestraState` + Pydantic schemas.
8 tests, no API key needed.

## 2026-07-24 — Phase 1.2: Task decomposition

Plans are now forced structured tool-calls (not regex'd JSON) and validated as a DAG
(no duplicate/unknown/self/forward-reference dependencies). Specialists actually receive their
dependencies' outputs now. Invalid plans get fed back to the model for up to 3 retries.
17/17 tests passing.

## 2026-07-25 — Phase 1.3: Tool registry

Real `ToolRegistry` — every call goes through domain authorization, rate limiting, and logging.
Five tools: `web_search`, `file_read_write` (sandboxed), `code_execution` (subprocess, timeout,
not yet resource-sandboxed), `database_query` (read-only), `api_call` (allowlisted hosts only).
Specialists can now actually call tools via an agentic loop. 39/39 tests passing.

## 2026-07-26 — Phase 1.4: Wave-based scheduler

Replaced strict step-by-step execution with wave-based dispatch: every round, all steps whose
dependencies are satisfied fan out in parallel via LangGraph `Send`; dependent steps wait for a
later wave. Escalation now triggers on repeated rejection *or* low reviewer confidence (<0.5).
Explicit `intake`/`delivery` bookend nodes added. 47/47 tests passing.

## 2026-07-27 — Phase 2.1: Short-term working memory

Added `WorkingMemory`, a Redis mirror of in-flight task state (plan, every attempt, approved
outputs, errors), scoped per `task_id` — the first state visible outside one `graph.invoke()`
call. Cleared on success, kept on escalation so a human can inspect it. Tests run against a fake
Redis client, no live server needed. 56/56 passing.

## 2026-07-30 — Phase 2.2: Long-term semantic memory + a new direction

Added `LongTermMemory` (ChromaDB): after a task completes, a `MemoryWriterAgent` node extracts
the approach, domain facts, and user preferences worth remembering and stores them; the
supervisor queries it before planning so past approaches inform new plans. `tools_used` is read
from state, not asked of the model, to avoid hallucinated tool names. Tested against a fake
in-memory Chroma stand-in. 62/62 passing.

Also decided what Orchestra is *for*: rather than staying a generic orchestration demo, it's
becoming a **research assistant for faculty** — triage newly published papers, summarize
methodology, flag contradictions against a professor's current findings, escalate uncertain
results to a human. The orchestration core stays domain-agnostic; this becomes the flagship
application on top of it. No real lab data gets ingested.

## 2026-07-31 — Phase 2.3: Memory retrieval feeds planning

Memory records now capture the *outcome* (`succeeded`/`escalated`) and the plan steps used, not
just the approach text — and `MemoryWriterAgent` now runs on escalated tasks too, not only
successful ones, so failed approaches get remembered as "don't do this again," not just dropped.
The supervisor's planning prompt now shows retrieved memories with their outcome, past plan, and
any facts/preferences, so a new plan can lean on what worked and avoid what didn't. Escalated
tasks route through `memory_writer` but skip `delivery`, so their working memory still survives
in Redis. 65/65 passing.

## 2026-08-10 — Phase 2.4: Memory management

Long-term memory now has upkeep instead of just growing forever. Records carry a `user_id`
(new concept — nothing tracked "who" before this), an `access_count`, and a `last_accessed_at`.
**Importance** blends access frequency with recency decay and re-ranks retrieval so a
frequently-useful memory can beat a barely-more-relevant one nobody's touched again; every
retrieval counts as an access, so importance compounds. **Expiration** deletes memories that are
both old *and* rarely accessed — decay deprioritizes, expiration removes outright.
**Consolidation** (`memory/consolidation.py`) clusters near-duplicate same-outcome memories per
user and merges them into one summary via a structured LLM call — running as maintenance, not a
graph node, which is why `_call_structured` got pulled out of `BaseAgent` into a free
`call_structured()` function both can use. A Streamlit **dashboard** (`ui/memory_dashboard.py`)
shows what's remembered about a user with importance/access stats, plus buttons to consolidate or
delete; `scripts/delete_user_memory.py` is the same deletion path as a CLI — the "delete endpoint"
for a data request, honestly scoped to a project with no HTTP server yet. 79/79 passing.

## 2026-08-10 — Phase 3.1: Escalation triggers

Started Phase 3 (Human-in-the-Loop) by defining, in one place (`hitl/triggers.py`), every
condition that sends a task to a human: **low plan confidence** — the supervisor now self-reports
a 0–1 confidence on the whole plan, checked once before any specialist runs; **sensitive
operation** — a plan step's own text gets keyword-matched for financial transactions, data
deletion, or external communication, deliberately not asked of the model (same reasoning as
`tools_used` in Phase 2.2 — an omission could slip past a self-report, not past a keyword match
on what the step already says); **specialist failed twice** — tightened from "3rd attempt" to
"2nd," matching the ticket's wording; **low quality score** — the reviewer's existing low-
confidence trigger, renamed to match; **user requested** — a new `user_requested_review` flag on
the incoming task forces escalation regardless of how confident anything downstream is. All five
route through the existing Phase 2.3 escalation path (`memory_writer` then skip `delivery`) — the
plan-level ones now short-circuit *before* a single specialist dispatches, not just after one
fails. 90/90 passing.

## 2026-08-10 — Phase 3.2: The approval queue

The part that makes an escalation actually pause-and-wait instead of just halting: a new
`approval_queue` graph node packages the full context — task, plan, completed steps, the step
awaiting a decision, the agent's proposed action — into an `ApprovalRequest` and pushes it to
Postgres (`hitl/approval_queue.py`), the first real use of Postgres in this project; working
memory lives in Redis and long-term memory in Chroma, deliberately, because neither needed to be
queried and filtered relationally the way a review queue does.

Resuming didn't need LangGraph's checkpointer machinery. The graph (Phase 1.4) was already built
so everything needed to keep going lives in `OrchestraState` itself, not call-stack position —
`hitl/resume.py` reconstructs that state from the request's snapshot, folds in the human's
decision the same way an automated approval/rejection already would have, and just calls
`graph.invoke()` again. `intake_node` had to learn not to reset status back to "planning" when a
plan is already present (a one-line fix), and a new `rejected` terminal status lets a plan-level
rejection end the task cleanly through the same routing instead of a special-cased return.

Three decisions, two shapes: step-level escalations (specialist failed twice / low quality
score) support **approve** (proposed output accepted as-is), **reject** (sent back for another
attempt seeded with the reviewer's own feedback — the same channel the automated reviewer's
rejections already use), and **modify** (the reviewer's edited output accepted instead). Plan-
level escalations (low confidence / sensitive operation / user requested) only support approve
(run the plan as proposed) or reject (task ends outright — there's nothing partial to retry, and
re-planning from scratch isn't a feature this has); modify is rejected with a clear error rather
than doing something plausible-looking with a whole edited plan.

Two review surfaces, same underlying `decide()` + `resume_task()`: `scripts/review_queue.py`
(list/show/approve/reject/modify from the terminal) and `ui/approval_dashboard.py` (a Streamlit
page — the Phase 3 "Review UI" the stack table always named). Tested against a real SQLite
in-memory engine standing in for Postgres (SQLAlchemy Core runs unmodified against either), so
the whole suite — including a full escalate-then-resume-to-completion graph run — still needs no
live services. 104/104 passing.