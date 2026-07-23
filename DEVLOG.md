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
