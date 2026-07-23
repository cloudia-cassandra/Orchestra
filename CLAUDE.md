# Orchestra

Multi-agent orchestration platform. See [README.md](README.md) for the stack and phase breakdown.

## Working agreements

- **Devlog**: when a phase wraps, or when the user says "log this" / "devlog", append a dated
  entry to [DEVLOG.md](DEVLOG.md) summarizing what was tried and what happened (or "this is my
  cool thing" for a straightforward win). Written for a teammate audience — no internal
  implementation trivia, just what changed and how it went.
- Project layout: `src/orchestra/{agents,orchestration,memory,tools,hitl,observability}`, with
  `ui/` for the review interface and `tests/` mirroring `src/`.
- Local services (Postgres, Redis, ChromaDB) run via `docker-compose up -d`.
