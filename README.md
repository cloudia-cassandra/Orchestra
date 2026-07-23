# Orchestra

A multi-agent orchestration platform: a supervisor agent decomposes complex tasks, delegates
subtasks to specialized tool-using agents, maintains persistent memory across interactions, and
escalates to a human operator when confidence is low or the task requires approval — with full
observability into every agent decision.

## Tech Stack

| Layer            | Choice                  | Why                                  |
|------------------|--------------------------|---------------------------------------|
| Language         | Python 3.11+             | Ecosystem standard                    |
| Orchestration    | LangGraph                 | State machine for agent workflows     |
| LLM Providers    | Anthropic                 | Multi-model agent routing             |
| Tool Framework   | Custom + MCP              | Extensible tool integration           |
| Memory           | PostgreSQL + ChromaDB     | Short-term + semantic long-term       |
| Queue            | Redis + Celery            | Async task execution                  |
| Review UI        | React or Streamlit        | Human-in-the-loop interface           |
| Containerization | Docker + docker-compose   | Full system orchestration             |

## Phases

1. Agent Architecture
2. Memory System
3. Human-in-the-Loop System
4. Observability and Debugging
5. Integration and End-to-End Testing
6. Polish for Portfolio

Progress notes for each phase are logged in [DEVLOG.md](DEVLOG.md).

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,ui]"
cp .env.example .env  # fill in ANTHROPIC_API_KEY
docker-compose up -d  # postgres, redis, chromadb
```
