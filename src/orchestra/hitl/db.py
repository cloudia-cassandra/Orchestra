"""Lazy singleton SQLAlchemy engine for Postgres-backed human-in-the-loop tables.

This is the first real use of Postgres in the project — working memory lives in Redis (Phase
2.1) and long-term memory in ChromaDB (Phase 2.2), both deliberately, since neither needs to be
queried and filtered relationally. The approval queue does: "show me every pending request for
this user," "has this exact escalation already been queued," "update this one row's decision" —
exactly what a relational table is for, and exactly what Postgres has been sitting in
docker-compose for since the project was scaffolded without yet being used.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ.get("DATABASE_URL") or _url_from_env(), future=True)
    return _engine


def _url_from_env() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "orchestra")
    user = os.environ.get("POSTGRES_USER", "orchestra")
    password = os.environ.get("POSTGRES_PASSWORD", "orchestra")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
