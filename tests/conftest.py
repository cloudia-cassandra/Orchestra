"""Test-wide fixtures."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from tests.fake_chroma import FakeChromaCollection
from tests.fake_redis import FakeRedis


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Every test gets a fresh in-memory Redis stand-in — nothing here should ever need a
    live Redis server. Patches the singleton getter directly so every WorkingMemory instance
    created during a test (across however many agent calls) shares the same fake store."""
    client = FakeRedis()
    monkeypatch.setattr("orchestra.memory.redis_client.get_redis_client", lambda: client)
    return client


@pytest.fixture(autouse=True)
def fake_chroma_collection(monkeypatch):
    """Every test gets a fresh in-memory Chroma collection stand-in — nothing here should ever
    need a live Chroma server or a real embedding model. Patches the singleton getter directly
    so every LongTermMemory instance created during a test shares the same fake store."""
    collection = FakeChromaCollection()
    monkeypatch.setattr("orchestra.memory.chroma_client.get_memory_collection", lambda: collection)
    return collection


@pytest.fixture(autouse=True)
def fake_postgres_engine(monkeypatch):
    """Every test gets a fresh in-memory SQLite engine standing in for Postgres — nothing here
    should ever need a live Postgres server. Real SQL (not a hand-rolled fake), just a portable
    backend: `ApprovalQueue`'s SQLAlchemy Core statements run unmodified against it.
    `StaticPool` keeps every connection on this engine pointed at the same in-memory database
    for the fixture's lifetime — the SQLite default would otherwise hand out a fresh, empty
    database per connection."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    monkeypatch.setattr("orchestra.hitl.db.get_engine", lambda: engine)
    return engine
