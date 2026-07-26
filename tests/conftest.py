"""Test-wide fixtures."""

import pytest

from tests.fake_redis import FakeRedis


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Every test gets a fresh in-memory Redis stand-in — nothing here should ever need a
    live Redis server. Patches the singleton getter directly so every WorkingMemory instance
    created during a test (across however many agent calls) shares the same fake store."""
    client = FakeRedis()
    monkeypatch.setattr("orchestra.memory.redis_client.get_redis_client", lambda: client)
    return client
