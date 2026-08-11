"""Tests for Phase 2.4: memory management — importance scoring, expiration, consolidation,
per-user listing, and deletion."""

from datetime import UTC, datetime, timedelta

from orchestra.memory.consolidation import MemoryConsolidator
from orchestra.memory.long_term_memory import LongTermMemory, MemoryRecord


def _make_record(**overrides) -> MemoryRecord:
    defaults = dict(
        task_id="t1",
        user_id="alice",
        task="summarize the quarterly sales report",
        outcome="succeeded",
        approach_summary="research pulled the figures, writing summarized them",
        plan_steps=["research: pull figures", "writing: summarize them"],
        domain_facts=["Q3 revenue grew 12% year over year"],
        user_preferences=["prefers bullet points over prose"],
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


# ---------- importance scoring ----------


def test_query_bumps_access_count_on_returned_records():
    memory = LongTermMemory()
    memory.store(_make_record())

    first = memory.query("summarize the quarterly sales report")[0]
    assert first.access_count == 1

    second = memory.query("summarize the quarterly sales report")[0]
    assert second.access_count == 2


def test_frequently_accessed_memory_outranks_a_similarly_relevant_unaccessed_one():
    memory = LongTermMemory()
    # Both records share the exact same task text, so semantic relevance ties — importance
    # (driven by access frequency) has to be what breaks the tie.
    memory.store(_make_record(task_id="rare", task="draft a project update"))
    memory.store(_make_record(task_id="frequent", task="draft a project update"))

    # Access the "frequent" one many times via direct queries scoped tightly enough that only
    # it comes back, so its access_count climbs independent of "rare".
    for _ in range(5):
        [r for r in memory.query("draft a project update", n_results=2) if r.task_id == "frequent"]
        # Manually bump only the frequent record to isolate the effect being tested.
        frequent = memory.list_for_user("alice")
        target = next(r for r in frequent if r.task_id == "frequent")
        target.access_count += 1
        memory.collection.update(ids=[target.id], metadatas=[target._metadata()])

    results = memory.query("draft a project update", n_results=1)
    assert results[0].task_id == "frequent"


def test_compute_importance_decays_with_age():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fresh = _make_record(last_accessed_at=now.isoformat())
    stale = _make_record(last_accessed_at=(now - timedelta(days=90)).isoformat())

    assert LongTermMemory.compute_importance(fresh, now) > LongTermMemory.compute_importance(
        stale, now
    )


def test_compute_importance_increases_with_access_count():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    rarely_used = _make_record(access_count=0, last_accessed_at=now.isoformat())
    often_used = _make_record(access_count=10, last_accessed_at=now.isoformat())

    assert LongTermMemory.compute_importance(often_used, now) > LongTermMemory.compute_importance(
        rarely_used, now
    )


# ---------- expiration ----------


def test_expire_stale_deletes_old_rarely_accessed_records():
    memory = LongTermMemory()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_and_unused = _make_record(
        task_id="old", access_count=0, completed_at=(now - timedelta(days=200)).isoformat()
    )
    memory.store(old_and_unused)

    deleted = memory.expire_stale(max_age_days=180, min_access_count=3, now=now)

    assert deleted == 1
    assert memory.list_for_user("alice") == []


def test_expire_stale_spares_frequently_accessed_records_despite_age():
    memory = LongTermMemory()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_but_popular = _make_record(
        task_id="popular",
        access_count=5,
        completed_at=(now - timedelta(days=200)).isoformat(),
        last_accessed_at=(now - timedelta(days=200)).isoformat(),
    )
    memory.store(old_but_popular)

    deleted = memory.expire_stale(max_age_days=180, min_access_count=3, now=now)

    assert deleted == 0
    assert len(memory.list_for_user("alice")) == 1


def test_expire_stale_spares_recent_records_regardless_of_access_count():
    memory = LongTermMemory()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    recent = _make_record(task_id="recent", access_count=0, completed_at=now.isoformat())
    memory.store(recent)

    deleted = memory.expire_stale(max_age_days=180, min_access_count=3, now=now)

    assert deleted == 0


# ---------- per-user listing and deletion ----------


def test_list_for_user_only_returns_that_users_records():
    memory = LongTermMemory()
    memory.store(_make_record(task_id="a1", user_id="alice"))
    memory.store(_make_record(task_id="b1", user_id="bob"))

    alice_records = memory.list_for_user("alice")

    assert [r.task_id for r in alice_records] == ["a1"]


def test_list_for_user_sorts_by_importance_descending():
    memory = LongTermMemory()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    memory.store(
        _make_record(task_id="low", access_count=0, last_accessed_at=(now - timedelta(days=100)).isoformat())
    )
    memory.store(_make_record(task_id="high", access_count=8, last_accessed_at=now.isoformat()))

    # list_for_user computes importance against "now" internally; freeze via monkeypatching
    # isn't needed here since both records are stored with explicit timestamps far enough apart
    # that real-clock "now" still orders them the same way.
    records = memory.list_for_user("alice")
    assert [r.task_id for r in records] == ["high", "low"]


def test_delete_for_user_removes_only_that_users_records():
    memory = LongTermMemory()
    memory.store(_make_record(task_id="a1", user_id="alice"))
    memory.store(_make_record(task_id="a2", user_id="alice"))
    memory.store(_make_record(task_id="b1", user_id="bob"))

    deleted = memory.delete_for_user("alice")

    assert deleted == 2
    assert memory.list_for_user("alice") == []
    assert len(memory.list_for_user("bob")) == 1


def test_delete_for_user_with_no_records_returns_zero():
    memory = LongTermMemory()
    assert memory.delete_for_user("nobody") == 0


# ---------- consolidation ----------


def test_consolidate_merges_similar_same_outcome_records(monkeypatch):
    memory = LongTermMemory()
    memory.store(
        _make_record(
            task_id="t1",
            task="summarize the Q3 sales report",
            domain_facts=["Q3 revenue grew 12%"],
        )
    )
    memory.store(
        _make_record(
            task_id="t2",
            task="summarize the Q3 sales report",
            domain_facts=["Q3 revenue grew 12%"],
        )
    )

    def fake_call_structured(model, system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        return {
            "task": "summarize a quarterly sales report",
            "approach_summary": "research pulls the figures, writing summarizes them",
            "domain_facts": ["Q3 revenue grew 12%"],
            "user_preferences": ["prefers bullet points over prose"],
        }

    monkeypatch.setattr("orchestra.memory.consolidation.call_structured", fake_call_structured)

    consolidator = MemoryConsolidator(memory=memory)
    merged = consolidator.consolidate("alice", similarity_threshold=0.9, min_cluster_size=2)

    assert len(merged) == 1
    assert merged[0].consolidated is True
    assert merged[0].source_count == 2

    remaining = memory.list_for_user("alice")
    assert len(remaining) == 1
    assert remaining[0].consolidated is True


def test_consolidate_does_not_merge_records_with_different_outcomes(monkeypatch):
    memory = LongTermMemory()
    memory.store(_make_record(task_id="t1", task="find obscure facts", outcome="succeeded"))
    memory.store(_make_record(task_id="t2", task="find obscure facts", outcome="escalated"))

    def fake_call_structured(*args, **kwargs):
        raise AssertionError("should not be called — nothing should cluster")

    monkeypatch.setattr("orchestra.memory.consolidation.call_structured", fake_call_structured)

    consolidator = MemoryConsolidator(memory=memory)
    merged = consolidator.consolidate("alice", similarity_threshold=0.9, min_cluster_size=2)

    assert merged == []
    assert len(memory.list_for_user("alice")) == 2


def test_consolidate_leaves_dissimilar_records_alone(monkeypatch):
    memory = LongTermMemory()
    memory.store(
        _make_record(
            task_id="t1",
            task="summarize the Q3 sales report",
            approach_summary="research pulled figures, writing summarized them",
            domain_facts=["Q3 revenue grew 12%"],
        )
    )
    memory.store(
        _make_record(
            task_id="t2",
            task="draft a completely unrelated poem about rain",
            approach_summary="writing free-associated some imagery",
            domain_facts=[],
        )
    )

    def fake_call_structured(*args, **kwargs):
        raise AssertionError("should not be called — nothing should cluster")

    monkeypatch.setattr("orchestra.memory.consolidation.call_structured", fake_call_structured)

    consolidator = MemoryConsolidator(memory=memory)
    merged = consolidator.consolidate("alice", similarity_threshold=0.1, min_cluster_size=2)

    assert merged == []
    assert len(memory.list_for_user("alice")) == 2
