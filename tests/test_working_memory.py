"""Tests for Phase 2.1's Redis-backed working memory — uses a fake client, no live Redis."""

from orchestra.memory.working_memory import WorkingMemory
from orchestra.orchestration.schemas import ExecutionPlan, SpecialistResult

from tests.fake_redis import FakeRedis
from tests.helpers import make_step


def _memory(task_id="t1", ttl_seconds=3600) -> WorkingMemory:
    return WorkingMemory(task_id, client=FakeRedis(), ttl_seconds=ttl_seconds)


def test_set_and_get_task():
    memory = _memory()
    memory.set_task("do the thing")
    assert memory.get_task() == "do the thing"


def test_get_task_before_set_returns_none():
    assert _memory().get_task() is None


def test_set_and_get_plan_round_trips():
    memory = _memory()
    plan = ExecutionPlan(reasoning="r", steps=[make_step("s1")])
    memory.set_plan(plan)

    fetched = memory.get_plan()
    assert fetched.reasoning == "r"
    assert fetched.steps[0].step_id == "s1"


def test_get_plan_before_set_returns_none():
    assert _memory().get_plan() is None


def test_completed_outputs_round_trip():
    memory = _memory()
    result = SpecialistResult(step_id="s1", domain="research", attempt=1, output="done", confidence=0.9)
    memory.add_completed_output(result)

    outputs = memory.get_completed_outputs()
    assert outputs["s1"].output == "done"
    assert outputs["s1"].attempt == 1


def test_intermediate_results_accumulate_across_attempts():
    memory = _memory()
    attempt_1 = SpecialistResult(step_id="s1", domain="research", attempt=1, output="try 1", confidence=0.5)
    attempt_2 = SpecialistResult(step_id="s1", domain="research", attempt=2, output="try 2", confidence=0.9)
    memory.add_intermediate_result(attempt_1)
    memory.add_intermediate_result(attempt_2)

    results = memory.get_intermediate_results("s1")
    assert [r.output for r in results] == ["try 1", "try 2"]


def test_error_log_round_trips():
    memory = _memory()
    memory.append_error_log("s1", "tool timed out", attempt=2)

    log = memory.get_error_log()
    assert len(log) == 1
    assert log[0]["step_id"] == "s1"
    assert log[0]["message"] == "tool timed out"
    assert log[0]["attempt"] == 2
    assert "timestamp" in log[0]


def test_clear_removes_everything_for_the_task_but_not_other_tasks():
    client = FakeRedis()
    memory_a = WorkingMemory("task-a", client=client)
    memory_b = WorkingMemory("task-b", client=client)

    memory_a.set_task("task a")
    memory_a.append_error_log(None, "oops")
    memory_b.set_task("task b")

    memory_a.clear()

    assert memory_a.get_task() is None
    assert memory_a.get_error_log() == []
    assert memory_b.get_task() == "task b"


def test_every_write_sets_a_ttl():
    client = FakeRedis()
    memory = WorkingMemory("t1", client=client, ttl_seconds=60)
    memory.set_task("x")
    key = memory._key("task")
    assert client.expiries[key] == 60
