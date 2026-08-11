"""Tests for Phase 2.2/2.3: long-term semantic memory and its use in planning."""

from orchestra.agents.memory_writer import MemoryWriterAgent
from orchestra.agents.supervisor import SupervisorAgent
from orchestra.memory.long_term_memory import LongTermMemory, MemoryRecord
from orchestra.orchestration.schemas import ExecutionPlan, SpecialistResult

from tests.helpers import make_step


def _make_record(**overrides) -> MemoryRecord:
    defaults = dict(
        task_id="t1",
        task="summarize the quarterly sales report",
        outcome="succeeded",
        approach_summary="research pulled the figures, writing summarized them",
        plan_steps=["research: pull figures", "writing: summarize them"],
        tools_used=["web_search"],
        domain_facts=["Q3 revenue grew 12% year over year"],
        user_preferences=["prefers bullet points over prose"],
    )
    defaults.update(overrides)
    return MemoryRecord(**defaults)


def test_store_and_query_round_trip():
    memory = LongTermMemory()
    memory.store(_make_record())

    results = memory.query("summarize the quarterly sales report")

    assert len(results) == 1
    assert results[0].task == "summarize the quarterly sales report"
    assert results[0].tools_used == ["web_search"]
    assert results[0].domain_facts == ["Q3 revenue grew 12% year over year"]
    assert results[0].user_preferences == ["prefers bullet points over prose"]


def test_query_returns_empty_list_when_store_is_empty():
    memory = LongTermMemory()
    assert memory.query("anything") == []


def test_query_ranks_more_relevant_record_first():
    memory = LongTermMemory()
    memory.store(_make_record(task_id="t1", task="write a poem about the ocean"))
    memory.store(_make_record(task_id="t2", task="summarize the quarterly sales report"))

    results = memory.query("summarize this quarter's sales figures", n_results=1)

    assert results[0].task_id == "t2"


def test_memory_writer_derives_tools_used_from_specialist_results_not_the_model():
    class FakeMemory:
        def __init__(self):
            self.stored = None

        def store(self, record):
            self.stored = record

    fake_memory = FakeMemory()
    agent = MemoryWriterAgent(memory=fake_memory)

    def fake_structured(system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        return {
            "approach_summary": "researched then wrote a summary",
            "domain_facts": ["fact one"],
            "user_preferences": [],
        }

    agent._call_structured = fake_structured

    plan = ExecutionPlan(
        reasoning="test",
        steps=[
            make_step("s1", domain="research", description="find stuff"),
            make_step("s2", domain="writing", depends_on=["s1"], description="write it up"),
        ],
    )
    state = {
        "task": "summarize something",
        "task_id": "t1",
        "status": "complete",
        "final_output": "done",
        "plan": plan,
        "specialist_results": [
            SpecialistResult(
                step_id="s1",
                domain="research",
                attempt=1,
                output="found stuff",
                confidence=0.9,
                tool_calls=["web_search", "web_search"],
            ),
            SpecialistResult(
                step_id="s2",
                domain="writing",
                attempt=1,
                output="wrote stuff",
                confidence=0.9,
                tool_calls=["file_read_write"],
            ),
        ],
    }

    update = agent(state)

    assert update == {}
    assert fake_memory.stored.outcome == "succeeded"
    assert fake_memory.stored.tools_used == ["web_search", "file_read_write"]
    assert fake_memory.stored.domain_facts == ["fact one"]
    assert fake_memory.stored.task == "summarize something"
    assert fake_memory.stored.plan_steps == ["research: find stuff", "writing: write it up"]


def test_memory_writer_records_escalated_outcome_with_escalation_context():
    class FakeMemory:
        def __init__(self):
            self.stored = None

        def store(self, record):
            self.stored = record

    fake_memory = FakeMemory()
    agent = MemoryWriterAgent(memory=fake_memory)
    captured = {}

    def fake_structured(system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        captured["system"] = system
        captured["user"] = user
        return {
            "approach_summary": "tried research-only, reviewer rejected for missing sources",
            "domain_facts": [],
            "user_preferences": [],
        }

    agent._call_structured = fake_structured

    plan = ExecutionPlan(reasoning="test", steps=[make_step("s1", domain="research")])
    state = {
        "task": "find obscure facts",
        "task_id": "t2",
        "status": "needs_escalation",
        "plan": plan,
        "specialist_results": [],
        "escalations": [
            {
                "step_id": "s1",
                "attempt": 3,
                "reason": "max_attempts_exceeded",
                "feedback": "sources were never cited",
                "confidence": 0.4,
            }
        ],
    }

    agent(state)

    assert fake_memory.stored.outcome == "escalated"
    assert fake_memory.stored.plan_steps == ["research: do the thing"]
    assert "sources were never cited" in captured["user"]
    assert "did NOT complete" in captured["system"]


def test_supervisor_plan_prompt_includes_relevant_past_experience(monkeypatch):
    class FakeMemory:
        def query(self, task, n_results=5, user_id=None):
            return [_make_record(task="a similar past task")]

    agent = SupervisorAgent(memory=FakeMemory())
    captured = {}

    def fake_structured(system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        captured["user"] = user
        return {
            "reasoning": "reuse the known approach",
            "steps": [
                {
                    "step_id": "s1",
                    "domain": "research",
                    "description": "find the facts",
                    "depends_on": [],
                    "required_inputs": [],
                    "expected_output_format": "bullets",
                    "estimated_complexity": "low",
                }
            ],
        }

    monkeypatch.setattr(agent, "_call_structured", fake_structured)

    agent({"task": "summarize the quarterly sales report", "task_id": "t2"})

    assert "a similar past task" in captured["user"]
    assert "succeeded" in captured["user"]
    assert "research pulled the figures" in captured["user"]
    assert "research: pull figures" in captured["user"]
    assert "Q3 revenue grew 12% year over year" in captured["user"]


def test_supervisor_plan_prompt_omits_memory_section_when_nothing_relevant(monkeypatch):
    class FakeMemory:
        def query(self, task, n_results=5, user_id=None):
            return []

    agent = SupervisorAgent(memory=FakeMemory())
    captured = {}

    def fake_structured(system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        captured["user"] = user
        return {
            "reasoning": "fresh task",
            "steps": [
                {
                    "step_id": "s1",
                    "domain": "research",
                    "description": "find the facts",
                    "depends_on": [],
                    "required_inputs": [],
                    "expected_output_format": "bullets",
                    "estimated_complexity": "low",
                }
            ],
        }

    monkeypatch.setattr(agent, "_call_structured", fake_structured)

    agent({"task": "a brand new task", "task_id": "t3"})

    assert "Relevant past experience" not in captured["user"]