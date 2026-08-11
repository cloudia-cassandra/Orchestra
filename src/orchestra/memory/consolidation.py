"""Memory consolidation (Phase 2.4): merge near-duplicate long-term memories into one
higher-level summary instead of letting the store fill up with many records that all say
roughly the same thing.

Runs as maintenance, not as part of any task's graph — there's no single task it belongs to, so
it isn't a `BaseAgent`/LangGraph node. It reuses `agents.base.call_structured` directly instead.

Clustering is deliberately simple: for each not-yet-grouped record, query the collection for its
own nearest neighbors (same user, same outcome — merging a working approach with a failed one
would produce a summary that's true of neither) and greedily absorb everything within
`similarity_threshold`. This isn't optimal clustering, but it doesn't need to be for memory
management housekeeping — good enough is the point, and it's cheap to rerun.
"""

from orchestra.agents.base import call_structured
from orchestra.memory.long_term_memory import LongTermMemory, MemoryRecord

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # max distance (lower = more similar) to count as "the same"
DEFAULT_MIN_CLUSTER_SIZE = 2
NEIGHBOR_FANOUT = 10  # how many nearest neighbors to consider per record when clustering

_CONSOLIDATION_SYSTEM_PROMPT = """You are consolidating several long-term memory records that \
all describe similar tasks with the same outcome. Merge them into one higher-level summary and \
submit it via the record_consolidated_memory tool.

- task: a short description of the general kind of task these records share (not any one \
original task verbatim).
- approach_summary: the shared approach, generalized enough to apply to the whole group — call \
out any variation between records that's worth keeping in mind.
- domain_facts: the union of facts worth remembering, with true duplicates collapsed.
- user_preferences: the union of preferences worth remembering, with true duplicates collapsed."""

_CONSOLIDATION_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
        "approach_summary": {"type": "string"},
        "domain_facts": {"type": "array", "items": {"type": "string"}},
        "user_preferences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["task", "approach_summary", "domain_facts", "user_preferences"],
}


class MemoryConsolidator:
    def __init__(self, memory: LongTermMemory | None = None, model: str = DEFAULT_MODEL):
        self.memory = memory or LongTermMemory()
        self.model = model

    def consolidate(
        self,
        user_id: str,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    ) -> list[MemoryRecord]:
        """Find and merge clusters of near-duplicate memories for one user. Returns the new
        consolidated records that were created (empty if nothing was similar enough to merge)."""
        records = [r for r in self.memory.list_for_user(user_id) if not r.consolidated]
        by_id = {r.id: r for r in records}
        clustered: set[str] = set()
        consolidated_records: list[MemoryRecord] = []

        for record in records:
            if record.id in clustered:
                continue
            cluster = self._find_cluster(record, by_id, clustered, similarity_threshold)
            if len(cluster) < min_cluster_size:
                continue

            clustered.update(r.id for r in cluster)
            merged = self._merge(user_id, cluster)
            self.memory.store(merged)
            self.memory.collection.delete(ids=[r.id for r in cluster])
            consolidated_records.append(merged)

        return consolidated_records

    def _find_cluster(
        self,
        record: MemoryRecord,
        by_id: dict[str, MemoryRecord],
        already_clustered: set[str],
        similarity_threshold: float,
    ) -> list[MemoryRecord]:
        results = self.memory.collection.query(
            query_texts=[record._document_text()],
            n_results=min(NEIGHBOR_FANOUT, len(by_id)),
            where={"user_id": record.user_id},
        )
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0] or [0.0] * len(ids)

        cluster = [record]
        for neighbor_id, distance in zip(ids, distances):
            if neighbor_id == record.id or neighbor_id in already_clustered:
                continue
            neighbor = by_id.get(neighbor_id)
            if neighbor is None or neighbor.outcome != record.outcome:
                continue
            if distance <= similarity_threshold:
                cluster.append(neighbor)
        return cluster

    def _merge(self, user_id: str, cluster: list[MemoryRecord]) -> MemoryRecord:
        summaries = "\n\n".join(
            f"- task: {r.task}\n  approach: {r.approach_summary}\n"
            f"  facts: {'; '.join(r.domain_facts) or '(none)'}\n"
            f"  preferences: {'; '.join(r.user_preferences) or '(none)'}"
            for r in cluster
        )
        extracted = call_structured(
            model=self.model,
            system=_CONSOLIDATION_SYSTEM_PROMPT,
            user=f"Outcome shared by all records: {cluster[0].outcome}\n\nRecords:\n{summaries}",
            tool_name="record_consolidated_memory",
            tool_description="Submit the consolidated summary for this group of memories.",
            input_schema=_CONSOLIDATION_TOOL_SCHEMA,
        )

        return MemoryRecord(
            task_id="+".join(sorted({r.task_id for r in cluster})),
            user_id=user_id,
            task=extracted["task"],
            outcome=cluster[0].outcome,
            approach_summary=extracted["approach_summary"],
            plan_steps=list(dict.fromkeys(step for r in cluster for step in r.plan_steps)),
            tools_used=list(dict.fromkeys(tool for r in cluster for tool in r.tools_used)),
            domain_facts=extracted["domain_facts"],
            user_preferences=extracted["user_preferences"],
            access_count=sum(r.access_count for r in cluster),
            consolidated=True,
            source_count=sum(r.source_count for r in cluster),
        )
