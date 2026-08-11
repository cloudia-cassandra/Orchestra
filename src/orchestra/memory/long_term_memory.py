"""Long-term semantic memory: after a task finishes, the reusable parts of what happened (what
was asked, what approach worked or didn't, which tools were used, any domain facts discovered,
any user preferences observed) get embedded and stored in ChromaDB. Future tasks query this by
the new task's text *before* planning, so the supervisor can lean on what's already worked (or
avoid what hasn't) instead of starting from a blank page every time.

Phase 2.4 adds memory *management* on top of that store:
- **Importance scoring** — a blend of access frequency and recency, recomputed on every read.
  Memories that keep getting retrieved stay important; memories nobody's touched decay.
- **Consolidation** (`memory/consolidation.py`) — near-duplicate memories for the same user and
  outcome get merged into one higher-level summary instead of piling up as noise.
- **Expiration** — memories that are both old *and* rarely accessed get deleted outright, as
  opposed to importance decay, which just deprioritizes them in ranking.
- **Per-user scoping** — every record carries a `user_id`, so `list_for_user`/`delete_for_user`
  can answer "what does the system remember about this person" and honor a deletion request.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from orchestra.memory import chroma_client

Outcome = Literal["succeeded", "escalated"]

# --- importance scoring -----------------------------------------------------------------------
# importance = ACCESS_WEIGHT * frequency_score + RECENCY_WEIGHT * recency_score, both in [0, 1].
# frequency_score saturates at FREQUENCY_SATURATION accesses (more accesses past that point
# don't make a memory *more* important, just confirms it already is). recency_score halves
# every RECENCY_HALF_LIFE_DAYS days since the memory was last accessed (or stored, if never
# accessed) — this is the "stale memories decay" half of Phase 2.4.
IMPORTANCE_ACCESS_WEIGHT = 0.6
IMPORTANCE_RECENCY_WEIGHT = 0.4
FREQUENCY_SATURATION = 10
RECENCY_HALF_LIFE_DAYS = 30.0

# --- retrieval ranking -------------------------------------------------------------------------
# query() blends semantic relevance with importance so a frequently-useful memory can outrank a
# barely-more-relevant one nobody's ever needed again — not so much that importance overrides
# relevance entirely (a highly important but off-topic memory shouldn't come back for anything).
RELEVANCE_WEIGHT = 0.7
IMPORTANCE_WEIGHT_IN_RANKING = 0.3
CANDIDATE_MULTIPLIER = 3  # over-fetch this many candidates before re-ranking by importance

# --- expiration ----------------------------------------------------------------------------
# A memory is expired outright (not just deprioritized) only once it's both old AND rarely
# used — a memory accessed often stays no matter its age, since frequent access is itself
# evidence it's still useful.
EXPIRATION_MAX_AGE_DAYS = 180
EXPIRATION_MIN_ACCESS_COUNT = 3


class MemoryRecord(BaseModel):
    id: str | None = Field(default=None, description="Chroma document id; unset until stored.")
    task_id: str
    user_id: str = "default_user"
    task: str
    outcome: Outcome
    approach_summary: str = Field(
        description="What approach was taken, and whether/why it worked or didn't."
    )
    plan_steps: list[str] = Field(
        default_factory=list, description="The execution plan's steps, as 'domain: description'."
    )
    tools_used: list[str] = Field(default_factory=list)
    domain_facts: list[str] = Field(
        default_factory=list, description="Domain-specific facts discovered while doing the task."
    )
    user_preferences: list[str] = Field(
        default_factory=list, description="Preferences the user expressed or implied."
    )
    completed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    access_count: int = 0
    last_accessed_at: str | None = None
    consolidated: bool = Field(
        default=False, description="True if this record is a merged summary of several originals."
    )
    source_count: int = Field(
        default=1, description="How many original records this one represents (1 if not merged)."
    )

    def _document_text(self) -> str:
        # What actually gets embedded — task + approach + facts, since that's what a future
        # task's own text would plausibly match against. Everything else rides along as
        # metadata; it's a structured lookup, not a semantic match.
        parts = [self.task, self.approach_summary, *self.domain_facts]
        return "\n".join(p for p in parts if p)

    def _metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "task": self.task,
            "outcome": self.outcome,
            "approach_summary": self.approach_summary,
            "plan_steps": json.dumps(self.plan_steps),
            "tools_used": json.dumps(self.tools_used),
            "domain_facts": json.dumps(self.domain_facts),
            "user_preferences": json.dumps(self.user_preferences),
            "completed_at": self.completed_at,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at or "",
            "consolidated": self.consolidated,
            "source_count": self.source_count,
        }

    @classmethod
    def _from_metadata(cls, metadata: dict, id: str | None = None) -> "MemoryRecord":
        return cls(
            id=id,
            task_id=metadata["task_id"],
            user_id=metadata.get("user_id", "default_user"),
            task=metadata["task"],
            outcome=metadata["outcome"],
            approach_summary=metadata["approach_summary"],
            plan_steps=json.loads(metadata["plan_steps"]),
            tools_used=json.loads(metadata["tools_used"]),
            domain_facts=json.loads(metadata["domain_facts"]),
            user_preferences=json.loads(metadata["user_preferences"]),
            completed_at=metadata["completed_at"],
            access_count=metadata.get("access_count", 0),
            last_accessed_at=metadata.get("last_accessed_at") or None,
            consolidated=metadata.get("consolidated", False),
            source_count=metadata.get("source_count", 1),
        )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


class LongTermMemory:
    def __init__(self, collection=None):
        self.collection = collection or chroma_client.get_memory_collection()

    # ---------- write ----------

    def store(self, record: MemoryRecord) -> str:
        record.id = record.id or uuid.uuid4().hex
        self.collection.add(
            ids=[record.id], documents=[record._document_text()], metadatas=[record._metadata()]
        )
        return record.id

    # ---------- importance ----------

    @staticmethod
    def age_days(record: MemoryRecord, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        reference = _parse_timestamp(record.last_accessed_at or record.completed_at)
        return max(0.0, (now - reference).total_seconds() / 86400)

    @staticmethod
    def compute_importance(record: MemoryRecord, now: datetime | None = None) -> float:
        """Blend of access frequency and recency, both normalized to [0, 1]. Frequently
        accessed memories score higher (up to a saturation point); memories that haven't been
        touched in a while decay toward zero as they go stale."""
        frequency_score = min(1.0, record.access_count / FREQUENCY_SATURATION)
        age = LongTermMemory.age_days(record, now)
        recency_score = 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)
        return IMPORTANCE_ACCESS_WEIGHT * frequency_score + IMPORTANCE_RECENCY_WEIGHT * recency_score

    def _bump_access(self, records: list[MemoryRecord], now: datetime) -> None:
        ids, metadatas = [], []
        for record in records:
            record.access_count += 1
            record.last_accessed_at = now.isoformat()
            ids.append(record.id)
            metadatas.append(record._metadata())
        if ids:
            self.collection.update(ids=ids, metadatas=metadatas)

    # ---------- read ----------

    def query(self, task: str, n_results: int = 5, user_id: str | None = None) -> list[MemoryRecord]:
        """Retrieve memories relevant to `task`, ranked by a blend of semantic relevance and
        importance (Phase 2.4) — not relevance alone. Retrieval itself counts as an access, so
        returned records get their access_count bumped, which is what lets frequently-useful
        memories compound in importance over time."""
        if self.collection.count() == 0:
            return []

        where = {"user_id": user_id} if user_id else None
        candidate_n = min(self.collection.count(), max(n_results * CANDIDATE_MULTIPLIER, n_results))
        results = self.collection.query(query_texts=[task], n_results=candidate_n, where=where)

        ids = (results.get("ids") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0] or [0.0] * len(ids)
        if not ids:
            return []

        now = datetime.now(UTC)
        scored = []
        for record_id, metadata, distance in zip(ids, metadatas, distances):
            record = MemoryRecord._from_metadata(metadata, id=record_id)
            semantic_score = 1.0 / (1.0 + distance)
            importance = self.compute_importance(record, now)
            blended = RELEVANCE_WEIGHT * semantic_score + IMPORTANCE_WEIGHT_IN_RANKING * importance
            scored.append((blended, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [record for _, record in scored[:n_results]]
        self._bump_access(top, now)
        return top

    def list_for_user(self, user_id: str) -> list[MemoryRecord]:
        """Everything the system currently remembers about one user, most important first —
        the data behind the memory dashboard."""
        result = self.collection.get(where={"user_id": user_id})
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        now = datetime.now(UTC)
        records = [MemoryRecord._from_metadata(m, id=i) for i, m in zip(ids, metadatas)]
        records.sort(key=lambda r: self.compute_importance(r, now), reverse=True)
        return records

    # ---------- delete ----------

    def delete_for_user(self, user_id: str) -> int:
        """Erase everything remembered about one user — the underlying operation behind the
        user-data-deletion endpoint. Returns how many records were removed."""
        result = self.collection.get(where={"user_id": user_id})
        ids = result.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # ---------- expiration ----------

    def expire_stale(
        self,
        max_age_days: float = EXPIRATION_MAX_AGE_DAYS,
        min_access_count: int = EXPIRATION_MIN_ACCESS_COUNT,
        now: datetime | None = None,
    ) -> int:
        """Delete memories that are both old and rarely accessed. Distinct from importance
        decay: decay just deprioritizes a memory in ranking, this actually removes it, so the
        store doesn't grow forever with one-off tasks nobody's ever revisited."""
        now = now or datetime.now(UTC)
        result = self.collection.get()
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []

        stale_ids = []
        for record_id, metadata in zip(ids, metadatas):
            record = MemoryRecord._from_metadata(metadata, id=record_id)
            if self.age_days(record, now) > max_age_days and record.access_count < min_access_count:
                stale_ids.append(record_id)

        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(stale_ids)
