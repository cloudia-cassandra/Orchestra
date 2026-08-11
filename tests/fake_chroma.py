"""Minimal in-memory stand-in for a chromadb Collection, covering only what LongTermMemory and
MemoryConsolidator call. Ranks query results by word overlap instead of real embeddings —
deterministic, no model download or network needed. Distances follow chromadb's convention:
lower means more similar (0.0 = identical bag of words, 1.0 = no overlap at all)."""


class FakeChromaCollection:
    def __init__(self):
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def add(self, ids, documents, metadatas):
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

    def count(self) -> int:
        return len(self._ids)

    @staticmethod
    def _matches_where(metadata: dict, where: dict | None) -> bool:
        if not where:
            return True
        return all(metadata.get(key) == value for key, value in where.items())

    def query(self, query_texts, n_results=5, where=None):
        query_words = set(query_texts[0].lower().split())
        candidates = [
            (record_id, doc, meta)
            for record_id, doc, meta in zip(self._ids, self._documents, self._metadatas)
            if self._matches_where(meta, where)
        ]
        scored = []
        for record_id, doc, meta in candidates:
            overlap = len(query_words & set(doc.lower().split()))
            distance = 1.0 / (1.0 + overlap)
            scored.append((distance, record_id, doc, meta))
        scored.sort(key=lambda row: row[0])  # ascending distance = most similar first

        top = scored[:n_results]
        return {
            "ids": [[row[1] for row in top]],
            "documents": [[row[2] for row in top]],
            "metadatas": [[row[3] for row in top]],
            "distances": [[row[0] for row in top]],
        }

    def get(self, ids=None, where=None):
        result_ids, result_metadatas = [], []
        for record_id, meta in zip(self._ids, self._metadatas):
            if ids is not None and record_id not in ids:
                continue
            if not self._matches_where(meta, where):
                continue
            result_ids.append(record_id)
            result_metadatas.append(meta)
        return {"ids": result_ids, "metadatas": result_metadatas}

    def update(self, ids, metadatas):
        for record_id, meta in zip(ids, metadatas):
            self._metadatas[self._ids.index(record_id)] = meta

    def delete(self, ids=None, where=None):
        keep = [
            idx
            for idx, (record_id, meta) in enumerate(zip(self._ids, self._metadatas))
            if not (
                (ids is not None and record_id in ids)
                or (where is not None and self._matches_where(meta, where))
            )
        ]
        self._ids = [self._ids[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
