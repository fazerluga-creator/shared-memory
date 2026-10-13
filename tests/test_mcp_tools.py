"""Integration-style tests for MCP tools with fake collection/embedder."""
from __future__ import annotations

from conftest import load_module

server = load_module("mcp-server/server.py", "shared_memory_server_integration")
# The fakes below mimic a Chroma collection; SQLite-specific behaviour (since in
# SQL, query_texts) is covered by tests/test_mcp_sqlite.py.
server.BACKEND = "chroma"


class FakeEmbedder:
    def embed_one(self, text: str):
        return [0.1, 0.2, 0.3]

    def embed_document_one(self, text: str):
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self):
        self.name = "shared_memory"
        self._docs: dict[str, dict] = {}
        self.last_query = None

    def count(self):
        return len(self._docs)

    def add(self, ids, embeddings, documents, metadatas):
        for i, doc, meta in zip(ids, documents, metadatas, strict=True):
            self._docs[i] = {"document": doc, "metadata": meta, "embedding": embeddings[0]}

    def get(self, include=None, limit=None, offset=0, where=None):
        items = list(self._docs.values())
        if limit is not None:
            items = items[offset : offset + limit]
        elif offset:
            items = items[offset:]
        return {"metadatas": [d["metadata"] for d in items], "ids": list(self._docs)[offset:]}

    def query(self, **kwargs):
        self.last_query = kwargs
        # Return in insertion order with increasing distance.
        docs, metas, dists = [], [], []
        for i, (_cid, row) in enumerate(self._docs.items()):
            docs.append(row["document"])
            metas.append(row["metadata"])
            dists.append(0.1 * i)
        n = kwargs.get("n_results", len(docs))
        return {
            "documents": [docs[:n]],
            "metadatas": [metas[:n]],
            "distances": [dists[:n]],
            "ids": [list(self._docs)[:n]],
        }


def _install_fakes(monkeypatch):
    col = FakeCollection()
    emb = FakeEmbedder()
    monkeypatch.setattr(server, "_get_col", lambda: col)
    monkeypatch.setattr(server, "_get_embedder", lambda: emb)
    return col


def test_memory_add_and_stats(monkeypatch):
    col = _install_fakes(monkeypatch)
    cid = server.memory_add("remember the beta ship date", {"agent": "notes"})
    assert "::manual::" in cid
    assert col.count() == 1
    stats = server.memory_stats()
    assert stats["total"] == 1
    assert stats["by_agent"]["notes"] == 1
    assert stats["by_source"]["memory_add"] == 1


def test_memory_search_filters_since_and_clamps_score(monkeypatch):
    col = _install_fakes(monkeypatch)
    col.add(
        ids=["old"],
        embeddings=[[0.1]],
        documents=["old decision about alpha"],
        metadatas=[{"agent": "notes", "source": "memory_add", "timestamp": "2020-01-01T00:00:00+00:00"}],
    )
    col.add(
        ids=["new"],
        embeddings=[[0.1]],
        documents=["new decision about beta"],
        metadatas=[{"agent": "notes", "source": "memory_add", "timestamp": "2026-07-01T00:00:00+00:00"}],
    )
    # Force a large distance on first hit to verify clamping path is exercised on
    # later items; FakeCollection uses 0.1 * index.
    results = server.memory_search("decision", top_k=5, since="2025-01-01T00:00:00Z")
    assert len(results) == 1
    assert "beta" in results[0]["text"]
    assert results[0]["score"] >= 0.0

    # Without since, both come back; score never negative even if distance > 1.
    col._docs.clear()
    col.add(
        ids=["far"],
        embeddings=[[0.1]],
        documents=["far away chunk"],
        metadatas=[{"agent": "notes", "source": "x", "timestamp": "2026-01-01T00:00:00+00:00"}],
    )
    # Monkey-patch query to return distance > 1.
    def far_query(**kwargs):
        return {
            "documents": [["far away chunk"]],
            "metadatas": [[{"agent": "notes", "source": "x", "timestamp": "2026-01-01T00:00:00+00:00"}]],
            "distances": [[1.5]],
        }

    monkeypatch.setattr(col, "query", far_query)
    out = server.memory_search("far")
    assert out[0]["score"] == 0.0


def test_memory_search_overfetches_when_since_set(monkeypatch):
    col = _install_fakes(monkeypatch)
    server.memory_search("q", top_k=5, since="2026-01-01T00:00:00Z")
    assert col.last_query["n_results"] == 50
    server.memory_search("q", top_k=5)
    assert col.last_query["n_results"] == 15
