"""MCP tools end-to-end on the real SQLite backend (fake embedder, no network)."""
from __future__ import annotations

import pytest
from conftest import load_module

from shared_memory.store_sqlite import SqliteCollection, Store

server = load_module("shared_memory/server.py", "shared_memory_server_sqlite")


class Emb:
    """Deterministic toy embedder: one dimension per keyword."""

    WORDS = ("kettle", "zebra", "giraffe", "window")

    def _v(self, text):
        return [float(w in text) for w in self.WORDS] + [0.01]

    embed_one = embed_document_one = _v


@pytest.fixture(autouse=True)
def sqlite_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "BACKEND", "sqlite")
    monkeypatch.setattr(server, "_collection", SqliteCollection(Store(tmp_path / "m.sqlite", "toy"), "t"))
    monkeypatch.setattr(server, "_embedder", Emb())


def fn(tool):
    return getattr(tool, "fn", tool)


def test_add_search_update_forget_roundtrip():
    cid = fn(server.memory_add)("kettle zebra note", {"agent": "a"})
    hit = fn(server.memory_search)("kettle zebra", top_k=1)[0]
    assert hit["id"] == cid and hit["text"] == "kettle zebra note"

    r = fn(server.memory_update)(cid, text="kettle giraffe note", metadata={"kind": "note"})
    assert r["changed"] == ["metadata", "text"] and r["metadata"]["agent"] == "a"
    assert fn(server.memory_search)("giraffe", top_k=1)[0]["text"] == "kettle giraffe note"

    dry = fn(server.memory_forget)(ids=[cid])
    assert dry["matched"] == 1 and dry["deleted"] == 0
    assert fn(server.memory_forget)(ids=[cid], dry_run=False)["deleted"] == 1
    assert fn(server.memory_stats)()["total"] == 0


def test_update_unknown_id_raises():
    with pytest.raises(ValueError):
        fn(server.memory_update)("nope", text="x")


def test_forget_by_filter_needs_expect_count():
    for i in range(3):
        fn(server.memory_add)(f"window {i}", {"agent": "bulk"})
    with pytest.raises(ValueError):
        fn(server.memory_forget)(agent_filter="bulk", dry_run=False)
    assert fn(server.memory_forget)(agent_filter="bulk", dry_run=False, expect_count=3)["deleted"] == 3
    with pytest.raises(ValueError):
        fn(server.memory_forget)()


def test_since_does_not_starve_results():
    # 30 strong old matches and one weak fresh one: the fresh one must still come back
    for i in range(30):
        fn(server.memory_add)(f"window window {i}", {"timestamp": "2020-01-01T00:00:00Z"})
    fresh = fn(server.memory_add)("kettle, also a window", {"timestamp": "2026-10-13T00:00:00Z"})
    hits = fn(server.memory_search)("window", top_k=1, since="2026-01-01")
    assert [h["id"] for h in hits] == [fresh]


def test_stats_reports_backend():
    assert fn(server.memory_stats)()["backend"] == "sqlite"
