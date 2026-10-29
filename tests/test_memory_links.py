"""memory_links migration, automatic links on add, tags, and the graph tool."""
from __future__ import annotations

import logging
import sqlite3

import pytest
from conftest import load_module

from shared_memory.links import GRAPH_NODE_CAP, parse_tags
from shared_memory.store_sqlite import (
    SCHEMA,
    SqliteCollection,
    Store,
    apply_memory_links,
    revert_memory_links,
)

server = load_module("shared_memory/server.py", "shared_memory_server_links")


class Emb:
    """One dimension per keyword, same idea as the SQLite MCP tests."""

    WORDS = ("kettle", "zebra", "giraffe", "window")

    def _v(self, text):
        return [float(w in text) for w in self.WORDS] + [0.01]

    embed_one = embed_document_one = _v


@pytest.fixture
def sqlite_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "BACKEND", "sqlite")
    monkeypatch.setattr(
        server, "_collection", SqliteCollection(Store(tmp_path / "m.sqlite", "toy"), "t"))
    monkeypatch.setattr(server, "_embedder", Emb())
    return server


def fn(tool):
    return getattr(tool, "fn", tool)


def test_parse_tags():
    assert parse_tags("see [[Тема]] and [[ Тема ]] and [[Другое]]") == ["Тема", "Другое"]
    assert parse_tags("no tags") == []
    assert parse_tags("[[]]") == []


def test_migration_is_reversible_and_leaves_chunks_alone(tmp_path):
    path = tmp_path / "m.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    chunks_sql = con.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0]
    chunks_cols = con.execute("PRAGMA table_info(chunks)").fetchall()
    con.execute(
        "INSERT INTO chunks(id, meta, doc, hash) VALUES ('keep', '{}', 'hello', 'h')")
    con.commit()
    con.close()

    assert "memory_links" not in SCHEMA

    store = Store(path, "toy")
    assert store.con.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0] == chunks_sql
    assert store.con.execute("PRAGMA table_info(chunks)").fetchall() == chunks_cols
    assert store.con.execute("SELECT doc FROM chunks WHERE id='keep'").fetchone()[0] == "hello"

    info = store.con.execute("PRAGMA table_info(memory_links)").fetchall()
    assert [(r[1], r[2], r[3], r[5]) for r in info] == [
        ("from_id", "TEXT", 1, 1),
        ("to_id", "TEXT", 1, 2),
        ("type", "TEXT", 1, 3),
        ("score", "REAL", 0, 0),
        ("created_at", "TEXT", 1, 0),
    ]
    names = {r[0] for r in store.con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memory_links'")}
    assert "memory_links_from_id" in names
    assert "memory_links_to_id" in names

    with pytest.raises(sqlite3.IntegrityError):
        store.con.execute(
            "INSERT INTO memory_links(from_id, to_id, type, score, created_at) "
            "VALUES ('a', 'b', 'nope', NULL, 't')")
    store.con.rollback()

    store.add_links([("keep", "other", "manual", 0.5)])
    revert_memory_links(store.con)
    assert store.con.execute(
        "SELECT name FROM sqlite_master WHERE name='memory_links'").fetchone() is None
    assert store.con.execute("SELECT id FROM chunks").fetchone()[0] == "keep"
    assert store.con.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0] == chunks_sql

    apply_memory_links(store.con)
    assert store.con.execute("SELECT count(*) FROM memory_links").fetchone()[0] == 0


def test_add_links_upserts_score_and_skips_self(tmp_path):
    store = Store(tmp_path / "m.sqlite", "toy")
    assert store.add_links([("a", "a", "manual", 1.0)]) == 0
    store.add_links([("a", "b", "manual", 0.5)])
    store.add_links([("a", "b", "manual", 0.9)])
    rows = store.links_for("b")
    assert len(rows) == 1
    assert rows[0]["from_id"] == "a" and rows[0]["to_id"] == "b"
    assert rows[0]["score"] == 0.9
    assert store.links_for("a", link_type="tag") == []
    with pytest.raises(ValueError):
        store.links_for("a", link_type="nope")
    store.add_links([("hub", f"t{i}", "manual", None) for i in range(205)])
    assert len(store.links_for("hub", limit=10_000)) == 200
    assert len(store.links_for("hub", limit=3)) == 3


def test_similar_links_use_threshold_and_skip_self(sqlite_server, monkeypatch):
    add = fn(server.memory_add)
    first = add("kettle zebra", {"agent": "a"})
    second = add("kettle zebra again", {"agent": "a"})
    other = add("giraffe only", {"agent": "a"})

    links = fn(server.memory_links)(second, type="similar")
    assert len(links) == 1
    assert links[0]["from_id"] == second
    assert links[0]["to_id"] == first
    assert links[0]["score"] >= 0.75
    assert links[0]["from_id"] != links[0]["to_id"]
    assert fn(server.memory_links)(second, type="tag") == []
    touching_other = fn(server.memory_links)(other, type="similar")
    assert all(first not in (row["from_id"], row["to_id"]) for row in touching_other)
    assert all(second not in (row["from_id"], row["to_id"]) for row in touching_other)

    monkeypatch.setenv("SHARED_MEMORY_LINK_THRESHOLD", "2")
    blocked = add("kettle zebra once more", {"agent": "a"})
    assert fn(server.memory_links)(blocked, type="similar") == []


def test_tag_links_and_limit(sqlite_server):
    add = fn(server.memory_add)
    first = add("kettle [[Тема]]", {"agent": "a"})
    second = add("zebra [[ Тема ]]", {"agent": "a"})
    tags = fn(server.memory_links)(second, type="tag")
    assert len(tags) == 1
    assert tags[0]["from_id"] == second and tags[0]["to_id"] == first
    assert tags[0]["score"] is None
    assert fn(server.memory_links)(second, type="similar") == []
    # the older chunk sees the incoming edge
    assert fn(server.memory_links)(first, type="tag")[0]["to_id"] == first

    ids = [add(f"note-{i} [[Общее]]", {"agent": "a"}) for i in range(25)]
    linked = fn(server.memory_links)(ids[-1], type="tag", limit=50)
    assert len(linked) == 20
    assert all(row["from_id"] == ids[-1] and row["to_id"] != ids[-1] for row in linked)


def test_graph_nodes_edges_and_caps(sqlite_server, monkeypatch):
    add = fn(server.memory_add)
    first = add("kettle [[Тема]]", {"agent": "notes"})
    second = add("zebra [[Тема]]", {"agent": "notes"})
    graph = fn(server.memory_graph)()
    ids = {node["id"] for node in graph["nodes"]}
    assert first in ids and second in ids
    assert any(edge["type"] == "tag" and {edge["from_id"], edge["to_id"]} == {first, second}
               for edge in graph["edges"])
    assert all(edge["from_id"] in ids and edge["to_id"] in ids for edge in graph["edges"])

    filtered = fn(server.memory_graph)(filter="zebra", limit=10)
    filtered_ids = {node["id"] for node in filtered["nodes"]}
    assert second in filtered_ids and first in filtered_ids
    assert filtered["edges"]

    store = server._collection.s
    n = GRAPH_NODE_CAP + 50
    bulk = [f"n{i:04d}" for i in range(n)]
    store.upsert(bulk, [f"doc {i}" for i in bulk], [{} for _ in bulk], [[1.0, 0.0]] * n)
    store.add_links([
        (bulk[i], bulk[i + 1], "manual", 1.0, f"2026-06-01T00:{i // 60:02d}:{i % 60:02d}+00:00")
        for i in range(n - 1)
    ])
    capped = fn(server.memory_graph)(limit=10_000)
    assert len(capped["nodes"]) == GRAPH_NODE_CAP
    assert len(capped["edges"]) <= 500
    have = {node["id"] for node in capped["nodes"]}
    assert all(edge["from_id"] in have and edge["to_id"] in have for edge in capped["edges"])

    monkeypatch.setattr("shared_memory.links.GRAPH_EDGE_CAP", 3)
    tiny = fn(server.memory_graph)(limit=50)
    assert len(tiny["edges"]) <= 3


def test_forget_removes_links(sqlite_server):
    add = fn(server.memory_add)
    first = add("kettle zebra", {"agent": "a"})
    second = add("kettle zebra too", {"agent": "a"})
    assert fn(server.memory_links)(second, type="similar")
    assert fn(server.memory_forget)(ids=[first], dry_run=False)["deleted"] == 1
    assert fn(server.memory_links)(second) == []


def test_add_survives_linking_failure(sqlite_server, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("link store down")

    monkeypatch.setattr(server, "link_new_chunk", boom)
    with caplog.at_level(logging.ERROR, logger="shared-memory-mcp"):
        cid = fn(server.memory_add)("kettle zebra", {"agent": "a"})
    assert cid
    assert server._collection.get(ids=[cid])["ids"] == [cid]
    assert any("linking failed" in rec.message for rec in caplog.records)


def test_similar_search_error_still_saves_the_chunk(sqlite_server, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("search down")

    monkeypatch.setattr(server._collection.s, "search", boom)
    cid = fn(server.memory_add)("kettle [[Тема]]", {"agent": "a"})
    assert server._collection.get(ids=[cid])["ids"] == [cid]
    assert fn(server.memory_links)(cid, type="similar") == []
