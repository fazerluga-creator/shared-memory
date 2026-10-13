"""SQLite backend: ranking, fallbacks and the Chroma-compatible collection API."""
from __future__ import annotations

import pytest

from shared_memory.store_sqlite import SqliteCollection, Store, exact_tokens, where_sql


def vec(*xs):
    return list(xs) + [0.0] * (4 - len(xs))


@pytest.fixture
def col(tmp_path):
    c = SqliteCollection(Store(tmp_path / "m.sqlite", "test/model"), "t")
    c.add(
        ids=["n1", "t1", "n2", "n3", "old"],
        documents=["chat window 48 hours", "chat window 48 hours, chatter",
                   "promo code SPRING2031 for a spare part", "old rule", "ancient fact"],
        metadatas=[
            {"kind": "note", "agent": "a", "timestamp": "2026-10-01T00:00:00+00:00"},
            {"kind": "transcript", "agent": "b", "timestamp": "2026-10-02T00:00:00+00:00"},
            {"kind": "note", "agent": "a", "timestamp": "2026-10-03T00:00:00+00:00"},
            {"kind": "note", "agent": "a", "superseded": True, "timestamp": "2026-10-04T00:00:00+00:00"},
            {"kind": "note", "agent": "a", "timestamp": "2020-01-01T00:00:00+00:00"},
        ],
        embeddings=[vec(1, 0), vec(1, 0), vec(0.8, 0.6), vec(0.9, 0.1), vec(1, 0)],
    )
    return c


def ranked(col, text, qv, **kw):
    return col.query([qv], n_results=10, query_texts=[text], **kw)["ids"][0]


def test_exact_tokens():
    assert exact_tokens("ABC-170") == ["ABC-170"]
    assert exact_tokens("order 4471953") == ["4471953"]
    assert exact_tokens("chat window 48 hours") == []


def test_note_beats_transcript_at_equal_cosine(col):
    order = ranked(col, "chat window", vec(1, 0))
    assert order.index("n1") < order.index("t1")


def test_superseded_ranks_lower(col):
    order = ranked(col, "rule", vec(1, 0))
    # 0.99 * 0.9 < 1.0 * 0.9 (transcript) < 1.0 (note)
    assert order.index("n1") < order.index("t1") < order.index("n3")


def test_exact_token_lifts_chunk(col):
    assert ranked(col, "SPRING2031", vec(1, 0))[0] == "n2"  # 0.8 + 0.3 > 1.0


def test_since_is_applied_before_ranking(col):
    assert "old" not in ranked(col, "fact", vec(1, 0), since="2026-01-01T00:00:00+00:00")
    assert set(ranked(col, "fact", vec(1, 0), since="2026-10-03T00:00:00+00:00")) == {"n2", "n3"}


def test_where_and_where_document(col):
    assert ranked(col, "x", vec(1, 0), where={"agent": {"$eq": "b"}}) == ["t1"]
    assert ranked(col, "x", vec(1, 0), where_document={"$contains": "SPRING"}) == ["n2"]


def test_bm25_fallback_without_vector(col):
    r = col.s.search("SPRING2031", None)
    assert r["mode"] == "bm25" and r["hits"][0]["id"] == "n2"


def test_vectors_of_another_model_are_ignored(col, tmp_path):
    other = Store(col.s.path, "other/model")
    assert other.search("chat", vec(1, 0))["mode"] == "bm25"


def test_update_keeps_vector_and_refreshes_fts(col):
    col.update(ids=["n1"], documents=["brand new text"])
    assert col.s.search("brand", None)["hits"][0]["id"] == "n1"
    assert ranked(col, "x", vec(1, 0))[0] == "n1"  # stored vector kept


def test_update_unknown_id_raises(col):
    with pytest.raises(ValueError):
        col.update(ids=["nope"], documents=["x"])


def test_get_delete_count_paging(col):
    assert col.count() == 5
    assert col.get(limit=2, offset=1)["ids"] == ["t1", "n2"]
    col.delete(where={"agent": "b"})
    assert col.count() == 4 and col.get(ids=[])["ids"] == []


def test_where_sql_rejects_odd_keys():
    with pytest.raises(ValueError):
        where_sql({"a') OR 1=1 --": 1}, [])


def test_other_process_writes_are_seen(col):
    writer = SqliteCollection(Store(col.s.path, "test/model"), "t")
    ranked(col, "x", vec(1, 0))  # build the matrix cache
    writer.add(ids=["fresh"], documents=["fresh"], metadatas=[{"kind": "note"}],
               embeddings=[vec(0, 0, 1)])
    assert ranked(col, "x", vec(0, 0, 1))[0] == "fresh"


def test_mixed_timestamp_formats_compare_correctly(tmp_path):
    c = SqliteCollection(Store(tmp_path / "ts.sqlite", "m"), "t")
    c.add(ids=["z", "off", "naive", "epoch"], documents=["a", "b", "c", "d"],
          metadatas=[{"timestamp": "2026-10-13T10:00:00Z"},
                     {"timestamp": "2026-10-13T12:00:00+02:00"},  # 10:00 UTC
                     {"timestamp": "2026-10-12T23:00:00"},
                     {"timestamp": "1791900000"}],  # 2026-10-13 14:00 UTC
          embeddings=[vec(1, 0)] * 4)
    assert set(ranked(c, "x", vec(1, 0), since="2026-10-13T09:59:59Z")) == {"z", "off", "epoch"}
