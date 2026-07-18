"""Tests for lib/store.py helpers using fake collections/embedders."""
from __future__ import annotations

import pytest
from conftest import ROOT  # noqa: F401

from lib.store import add_chunks, existing_session_ids


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeCollection:
    def __init__(self, metadatas=None):
        self._metadatas = metadatas or []
        self.added = None
        self.upserted = None

    def get(self, where=None, include=None):
        return {"metadatas": self._metadatas}

    def add(self, **kwargs):
        self.added = kwargs

    def upsert(self, **kwargs):
        self.upserted = kwargs


def test_add_chunks_noop_on_empty():
    col = FakeCollection()
    add_chunks(col, FakeEmbedder(), [], [], [])
    assert col.added is None and col.upserted is None


def test_add_chunks_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        add_chunks(FakeCollection(), FakeEmbedder(), ["a"], ["t1", "t2"], [{}])


def test_add_chunks_add_vs_upsert():
    col = FakeCollection()
    add_chunks(col, FakeEmbedder(), ["a"], ["text"], [{"k": 1}])
    assert col.added is not None and col.upserted is None
    assert col.added["ids"] == ["a"]
    assert col.added["embeddings"] == [[0.1, 0.2]]

    col2 = FakeCollection()
    add_chunks(col2, FakeEmbedder(), ["a"], ["text"], [{"k": 1}], upsert=True)
    assert col2.upserted is not None and col2.added is None


def test_existing_session_ids_extracts_unique():
    col = FakeCollection(
        metadatas=[
            {"session_id": "s1"},
            {"session_id": "s2"},
            {"session_id": "s1"},
            {"other": "x"},
            None,
        ]
    )
    assert existing_session_ids(col) == {"s1", "s2"}


def test_existing_session_ids_survives_get_failure():
    class Broken:
        def get(self, **kwargs):
            raise RuntimeError("empty collection")

    assert existing_session_ids(Broken()) == set()
