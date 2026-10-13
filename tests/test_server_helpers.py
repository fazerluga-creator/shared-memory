"""Tests for pure helpers in mcp-server/server.py."""
from __future__ import annotations

from conftest import load_module

server = load_module("shared_memory/server.py", "shared_memory_server")


def test_normalize_since_passes_iso_through():
    out = server._normalize_since("2026-07-01T00:00:00+00:00")
    assert out.isoformat().startswith("2026-07-01T00:00:00")


def test_normalize_since_converts_epoch():
    out = server._normalize_since("1751328000")  # 2025-07-01 UTC
    assert out.year == 2025
    assert out.month == 7
    assert out.day == 1


def test_normalize_since_accepts_z_suffix():
    out = server._normalize_since("2026-07-01T00:00:00Z")
    assert out.isoformat().startswith("2026-07-01T00:00:00")


def test_build_where_empty():
    assert server._build_where(None, None) is None


def test_build_where_single_clause():
    assert server._build_where("agent-a", None) == {"agent": {"$eq": "agent-a"}}
    assert server._build_where(None, "markdown") == {"source": {"$eq": "markdown"}}


def test_build_where_combines_with_and():
    out = server._build_where("agent-a", "markdown")
    assert out == {
        "$and": [
            {"agent": {"$eq": "agent-a"}},
            {"source": {"$eq": "markdown"}},
        ]
    }


def test_distance_to_score_clamps_negative():
    assert server._distance_to_score(0.0) == 1.0
    assert server._distance_to_score(0.25) == 0.75
    assert server._distance_to_score(1.5) == 0.0
    assert server._distance_to_score(None) == 0.0


def test_fetch_n_larger_when_since_set():
    assert server._fetch_n_for_search(5, since=False) == 15
    assert server._fetch_n_for_search(5, since=True) == 50
    assert server._fetch_n_for_search(50, since=True) == 200
