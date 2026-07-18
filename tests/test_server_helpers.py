"""Tests for pure helpers in mcp-server/server.py."""
from __future__ import annotations

from conftest import load_module

server = load_module("mcp-server/server.py", "shared_memory_server")


def test_normalize_since_passes_iso_through():
    assert server._normalize_since("2026-07-01T00:00:00+00:00") == "2026-07-01T00:00:00+00:00"


def test_normalize_since_converts_epoch():
    out = server._normalize_since("1751328000")  # 2025-07-01 UTC
    assert out.startswith("2025-07-01")
    assert "T" in out


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
