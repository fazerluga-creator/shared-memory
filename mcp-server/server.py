#!/usr/bin/env python3
"""MCP server exposing shared vector memory.

Tools:
  - memory_search(query, top_k=5, agent_filter=None, source_filter=None, since=None)
      Returns list of {text, metadata, score}.
  - memory_add(text, metadata)
      Returns id of the added chunk.
  - memory_stats()
      Returns {total, by_agent, by_source, collection}.

Storage is a persistent ChromaDB directory (see lib/store.py, configurable via
$SHARED_MEMORY_DATA_DIR). Embeddings are provider-aware; the default is a local
fastembed model configured by embedding.env (see embedding.env.example).
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.embedder import Embedder  # noqa: E402
from lib.store import (  # noqa: E402
    DEFAULT_COLLECTION,
    get_client,
    get_collection,
)

from fastmcp import FastMCP  # noqa: E402

LOG = logging.getLogger("shared-memory-mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)

mcp = FastMCP("shared-memory")

# Lazy singletons
_embedder: Embedder | None = None
_collection = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_col():
    global _collection
    if _collection is None:
        _collection = get_collection(get_client(), name=DEFAULT_COLLECTION)
    return _collection


def _normalize_since(since: str) -> str:
    """Normalize `since` to the ISO format used for timestamps in the collection.

    Accepts an ISO string or a unix-epoch string (digits only).
    """
    s = since.strip()
    if s.isdigit():
        return datetime.fromtimestamp(int(s), tz=timezone.utc).isoformat()
    return s


def _build_where(
    agent_filter: str | None,
    source_filter: str | None,
) -> dict[str, Any] | None:
    # NOTE: `since` is not applied here. Chroma's $gte works on numbers only,
    # while the timestamp stored in the collection is an ISO string, so `since`
    # is filtered afterwards in memory_search (ISO sorts lexicographically).
    clauses: list[dict[str, Any]] = []
    if agent_filter:
        clauses.append({"agent": {"$eq": agent_filter}})
    if source_filter:
        clauses.append({"source": {"$eq": source_filter}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@mcp.tool()
def memory_search(
    query: str,
    top_k: int = 5,
    agent_filter: str | None = None,
    source_filter: str | None = None,
    since: str | None = None,
) -> list[dict]:
    """Semantic search across shared memory.

    Args:
        query: natural-language query
        top_k: number of results (default 5, max 50)
        agent_filter: restrict to a single agent label stored in metadata.agent
        source_filter: restrict to a metadata.source value (e.g. 'sessions', 'memory')
        since: ISO timestamp lower bound on metadata.timestamp

    Returns:
        [{text, metadata, score}, ...] sorted by relevance desc.
    """
    top_k = max(1, min(50, int(top_k)))
    col = _get_col()
    emb = _get_embedder().embed_one(query)
    where = _build_where(agent_filter, source_filter)
    since_iso = _normalize_since(since) if since else None

    kwargs: dict[str, Any] = {
        "query_embeddings": [emb],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    # Fetch more than top_k so we can dedupe and filter noise, then trim.
    kwargs["n_results"] = min(top_k * 3, 100)
    res = col.query(**kwargs)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    seen_hashes: set[str] = set()
    out: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not doc:
            continue
        # since filter: ISO strings compare lexicographically
        if since_iso and ((meta or {}).get("timestamp") or "") < since_iso:
            continue
        # Skip known LLM-error noise
        stripped = doc.strip()
        if len(stripped) < 300 and (
            "API Error" in stripped or "Content block not found" in stripped
        ):
            continue
        # Content-level dedup: identical chunks from different files
        h = hashlib.sha1(doc.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        score = 1.0 - float(dist) if dist is not None else 0.0
        out.append({"text": doc, "metadata": meta or {}, "score": round(score, 4)})
        if len(out) >= top_k:
            break
    return out


@mcp.tool()
def memory_add(text: str, metadata: dict | None = None) -> str:
    """Add an ad-hoc memory chunk. Returns the generated id."""
    if not text or not text.strip():
        raise ValueError("empty text")
    meta = dict(metadata or {})
    meta.setdefault("agent", "manual")
    meta.setdefault("source", "memory_add")
    # ISO UTC is the single timestamp format used across the collection
    # (not unix-epoch!).
    meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    if str(meta["timestamp"]).isdigit():
        meta["timestamp"] = datetime.fromtimestamp(
            int(meta["timestamp"]), tz=timezone.utc
        ).isoformat()

    col = _get_col()
    emb = _get_embedder().embed_document_one(text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    cid = f"{meta['agent']}::manual::{int(time.time())}::{digest}"
    col.add(
        ids=[cid],
        embeddings=[emb],
        documents=[text],
        metadatas=[meta],
    )
    return cid


@mcp.tool()
def memory_stats() -> dict:
    """Return collection statistics: total count, breakdown by agent/source."""
    col = _get_col()
    total = col.count()
    if total == 0:
        return {"total": 0, "by_agent": {}, "by_source": {}, "collection": col.name}

    # Fetch all metadatas (only metadata, not docs/embeddings — cheap).
    got = col.get(include=["metadatas"])
    metas = got.get("metadatas") or []
    by_agent: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for m in metas:
        if not m:
            continue
        a = m.get("agent", "?")
        s = m.get("source", "?")
        by_agent[a] = by_agent.get(a, 0) + 1
        by_source[s] = by_source.get(s, 0) + 1
    return {
        "total": total,
        "by_agent": by_agent,
        "by_source": by_source,
        "collection": col.name,
    }


if __name__ == "__main__":
    mcp.run()
