#!/usr/bin/env python3
"""MCP server exposing shared vector memory.

Tools:
  - memory_search(query, top_k=5, agent_filter=None, source_filter=None, since=None)
      Returns list of {id, text, metadata, score}.
  - memory_add(text, metadata)
      Returns id of the added chunk. On SQLite, also records similar and [[tag]] links.
  - memory_update(id, text=None, metadata=None, replace_metadata=False)
      Edits a chunk in place; the id is kept.
  - memory_forget(ids=None, agent_filter=None, source_filter=None, dry_run=True, expect_count=None)
      Deletes chunks; dry run by default.
  - memory_stats()
      Returns {total, by_agent, by_source, collection, backend}.
  - memory_links(id, type=None, limit=50)
      Links touching one chunk.
  - memory_graph(filter=None, limit=100)
      ``{nodes, edges}`` with a hard size cap.

Storage is a single SQLite file by default (shared_memory/store_sqlite.py) or a ChromaDB
directory (SHARED_MEMORY_BACKEND=chroma); see shared_memory/store.py. Embeddings are
provider-aware; the default is a local fastembed model configured by
embedding.env (see embedding.env.example).

SECURITY: the server has no authentication. Anyone who can reach it can read,
change and delete every memory. Bind it to 127.0.0.1 (FastMCP's default) and
never expose it to a network you do not fully control.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP

from .embedder import Embedder
from .links import build_graph, link_new_chunk
from .store import (
    BACKEND,
    DEFAULT_COLLECTION,
    format_timestamp,
    get_client,
    get_collection,
    parse_timestamp,
)

LOG = logging.getLogger("shared-memory-mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)

mcp = FastMCP("shared-memory", version="0.3.0")

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


def _normalize_since(since: str) -> datetime:
    """Normalize `since` to UTC for comparisons against collection timestamps."""
    return parse_timestamp(since)


def _build_where(
    agent_filter: str | None,
    source_filter: str | None,
) -> dict[str, Any] | None:
    # NOTE: `since` is not part of this filter. Chroma's $gte works on numbers
    # only while timestamps are ISO strings, so on the chroma backend `since` is
    # filtered afterwards; the sqlite backend applies it in SQL before ranking.
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


def _distance_to_score(dist: float | None) -> float:
    """Map cosine distance to a non-negative relevance score."""
    if dist is None:
        return 0.0
    return max(0.0, 1.0 - float(dist))


def _fetch_n_for_search(top_k: int, *, since: bool) -> int:
    # Over-fetch so post-filters (since / dedupe / noise) can still fill top_k.
    multiplier = 10 if since else 3
    return min(max(top_k * multiplier, top_k), 200)


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
        [{id, text, metadata, score}, ...] sorted by relevance desc. The id can be
        passed to memory_update / memory_forget. Scores are clamped to ``[0, 1]``.
    """
    top_k = max(1, min(50, int(top_k)))
    col = _get_col()
    emb = _get_embedder().embed_one(query)
    where = _build_where(agent_filter, source_filter)
    since_dt = _normalize_since(since) if since else None

    kwargs: dict[str, Any] = {
        "query_embeddings": [emb],
        "n_results": _fetch_n_for_search(top_k, since=since_dt is not None),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    sql_since = BACKEND == "sqlite"
    if sql_since:
        # Query text drives exact-token boosting; `since` is filtered in SQL, so a
        # narrow window cannot be starved by older chunks that rank higher.
        kwargs["query_texts"] = [query]
        if since_dt:
            kwargs["since"] = format_timestamp(since_dt)
    res = col.query(**kwargs)
    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0] or [None] * len(docs)
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    seen_hashes: set[str] = set()
    out: list[dict] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
        if not doc:
            continue
        # since filter (chroma backend): compare parsed UTC datetimes.
        if since_dt and not sql_since:
            raw_ts = (meta or {}).get("timestamp")
            if not raw_ts:
                continue
            try:
                if parse_timestamp(str(raw_ts)) < since_dt:
                    continue
            except ValueError:
                LOG.debug("skipping chunk with invalid timestamp: %r", raw_ts)
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
        score = _distance_to_score(dist)
        out.append({"id": cid, "text": doc, "metadata": meta or {}, "score": round(score, 4)})
        if len(out) >= top_k:
            break
    return out


@mcp.tool()
def memory_add(text: str, metadata: dict | None = None) -> str:
    """Add an ad-hoc memory chunk. Returns the generated id.

    On SQLite, the insert is followed by up to 5 ``similar`` links (hybrid
    score ≥ ``SHARED_MEMORY_LINK_THRESHOLD``, default 0.75) and up to 20
    ``[[topic]]`` tag links. A linking failure is logged and does not fail
    the add.
    """
    if not text or not text.strip():
        raise ValueError("empty text")
    meta = dict(metadata or {})
    meta.setdefault("agent", "manual")
    meta.setdefault("source", "memory_add")
    # ISO UTC is the single timestamp format used across the collection
    # (not unix-epoch!).
    meta.setdefault("timestamp", format_timestamp())
    if str(meta["timestamp"]).isdigit():
        meta["timestamp"] = format_timestamp(
            datetime.fromtimestamp(int(meta["timestamp"]), tz=timezone.utc)
        )
    elif meta.get("timestamp"):
        try:
            meta["timestamp"] = format_timestamp(parse_timestamp(str(meta["timestamp"])))
        except ValueError:
            meta["timestamp"] = format_timestamp()

    col = _get_col()
    emb = _get_embedder().embed_document_one(text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    cid = f"{meta['agent']}::manual::{uuid.uuid4().hex[:12]}::{digest}"
    col.add(
        ids=[cid],
        embeddings=[emb],
        documents=[text],
        metadatas=[meta],
    )
    # Linking is best-effort: a search or SQL failure must not lose the insert.
    try:
        link_new_chunk(col, cid, text, emb)
    except Exception:
        LOG.exception("memory_add saved %s but linking failed", cid)
    return cid


@mcp.tool()
def memory_update(
    id: str,
    text: str | None = None,
    metadata: dict | None = None,
    replace_metadata: bool = False,
) -> dict:
    """Update an existing memory chunk in place; the id is kept.

    Keeping the id means references to it stay valid. An unknown id raises, so a
    typo never silently turns into a new chunk.

    Args:
        id: chunk id (memory_search returns it in "id")
        text: new text; the embedding is recomputed when given
        metadata: metadata fields to change (merged unless replace_metadata)
        replace_metadata: replace metadata wholesale instead of merging

    Returns:
        {id, changed: [...], metadata: {...}}
    """
    if not id or not id.strip():
        raise ValueError("empty id")
    if text is None and metadata is None:
        raise ValueError("nothing to update: pass text and/or metadata")
    if text is not None and not text.strip():
        raise ValueError("empty text: use memory_forget to remove a chunk")

    col = _get_col()
    got = col.get(ids=[id], include=["documents", "metadatas"])
    if not (got.get("ids") or []):
        raise ValueError(f"unknown id: {id!r}")

    old_meta = dict((got.get("metadatas") or [{}])[0] or {})
    if replace_metadata:
        meta = dict(metadata or {})
        # agent and source are the filter axes; without them a chunk drops out
        # of agent_filter / source_filter searches.
        meta.setdefault("agent", old_meta.get("agent", "manual"))
        meta.setdefault("source", old_meta.get("source", "memory_add"))
        meta.setdefault("timestamp", old_meta.get("timestamp") or format_timestamp())
    else:
        meta = old_meta
        meta.update(metadata or {})
    meta["updated_at"] = format_timestamp()

    kwargs: dict[str, Any] = {"ids": [id], "metadatas": [meta]}
    changed = ["metadata"] if metadata is not None or replace_metadata else []
    if text is not None:
        kwargs["documents"] = [text]
        kwargs["embeddings"] = [_get_embedder().embed_document_one(text)]
        changed.append("text")
    col.update(**kwargs)
    LOG.info("memory_update id=%s changed=%s", id, ",".join(changed))
    return {"id": id, "changed": changed, "metadata": meta}


@mcp.tool()
def memory_forget(
    ids: list[str] | None = None,
    agent_filter: str | None = None,
    source_filter: str | None = None,
    dry_run: bool = True,
    expect_count: int | None = None,
) -> dict:
    """Delete memory chunks by explicit ids or by agent/source filter.

    Deletion is irreversible, so there are two safety catches: dry_run is on by
    default, and deleting by filter requires expect_count equal to the number
    of matches, so an over-broad filter cannot wipe the collection by mistake.
    A call with no ids and no filters is refused.

    Args:
        ids: specific chunks (memory_search returns ids)
        agent_filter: everything from this agent
        source_filter: everything from this source
        dry_run: True (default) only reports what would be deleted
        expect_count: required for filter deletes with dry_run=False

    Returns:
        {matched, deleted, dry_run, sample: [...], missing: [...]}
    """
    if not ids and not agent_filter and not source_filter:
        raise ValueError("refusing to delete everything: pass ids, agent_filter or source_filter")

    col = _get_col()
    missing: list[str] = []
    if ids:
        got = col.get(ids=list(ids), include=["metadatas"])
        found = list(got.get("ids") or [])
        missing = [i for i in ids if i not in set(found)]
        targets = found
    else:
        got = col.get(where=_build_where(agent_filter, source_filter), include=["metadatas"])
        targets = list(got.get("ids") or [])

    matched = len(targets)
    result: dict[str, Any] = {"matched": matched, "deleted": 0, "dry_run": dry_run,
                              "sample": targets[:10]}
    if missing:
        result["missing"] = missing
    if dry_run:
        result["hint"] = "dry run: nothing deleted. Call again with dry_run=False" + (
            f" and expect_count={matched}" if not ids else "")
        return result
    if not ids and expect_count != matched:
        raise ValueError(
            f"expect_count mismatch: filter matches {matched}, expect_count={expect_count}. "
            "Re-run with dry_run=True first.")
    if matched:
        col.delete(ids=targets)
        result["deleted"] = matched
        LOG.warning("memory_forget deleted=%d agent=%s source=%s",
                    matched, agent_filter, source_filter)
    return result


_STATS_PAGE = 500


@mcp.tool()
def memory_stats() -> dict:
    """Return collection statistics: total count, breakdown by agent/source."""
    col = _get_col()
    total = col.count()
    if total == 0:
        return {"total": 0, "by_agent": {}, "by_source": {}, "collection": col.name,
                "backend": BACKEND}

    # Page through metadatas to keep peak memory bounded on large collections.
    by_agent: dict[str, int] = {}
    by_source: dict[str, int] = {}
    offset = 0
    while offset < total:
        try:
            got = col.get(include=["metadatas"], limit=_STATS_PAGE, offset=offset)
        except TypeError:
            # Older Chroma without limit/offset — fall back to a single fetch.
            got = col.get(include=["metadatas"])
            offset = total  # exit after this page
        metas = got.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            if not m:
                continue
            a = m.get("agent", "?")
            s = m.get("source", "?")
            by_agent[a] = by_agent.get(a, 0) + 1
            by_source[s] = by_source.get(s, 0) + 1
        offset += len(metas)
        if len(metas) < _STATS_PAGE:
            break
    return {
        "total": total,
        "by_agent": by_agent,
        "by_source": by_source,
        "collection": col.name,
        "backend": BACKEND,
    }


@mcp.tool()
def memory_links(id: str, type: str | None = None, limit: int = 50) -> list[dict]:
    """Links that touch one chunk, in either direction.

    Args:
        id: chunk id (``memory_search`` / ``memory_add`` return it)
        type: optional ``similar``, ``tag``, ``llm`` or ``manual``
        limit: max rows (default 50, hard cap 200)

    Returns:
        ``[{from_id, to_id, type, score, created_at}, ...]`` newest first.
        ``score`` is null for tag links. Empty on the Chroma backend.
    """
    if not id or not str(id).strip():
        raise ValueError("empty id")
    store = getattr(_get_col(), "s", None)
    if store is None:
        return []
    return store.links_for(id, type, limit)


@mcp.tool()
def memory_graph(filter: str | None = None, limit: int = 100) -> dict:
    """Cross-link graph for the SQLite store.

    Args:
        filter: optional substring of a chunk id, its text, or ``metadata.agent``.
            Direct neighbours are included while the node cap allows.
        limit: max nodes (default 100, hard cap 200). Edges are capped at 500
            and only connect nodes in the response.

    Returns:
        ``{"nodes": [{id, text, metadata}, ...], "edges": [{from_id, to_id, type, score}, ...]}``.
        Empty when the backend is not SQLite.
    """
    store = getattr(_get_col(), "s", None)
    if store is None:
        return {"nodes": [], "edges": []}
    return build_graph(store, filter, limit)


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def check_bind_host(host: str, allow_remote: bool) -> None:
    """Refuse a non-loopback HTTP bind unless explicitly allowed.

    The server has no authentication: whoever reaches the port can read, rewrite
    and delete every memory. Exposing it must be a deliberate choice.
    """
    if host in _LOOPBACK:
        return
    if not allow_remote:
        raise SystemExit(
            f"refusing to listen on {host!r}: shared-memory has no authentication. "
            "Use 127.0.0.1, or pass --allow-unauthenticated-remote if a VPN or an "
            "authenticating proxy protects this port."
        )
    LOG.warning("listening on %s WITHOUT authentication: anyone who can reach this "
                "port can read, change and delete every memory", host)


def main(argv: list[str] | None = None) -> None:
    """Console entry point ``shared-memory-mcp``: stdio by default.

    ``shared-memory-mcp --transport http`` serves streamable HTTP on
    127.0.0.1:8000; see the Security section of the README before changing --host.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="shared-memory-mcp", description=__doc__.split("\n")[0])
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-unauthenticated-remote", action="store_true",
                    help="bind HTTP to a non-loopback address even though the server has "
                         "no authentication (put it behind a VPN or an authenticating proxy)")
    a = ap.parse_args(argv)
    if a.transport == "http":
        check_bind_host(a.host, a.allow_unauthenticated_remote)
    if a.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="http", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
