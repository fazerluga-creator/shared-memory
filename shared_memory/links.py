"""Cross-links between memory chunks (SQLite).

``memory_add`` calls :func:`link_new_chunk` after a successful insert:

* ``similar`` — hybrid-search neighbours in the top 5 whose score is at or
  above ``SHARED_MEMORY_LINK_THRESHOLD`` (default 0.75);
* ``tag`` — other chunks that share a ``[[topic]]`` marker (at most 20).

Failures are logged and swallowed by the caller so adding a memory still
succeeds. The graph helpers behind ``memory_links`` / ``memory_graph`` live
here too.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone

LOG = logging.getLogger("shared-memory-mcp")

SIMILAR_TOP_K = 5
TAG_LINK_LIMIT = 20
DEFAULT_LINK_THRESHOLD = 0.75
GRAPH_NODE_CAP = 200
GRAPH_EDGE_CAP = 500
_TAG_MAX = 80

# ``[[Тема]]`` — inner text, no nested brackets. Whitespace is collapsed.
_TAG_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")


def parse_tags(text: str) -> list[str]:
    """Unique ``[[topic]]`` markers in order, inner whitespace collapsed."""
    found: list[str] = []
    for raw in _TAG_RE.findall(text or ""):
        tag = " ".join(raw.split())
        if not tag or len(tag) > _TAG_MAX or tag in found:
            continue
        found.append(tag)
    return found


def link_threshold() -> float:
    """Score cutoff for ``similar`` links. Tag links ignore it."""
    from .store import _setting

    raw = _setting("SHARED_MEMORY_LINK_THRESHOLD", str(DEFAULT_LINK_THRESHOLD))
    try:
        return float(raw)
    except (TypeError, ValueError):
        LOG.warning("invalid SHARED_MEMORY_LINK_THRESHOLD %r; using %s",
                    raw, DEFAULT_LINK_THRESHOLD)
        return DEFAULT_LINK_THRESHOLD


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def link_new_chunk(collection, cid: str, text: str, embedding) -> dict:
    """Write similar and tag links for a chunk that was just inserted.

    No-op when the collection is not the SQLite store (Chroma has no
    ``memory_links`` table). Similar and tag passes fail independently.
    """
    store = getattr(collection, "s", None)
    stats = {"similar": 0, "tag": 0}
    if store is None:
        return stats
    try:
        stats["similar"] = _link_similar(store, cid, text, embedding)
    except Exception:
        LOG.exception("similar links failed for %s", cid)
    try:
        stats["tag"] = _link_tags(store, cid, text)
    except Exception:
        LOG.exception("tag links failed for %s", cid)
    return stats


def _link_similar(store, cid: str, text: str, embedding) -> int:
    threshold = link_threshold()
    # +1 so the chunk's own hit does not use up one of the five slots.
    hits = store.search(text or "", embedding, SIMILAR_TOP_K + 1)["hits"]
    rows = []
    for hit in hits:
        if hit["id"] == cid:
            continue
        score = float(hit["score"])
        if not math.isfinite(score) or score < threshold:
            continue
        rows.append((cid, hit["id"], "similar", score, _now()))
        if len(rows) >= SIMILAR_TOP_K:
            break
    return store.add_links(rows)


def _link_tags(store, cid: str, text: str) -> int:
    tags = parse_tags(text)
    if not tags:
        return 0
    rows = []
    seen: set[str] = set()
    for tag in tags:
        if len(rows) >= TAG_LINK_LIMIT:
            break
        for other_id, doc in store.docs_containing(tag, cid, limit=200):
            if other_id in seen:
                continue
            if tag not in parse_tags(doc):
                continue
            seen.add(other_id)
            rows.append((cid, other_id, "tag", None, _now()))
            if len(rows) >= TAG_LINK_LIMIT:
                break
    return store.add_links(rows)


def like_contains(term: str) -> str:
    """``LIKE`` pattern that treats ``%``, ``_`` and ``\\`` in ``term`` as text."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _node(cid: str, doc: str, meta_json: str) -> dict:
    meta = json.loads(meta_json) if meta_json else {}
    text = doc or ""
    if len(text) > 240:
        text = text[:240] + "..."
    return {
        "id": cid,
        "text": text,
        "metadata": {"agent": meta.get("agent"), "source": meta.get("source")},
    }


def _edge(from_id, to_id, typ, score) -> dict:
    return {
        "from_id": from_id,
        "to_id": to_id,
        "type": typ,
        "score": None if score is None else float(score),
    }


def _materialize(store, node_ids: list[str], edges: list[dict]) -> dict:
    if not node_ids:
        return {"nodes": [], "edges": []}
    marks = ",".join("?" * len(node_ids))
    found = {
        row[0]: row
        for row in store.con.execute(
            f"SELECT id, doc, meta FROM chunks WHERE id IN ({marks})", node_ids)
    }
    nodes = [_node(*found[cid]) for cid in node_ids if cid in found]
    have = {n["id"] for n in nodes}
    kept = [e for e in edges if e["from_id"] in have and e["to_id"] in have]
    return {"nodes": nodes, "edges": kept[:GRAPH_EDGE_CAP]}


def build_graph(store, filt: str | None = None, limit: int = 100) -> dict:
    """``{"nodes", "edges"}`` capped at :data:`GRAPH_NODE_CAP` nodes.

    With no filter, nodes are endpoints of the newest links. With a filter,
    nodes start as chunks whose id, text or ``agent`` contains it, plus direct
    neighbours while the cap allows. Edges always stay inside the node set.
    """
    node_limit = max(1, min(int(limit), GRAPH_NODE_CAP))
    text = (filt or "").strip()
    if text:
        return _graph_filtered(store, text, node_limit)
    return _graph_recent(store, node_limit)


def _graph_recent(store, node_limit: int) -> dict:
    rows = store.con.execute(
        """SELECT from_id, to_id, type, score FROM memory_links
           ORDER BY created_at DESC
           LIMIT ?""",
        (max(GRAPH_EDGE_CAP, node_limit),),
    ).fetchall()
    node_ids: list[str] = []
    seen: set[str] = set()
    edges: list[dict] = []
    for from_id, to_id, typ, score in rows:
        if len(edges) >= GRAPH_EDGE_CAP:
            break
        extra = [i for i in (from_id, to_id) if i not in seen]
        if len(seen) + len(extra) > node_limit:
            if len(seen) >= node_limit:
                break
            continue
        for i in extra:
            seen.add(i)
            node_ids.append(i)
        edges.append(_edge(from_id, to_id, typ, score))
    return _materialize(store, node_ids, edges)


def _graph_filtered(store, text: str, node_limit: int) -> dict:
    like = like_contains(text)
    seeds = [
        row[0] for row in store.con.execute(
            """SELECT id FROM chunks
               WHERE id LIKE ? ESCAPE '\\'
                  OR doc LIKE ? ESCAPE '\\'
                  OR IFNULL(json_extract(meta, '$.agent'), '') LIKE ? ESCAPE '\\'
               ORDER BY rowid DESC
               LIMIT ?""",
            (like, like, like, node_limit),
        )
    ]
    if not seeds:
        return {"nodes": [], "edges": []}
    marks = ",".join("?" * len(seeds))
    rows = store.con.execute(
        f"""SELECT from_id, to_id, type, score FROM memory_links
            WHERE from_id IN ({marks}) OR to_id IN ({marks})
            ORDER BY created_at DESC
            LIMIT ?""",
        (*seeds, *seeds, GRAPH_EDGE_CAP * 2),
    ).fetchall()
    node_ids = list(seeds)
    seen = set(seeds)
    edges: list[dict] = []
    for from_id, to_id, typ, score in rows:
        if len(edges) >= GRAPH_EDGE_CAP:
            break
        extra = [i for i in (from_id, to_id) if i not in seen]
        if len(seen) + len(extra) > node_limit:
            continue
        for i in extra:
            seen.add(i)
            node_ids.append(i)
        edges.append(_edge(from_id, to_id, typ, score))
    return _materialize(store, node_ids, edges)
