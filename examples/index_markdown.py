#!/usr/bin/env python3
"""Example indexer: ingest a directory of markdown files into shared memory.

This is a minimal, generic adapter meant as a starting point. It walks a
directory, splits each ``*.md`` file into chunks and upserts them into the
Chroma collection so they become searchable via the MCP ``memory_search`` tool.

Chunking:
  - Split by H2 headings (``##``). If a file has no H2s, split by paragraphs.
  - Soft-cap per chunk: 8000 chars, preferring word/newline boundaries.

Idempotency:
  - ID = ``{agent}::markdown::{posix_path}::{chunk_idx}``
  - Paths are stored as POSIX (forward slashes) for cross-OS stability.
  - The file mtime is stored in metadata. With ``--incremental`` files whose
    mtime is unchanged are skipped.

Usage:
    python examples/index_markdown.py ./notes                 # full re-index
    python examples/index_markdown.py ./notes --agent kb --incremental
    python examples/index_markdown.py ./notes --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.embedder import Embedder  # noqa: E402
from lib.store import (  # noqa: E402
    DEFAULT_COLLECTION,
    add_chunks,
    format_timestamp,
    get_client,
    get_collection,
)

MAX_CHUNK_CHARS = 8000


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )


@dataclass
class MdFile:
    path: Path
    agent: str


def _discover(root: Path, agent: str) -> list[MdFile]:
    if not root.exists():
        return []
    return [MdFile(path=p, agent=agent) for p in sorted(root.rglob("*.md"))]


def _split_by_h2(text: str) -> list[tuple[str, str]]:
    """Split markdown by ## headings.

    Returns list of (heading, body). If no H2 found, returns [('<intro>', text)].
    """
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = [("<intro>", [])]
    h2_re = re.compile(r"^##\s+(.*)$")
    for ln in lines:
        m = h2_re.match(ln)
        if m:
            sections.append((m.group(1).strip(), []))
        else:
            sections[-1][1].append(ln)
    out: list[tuple[str, str]] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            out.append((heading, body))
    return out or [("<intro>", text)]


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paras or [text]


def _soft_split(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into pieces of at most ``max_chars``, preferring word breaks."""
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text[start:end]
            # Prefer the last whitespace/newline in the window.
            cut = max(window.rfind("\n"), window.rfind(" "), window.rfind("\t"))
            if cut > max_chars // 4:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end <= start:
            # Safety: always advance to avoid infinite loops on pathological input.
            end = start + max_chars
        start = end
    return pieces or ([text[:max_chars]] if text else [])


def _path_key(path: Path) -> str:
    """Stable POSIX path string for ids and metadata across OSes."""
    return path.as_posix()


def _chunks_for(md: MdFile) -> list[tuple[str, dict]]:
    """Produce list of (chunk_text, metadata) for a single markdown file."""
    text = md.path.read_text(encoding="utf-8", errors="replace")
    mtime = md.path.stat().st_mtime
    h2_sections = _split_by_h2(text)
    path_key = _path_key(md.path)

    out: list[tuple[str, dict]] = []
    chunk_idx = 0

    base_meta = {
        "agent": md.agent,
        "source": "markdown",
        "file": md.path.name,
        "path": path_key,
        "mtime": int(mtime),
        "timestamp": format_timestamp(datetime.fromtimestamp(mtime, tz=timezone.utc)),
    }

    for heading, body in h2_sections:
        # If section is small, emit as-is. If big, split by paragraphs then soft-cap.
        if len(body) <= MAX_CHUNK_CHARS:
            chunk_text = f"# {md.path.name} :: {heading}\n\n{body}"
            meta = {**base_meta, "heading": heading, "chunk_index": chunk_idx}
            out.append((chunk_text, meta))
            chunk_idx += 1
        else:
            for para in _split_paragraphs(body):
                for piece in _soft_split(para, MAX_CHUNK_CHARS):
                    chunk_text = f"# {md.path.name} :: {heading}\n\n{piece}"
                    meta = {**base_meta, "heading": heading, "chunk_index": chunk_idx}
                    out.append((chunk_text, meta))
                    chunk_idx += 1
    return out


def _existing_mtime_by_path(collection, agent: str) -> dict[str, int]:
    """Return {path: max_mtime_seen} for already-indexed files of given agent."""
    out: dict[str, int] = {}
    try:
        got = collection.get(
            where={"$and": [{"agent": {"$eq": agent}}, {"source": {"$eq": "markdown"}}]},
            include=["metadatas"],
        )
    except Exception as exc:
        logging.debug("mtime lookup failed: %s", exc)
        return out
    for m in got.get("metadatas") or []:
        if not m:
            continue
        p = m.get("path")
        mt = m.get("mtime")
        if not p or mt is None:
            continue
        try:
            mt = int(mt)
        except (ValueError, TypeError):
            continue
        if mt > out.get(p, 0):
            out[p] = mt
    return out


def _existing_ids_for_path(collection, agent: str, path: str) -> list[str]:
    """Return ids already indexed for a markdown file path."""
    try:
        got = collection.get(
            where={
                "$and": [
                    {"agent": {"$eq": agent}},
                    {"source": {"$eq": "markdown"}},
                    {"path": {"$eq": path}},
                ]
            },
            include=[],
        )
    except Exception as exc:
        logging.debug("id lookup failed for %s: %s", path, exc)
        return []
    return list(got.get("ids") or [])


def run(
    root: Path,
    agent: str,
    collection_name: str = DEFAULT_COLLECTION,
    incremental: bool = False,
    dry_run: bool = False,
    batch: int = 32,
) -> dict:
    client = get_client()
    collection = get_collection(client, name=collection_name)
    embedder = Embedder(batch_size=batch)

    files = _discover(root, agent)
    logging.info("discovered %d markdown files under %s", len(files), root)

    existing: dict[str, int] = {}
    if incremental:
        existing = _existing_mtime_by_path(collection, agent)

    stats = {
        "files_total": len(files),
        "files_indexed": 0,
        "files_skipped_unchanged": 0,
        "files_failed": 0,
        "chunks_added": 0,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    for md in files:
        try:
            cur_mtime = int(md.path.stat().st_mtime)
            path_key = _path_key(md.path)
            seen_mtime = max(existing.get(path_key, -1), existing.get(str(md.path), -1))
            if incremental and seen_mtime >= cur_mtime:
                stats["files_skipped_unchanged"] += 1
                continue
            chunks = _chunks_for(md)
            if not chunks:
                continue
            ids, texts, metas = [], [], []
            for idx, (txt, meta) in enumerate(chunks):
                ids.append(f"{md.agent}::markdown::{path_key}::{idx:03d}")
                texts.append(txt)
                metas.append(meta)
            if not dry_run:
                add_chunks(collection, embedder, ids, texts, metas, upsert=True)
                stale_ids: list[str] = []
                for p in {path_key, str(md.path)}:
                    for cid in _existing_ids_for_path(collection, agent, p):
                        if cid not in ids and cid not in stale_ids:
                            stale_ids.append(cid)
                if stale_ids:
                    collection.delete(ids=stale_ids)
            stats["chunks_added"] += len(chunks)
            stats["files_indexed"] += 1
        except Exception as exc:  # noqa: BLE001
            logging.exception("failed to index %s: %s", md.path, exc)
            stats["files_failed"] += 1

    stats["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
    logging.info("done: %s", stats)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="directory to index (walked recursively for *.md)")
    ap.add_argument("--agent", default="notes", help="agent label stored in metadata.agent")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--incremental", action="store_true", help="skip files unchanged by mtime")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)
    stats = run(
        root=Path(args.path),
        agent=args.agent,
        collection_name=args.collection,
        incremental=args.incremental,
        dry_run=args.dry_run,
        batch=args.batch,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
