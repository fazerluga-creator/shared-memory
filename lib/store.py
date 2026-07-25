"""ChromaDB persistent-client helper shared by indexers and the MCP server."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embedder import ENV_PATH, Embedder, _load_env

LOG = logging.getLogger(__name__)


def parse_timestamp(value: str) -> datetime:
    """Parse ISO-8601 or unix-epoch strings into timezone-aware UTC datetimes."""
    s = str(value).strip()
    if not s:
        raise ValueError("empty timestamp")
    if s.isdigit():
        return datetime.fromtimestamp(int(s), tz=timezone.utc)
    if s.endswith("Z"):
        s = f"{s[:-1]}+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_timestamp(dt: datetime | None = None) -> str:
    """Return a canonical UTC ISO timestamp used across the collection."""
    when = dt or datetime.now(timezone.utc)
    return when.astimezone(timezone.utc).isoformat()


DEFAULT_DATA_DIR = Path(
    os.environ.get("SHARED_MEMORY_DATA_DIR")
    or (Path(__file__).resolve().parent.parent / "chroma_data")
)
DEFAULT_COLLECTION = os.environ.get(
    "EMBEDDING_COLLECTION",
    _load_env(ENV_PATH).get("EMBEDDING_COLLECTION", "shared_memory"),
)


def get_client(data_dir: Path | str = DEFAULT_DATA_DIR) -> chromadb.api.ClientAPI:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )


def get_collection(
    client: chromadb.api.ClientAPI | None = None,
    name: str = DEFAULT_COLLECTION,
):
    """Return the collection. Embeddings are computed by us (not by Chroma).

    We pass embeddings explicitly on add() and query_embeddings on query(),
    so the collection itself does NOT need an embedding_function. That keeps
    the store decoupled from the embedder's availability.
    """
    client = client or get_client()
    try:
        emb = Embedder()
        metadata = {
            "hnsw:space": "cosine",
            "embedder": f"{emb.provider}/{emb.model}",
            "dim": str(emb.final_dim),
        }
    except Exception as exc:  # noqa: BLE001
        LOG.warning("cannot initialize embedder for collection metadata: %s", exc)
        metadata = {"hnsw:space": "cosine"}
    collection = client.get_or_create_collection(
        name=name,
        metadata=metadata,
    )
    stored_meta = collection.metadata or {}
    expected_embedder = metadata.get("embedder")
    stored_embedder = stored_meta.get("embedder")
    if stored_embedder and expected_embedder and stored_embedder != expected_embedder:
        LOG.warning(
            "collection %r embedder mismatch: stored=%s current=%s; "
            "re-index or use a new collection to avoid degraded search",
            name,
            stored_embedder,
            expected_embedder,
        )
    return collection


def existing_session_ids(collection, agent: str | None = None) -> set[str]:
    """Return set of session_id values already present in the collection.

    Intended for session/chat-log indexers that skip already-ingested sessions.
    Uses collection.get() with a where filter and only fetches metadatas.
    """
    where: dict[str, Any] | None = None
    if agent:
        where = {"agent": {"$eq": agent}}
    try:
        got = collection.get(where=where, include=["metadatas"])
    except Exception as exc:  # chroma can throw if collection empty on some versions
        LOG.debug("collection.get failed (empty?): %s", exc)
        return set()
    metas = got.get("metadatas") or []
    out: set[str] = set()
    for m in metas:
        if not m:
            continue
        sid = m.get("session_id")
        if sid:
            out.add(sid)
    return out


def add_chunks(
    collection,
    embedder: Embedder,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    *,
    upsert: bool = False,
) -> None:
    if not ids:
        return
    if not (len(ids) == len(texts) == len(metadatas)):
        raise ValueError("ids/texts/metadatas length mismatch")
    vectors = embedder.embed_documents(texts)
    kwargs = dict(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    if upsert:
        collection.upsert(**kwargs)
    else:
        collection.add(**kwargs)
