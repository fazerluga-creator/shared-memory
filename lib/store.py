"""Storage entry point shared by indexers and the MCP server.

Two backends behind one Chroma-style collection API:

* ``sqlite`` (default since 0.2.0) - one SQLite file with chunks, an FTS5 index
  and vectors; hybrid ranking, see ``lib/store_sqlite.py``. No extra service.
* ``chroma`` - the original persistent ChromaDB directory. Needs the optional
  ``chromadb`` dependency (``pip install "shared-memory[chroma]"``).

Choose with ``SHARED_MEMORY_BACKEND`` (environment or ``embedding.env``). When it
is unset and an existing ``chroma_data`` directory is found without a SQLite
file next to it, the chroma backend is kept so an upgrade never hides existing
memories; migrate with ``examples/migrate_chroma_to_sqlite.py``.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


_ENV = _load_env(ENV_PATH)
_ROOT = Path(__file__).resolve().parent.parent


def _setting(key: str, default: str) -> str:
    return os.environ.get(key) or _ENV.get(key) or default


DEFAULT_DATA_DIR = Path(_setting("SHARED_MEMORY_DATA_DIR", str(_ROOT / "chroma_data")))
DEFAULT_SQLITE_PATH = Path(_setting("SHARED_MEMORY_DB", str(_ROOT / "memory.sqlite")))
DEFAULT_COLLECTION = _setting("EMBEDDING_COLLECTION", "shared_memory")


def _detect_backend() -> str:
    explicit = _setting("SHARED_MEMORY_BACKEND", "")
    if explicit:
        if explicit not in ("sqlite", "chroma"):
            raise ValueError(f"SHARED_MEMORY_BACKEND must be sqlite or chroma, got {explicit!r}")
        return explicit
    legacy = DEFAULT_DATA_DIR / "chroma.sqlite3"
    if legacy.exists() and not DEFAULT_SQLITE_PATH.exists():
        LOG.warning(
            "found existing Chroma data in %s and no %s: staying on the chroma backend. "
            "Set SHARED_MEMORY_BACKEND=chroma to silence this, or migrate with "
            "examples/migrate_chroma_to_sqlite.py",
            DEFAULT_DATA_DIR, DEFAULT_SQLITE_PATH)
        return "chroma"
    return "sqlite"


BACKEND = _detect_backend()
_sqlite_collections: dict[str, Any] = {}


def get_client(data_dir: Path | str = DEFAULT_DATA_DIR):
    """Chroma client, or ``None`` on the sqlite backend (``get_collection`` handles both)."""
    if BACKEND == "sqlite":
        return None
    import chromadb
    from chromadb.config import Settings

    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )


def get_collection(client=None, name: str = DEFAULT_COLLECTION):
    """Return the collection. Embeddings are computed by us (not by Chroma).

    We pass embeddings explicitly on add() and query_embeddings on query(),
    so the collection itself does NOT need an embedding_function. That keeps
    the store decoupled from the embedder's availability.

    On the sqlite backend all collection names share one file; the name is kept
    for display only. Vectors are tagged with the embedding model, and search
    ignores vectors of another model, so switching models means re-indexing.
    """
    if BACKEND == "sqlite":
        if name not in _sqlite_collections:
            from .embedder import resolve_model
            from .store_sqlite import SqliteCollection, Store

            # same env file the Embedder reads, so vector tags and embeddings agree
            model = "/".join(resolve_model(_ENV))  # e.g. "voyage/voyage-4"
            _sqlite_collections[name] = SqliteCollection(Store(DEFAULT_SQLITE_PATH, model), name)
        return _sqlite_collections[name]
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
