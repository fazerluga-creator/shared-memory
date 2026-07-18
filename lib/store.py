"""ChromaDB persistent-client helper shared by indexers and the MCP server."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embedder import ENV_PATH, Embedder, _load_env

LOG = logging.getLogger(__name__)

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
    return client.get_or_create_collection(
        name=name,
        metadata=metadata,
    )


def existing_session_ids(collection, agent: str | None = None) -> set[str]:
    """Return set of session_id values already present in the collection.

    Uses collection.get() with a where filter. For large collections this is
    still cheap because we only fetch metadatas, not documents or embeddings.
    """
    where: dict[str, Any] | None = None
    if agent:
        where = {"agent": agent}
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
