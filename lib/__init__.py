"""shared-memory: a portable semantic memory layer for LLM agents."""

from .embedder import Embedder, EmbeddingConfigError
from .store import (
    add_chunks,
    existing_session_ids,
    format_timestamp,
    get_client,
    get_collection,
    parse_timestamp,
)

__all__ = [
    "Embedder",
    "EmbeddingConfigError",
    "add_chunks",
    "existing_session_ids",
    "format_timestamp",
    "get_client",
    "get_collection",
    "parse_timestamp",
]
