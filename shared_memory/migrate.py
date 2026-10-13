#!/usr/bin/env python3
"""Copy a Chroma collection into the SQLite backend (shared-memory 0.2.0+).

By default the stored Chroma vectors are reused, so nothing is re-embedded and
no API is called: they were produced by the model in your current
embedding.env. Pass --reembed when you are switching models at the same time
(for example to Voyage); then every chunk is embedded again in batches.

The Chroma directory is only read, never changed. After checking the result,
set SHARED_MEMORY_BACKEND=sqlite (or remove chroma_data) and restart the server.

    shared-memory-migrate           # reuse vectors
    shared-memory-migrate--reembed  # recompute with current embedder
"""
from __future__ import annotations

import argparse

from .embedder import ENV_PATH, Embedder, _load_env, resolve_model
from .store import DEFAULT_COLLECTION, DEFAULT_DATA_DIR, DEFAULT_SQLITE_PATH
from .store_sqlite import Store

PAGE = 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--db", default=str(DEFAULT_SQLITE_PATH))
    ap.add_argument("--reembed", action="store_true", help="recompute vectors with the current embedder")
    a = ap.parse_args()

    import chromadb  # optional dependency: pip install "shared-memory[chroma]"

    col = chromadb.PersistentClient(a.chroma_dir).get_collection(a.collection)
    store = Store(a.db, "/".join(resolve_model(_load_env(ENV_PATH))))
    emb = Embedder() if a.reembed else None
    total, off = col.count(), 0
    while off < total:
        include = ["documents", "metadatas"] + ([] if a.reembed else ["embeddings"])
        r = col.get(include=include, limit=PAGE, offset=off)
        if not r["ids"]:
            break
        vecs = emb.embed_documents(r["documents"]) if emb else r["embeddings"]
        store.upsert(r["ids"], r["documents"], r["metadatas"], vecs)
        off += len(r["ids"])
        print(f"  {off}/{total}", flush=True)
    print(f"done: {store.count()} chunks in {a.db} (model {store.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
