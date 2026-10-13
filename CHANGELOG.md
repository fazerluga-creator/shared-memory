# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (while on 0.x, a minor version may
contain breaking changes; they are listed under **Changed**).

## [0.3.0] - 2026-10-13

### Added
- Console commands: `shared-memory-mcp` (server; `--transport http`,
  `--host`, `--port`), `shared-memory-index` (markdown indexer) and
  `shared-memory-migrate` (Chroma → SQLite). No more full paths in client configs.
- Install straight from GitHub: `pip install "git+https://github.com/fazerluga-creator/shared-memory.git@v0.3.0"`.
- README: "Agents without access to the server: a file bus" — the shared-folder
  convention for agents that cannot reach the MCP server.
- `SHARED_MEMORY_HOME` to move config and data together.

### Changed
- **The package is renamed from `lib` to `shared_memory`** (`lib` was too
  generic and could collide with other installed packages). Imports change from
  `lib.store` to `shared_memory.store`; see "Upgrading from 0.2.x" in the README.
- The server, the markdown indexer and the migration script moved into the
  package (`shared_memory/server.py`, `index_markdown.py`, `migrate.py`); the
  `examples/` folder is gone.
- Config and data of a regular (non-editable) install live in
  `~/.local/share/shared-memory/` instead of next to the installed code. A
  source checkout keeps using the repo folder, as before.

### Deprecated
- `mcp-server/server.py` is a thin shim kept for 0.1/0.2 client configs; it
  will be removed in 1.0.

## [0.2.0] - 2026-10-13

### Added
- **SQLite backend** (`lib/store_sqlite.py`), now the default: chunks, an FTS5
  full-text index and vectors in one file, behind the same collection API the
  server and indexers already use. No vector database service is needed.
- **Hybrid ranking** on SQLite: cosine similarity weighted by `metadata.kind`
  (`note` 1.0, `doc` 0.97, `transcript` 0.90), a bonus for exact identifiers
  from the query found verbatim in a chunk, and a lower weight for chunks
  marked `metadata.superseded`. Falls back to BM25 when no query vector is
  available. On a 32-question benchmark: 16/32 → 30/32 right file in top 3.
- **Voyage AI embedding provider** (`EMBEDDING_PROVIDER=voyage`), with
  query/document input types and batching by count and estimated tokens.
- **`memory_update`** (edit a chunk in place, id kept) and **`memory_forget`**
  (delete by ids or filter; dry run by default, `expect_count` required for
  filter deletes) MCP tools.
- `memory_search` results now include the chunk `id`.
- `memory_stats` reports the active `backend`.
- `examples/migrate_chroma_to_sqlite.py`: copy a Chroma store into SQLite,
  reusing stored vectors (no API calls) or re-embedding with `--reembed`.
- `--kind` option for `examples/index_markdown.py`.
- README sections: Security, Search quality, Upgrading from 0.1.x, an example
  search result.

### Changed
- **Default backend is SQLite.** An existing `chroma_data` directory without a
  `memory.sqlite` keeps the Chroma backend (with a warning), so upgrading never
  hides existing memories. Choose explicitly with `SHARED_MEMORY_BACKEND`.
- `chromadb` is now an optional dependency: `pip install "shared-memory[chroma]"`.
  `numpy` is now a direct dependency.
- On SQLite, `since` is applied in SQL before ranking. Previously it filtered
  an over-fetched candidate list, so on a large store a narrow time window
  could return fewer results than exist.
- Timestamps are normalized to UTC ISO on write (SQLite backend), so `Z`,
  offsets, naive and unix-epoch inputs compare correctly.

### Security
- Documented that the MCP server has no authentication and must not be exposed
  beyond localhost or a trusted network; recommended file permissions for the
  database and a note on sending text to remote embedding providers.

## [0.1.0] - 2026-07-25

First public release.

- MCP server with `memory_search`, `memory_add`, `memory_stats` over a
  persistent ChromaDB collection.
- Embeddings: local fastembed (default) or an OpenAI-compatible endpoint.
- Generic markdown indexer (`examples/index_markdown.py`).
- Hardening of search, indexing and timestamps; tests and CI.

[0.3.0]: https://github.com/fazerluga-creator/shared-memory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/fazerluga-creator/shared-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fazerluga-creator/shared-memory/releases/tag/v0.1.0
