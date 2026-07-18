# shared-memory

[![CI](https://github.com/fazerluga-creator/shared-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/fazerluga-creator/shared-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

A portable, self-hosted **semantic memory layer for LLM agents**, exposed over
the [Model Context Protocol](https://modelcontextprotocol.io) (MCP).

Point any number of agents — Claude Code, other MCP clients, or your own
scripts — at one shared vector store and let them recall past decisions,
notes, and conversations by meaning instead of grepping files.

- **Local by default.** Embeddings run on-device via [fastembed](https://github.com/qdrant/fastembed);
  nothing leaves the host unless you opt into a remote API.
- **One store, many agents.** Every chunk carries an `agent` and `source` tag,
  so searches can be global or scoped to a single agent.
- **Three MCP tools.** `memory_search`, `memory_add`, `memory_stats` — that's
  the whole surface.
- **Bring your own content.** Ships with a generic markdown indexer; writing an
  adapter for chat logs, tickets, or docs is a short script.

## Why

Agents are stateless between sessions. The common fix is a short "always-loaded"
memory file, but it can't scale — you can't paste everything you've ever decided
into every prompt.

shared-memory is the **second layer**: an unbounded store you search on demand.

- **Layer A** — a short, hand-curated memory file loaded into every prompt (you
  keep whatever you already use for this).
- **Layer B (this project)** — unbounded, searched semantically. "What did we
  decide about X?" returns the relevant past chunk instead of a file dump.

## Architecture

```
                 ┌────────────────┐
   agent A ─┐    │  MCP server    │   memory_search
   agent B ─┼──▶ │  (FastMCP)     │   memory_add     ──▶  ChromaDB (persistent)
   scripts ─┘    │                │   memory_stats          + local embeddings
                 └────────────────┘
        ▲
        │ indexers / adapters
   your content (markdown, sessions, docs, …)
```

```
shared-memory/
├── lib/
│   ├── embedder.py     # provider-aware embeddings (local fastembed | OpenAI-compatible)
│   └── store.py        # ChromaDB persistent-client helpers
├── mcp-server/
│   └── server.py       # FastMCP server: memory_search, memory_add, memory_stats
├── examples/
│   └── index_markdown.py   # generic adapter: index a folder of *.md
├── embedding.env.example   # copy to embedding.env to customize (optional)
└── requirements.txt
```

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/fazerluga-creator/shared-memory.git
cd shared-memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optionally copy the config and tweak it (defaults are fine to start):

```bash
cp embedding.env.example embedding.env
```

The first run downloads the local embedding model (a few hundred MB) once.

## Quick start

Index some markdown and search it:

```bash
# 1. ingest a folder of notes
python examples/index_markdown.py ./notes --agent notes

# 2. run the MCP server (stdio transport)
python mcp-server/server.py
```

Or use the library directly:

```python
from lib.embedder import Embedder
from lib.store import get_client, get_collection

col = get_collection(get_client())
emb = Embedder()

col.add(
    ids=["note::1"],
    embeddings=[emb.embed_document_one("Ship the beta on Friday, feature-flag the new UI.")],
    documents=["Ship the beta on Friday, feature-flag the new UI."],
    metadatas=[{"agent": "notes", "source": "manual"}],
)

res = col.query(query_embeddings=[emb.embed_one("when is the beta?")], n_results=3)
print(res["documents"])
```

## MCP tools

| Tool | Signature | Returns |
| --- | --- | --- |
| `memory_search` | `(query, top_k=5, agent_filter=None, source_filter=None, since=None)` | `[{text, metadata, score}, ...]` sorted by relevance |
| `memory_add` | `(text, metadata=None)` | id of the added chunk |
| `memory_stats` | `()` | `{total, by_agent, by_source, collection}` |

Filters:
- `agent_filter` — restrict to a single `metadata.agent` value.
- `source_filter` — restrict to a `metadata.source` value (e.g. `markdown`, `sessions`).
- `since` — ISO timestamp lower bound on `metadata.timestamp`.

### Connect to Claude Code / any MCP client

Add the server to your client config (paths are examples):

```json
{
  "mcpServers": {
    "shared-memory": {
      "command": "/path/to/shared-memory/.venv/bin/python",
      "args": ["/path/to/shared-memory/mcp-server/server.py"]
    }
  }
}
```

## Configuration

All settings are read from `embedding.env` (see `embedding.env.example`). Paths
are overridable by environment variable:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SHARED_MEMORY_ENV` | `./embedding.env` | location of the env file |
| `SHARED_MEMORY_DATA_DIR` | `./chroma_data` | ChromaDB persistence directory |
| `EMBEDDING_PROVIDER` | `local` | `local` (fastembed) or `openai_compatible` |
| `EMBEDDING_MODEL` | MiniLM multilingual | embedding model id |
| `EMBEDDING_COLLECTION` | `shared_memory` | Chroma collection name |

To route embeddings through a remote OpenAI-compatible `/v1/embeddings`
endpoint, set `EMBEDDING_PROVIDER=openai_compatible` and fill in
`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`.

## Writing your own indexer

`examples/index_markdown.py` is the template. An adapter needs to:

1. discover source items,
2. turn each into `(text, metadata)` chunks — always set `agent` and `source`,
   and an ISO `timestamp` if you want `since` filtering,
3. call `add_chunks(collection, embedder, ids, texts, metadatas, upsert=True)`.

Use a stable, deterministic id scheme so re-runs upsert instead of duplicating.

## Design notes

- **Embeddings are computed by us, not by Chroma.** Vectors are passed
  explicitly on `add`/`query`, so the store stays decoupled from embedder
  availability.
- **Chunking (markdown):** split by `##` headings; fall back to paragraphs;
  soft-cap 8000 chars.
- **Timestamps** are ISO-UTC strings throughout (ISO sorts lexicographically,
  which is what the `since` filter relies on).
- **Search-time hygiene:** results are de-duplicated by content hash and short
  known-error strings are dropped.

## Roadmap

- [ ] TTL / forgetting policy for high-volume sources.
- [ ] Optional PII filter at index time.
- [ ] Session-end hooks to auto-index new content.
- [ ] More example adapters (chat logs, issue trackers).

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). New indexer
adapters (chat logs, issue trackers, docs) are a great place to start.

## License

[MIT](LICENSE).
