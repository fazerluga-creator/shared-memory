# shared-memory

[![CI](https://github.com/fazerluga-creator/shared-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/fazerluga-creator/shared-memory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

A portable, self-hosted **semantic memory layer for LLM agents**, exposed over
the [Model Context Protocol](https://modelcontextprotocol.io) (MCP).

Point any number of agents — Claude Code, other MCP clients, or your own
scripts — at one shared store and let them recall past decisions,
notes, and conversations by meaning instead of grepping files.

- **One SQLite file.** Text, a full-text index and vectors live together; no
  database service to run. ChromaDB is still available as an option.
- **Hybrid ranking.** Vector similarity, plus exact matching of identifiers
  (order numbers, `ABC-123` ids) that vectors alone miss, plus weights that put
  distilled notes above raw transcripts. See [Search quality](#search-quality).
- **Local by default.** Embeddings run on-device via [fastembed](https://github.com/qdrant/fastembed);
  nothing leaves the host unless you opt into a remote API (OpenAI-compatible or Voyage AI).
- **One store, many agents.** Every chunk carries an `agent` and `source` tag,
  so searches can be global or scoped to a single agent.
- **Five MCP tools.** `memory_search`, `memory_add`, `memory_update`,
  `memory_forget`, `memory_stats`.
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
   agent A ─┐    │  MCP server    │   memory_search      ──▶  memory.sqlite
   agent B ─┼──▶ │  (FastMCP)     │   memory_add / update      chunks + FTS5 + vectors
   scripts ─┘    │                │   memory_forget / stats    (or ChromaDB, optional)
                 └────────────────┘
        ▲
        │ indexers / adapters
   your content (markdown, sessions, docs, …)
```

```
shared-memory/
├── shared_memory/
│   ├── server.py         # FastMCP server, the five memory_* tools  → shared-memory-mcp
│   ├── index_markdown.py # generic adapter: index a folder of *.md  → shared-memory-index
│   ├── migrate.py        # copy a Chroma store into SQLite          → shared-memory-migrate
│   ├── embedder.py       # embeddings: local fastembed | OpenAI-compatible | Voyage AI
│   ├── store.py          # backend selection, shared helpers
│   └── store_sqlite.py   # SQLite backend: storage, hybrid ranking, Chroma-style API
├── mcp-server/server.py  # compatibility shim for 0.1/0.2 configs (removed in 1.0)
├── embedding.env.example # copy to embedding.env to customize (optional)
└── requirements.txt
```

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/fazerluga-creator/shared-memory.git
cd shared-memory
python -m venv .venv && source .venv/bin/activate
pip install -e .              # SQLite backend (default)
pip install -e ".[chroma]"    # also the ChromaDB backend / migration
```

Or without cloning:

```bash
pip install "git+https://github.com/fazerluga-creator/shared-memory.git@v0.3.0"
```

This installs three commands: `shared-memory-mcp` (the server),
`shared-memory-index` (markdown indexer) and `shared-memory-migrate`.

**Where data lives.** From a clone (`pip install -e .`) the config and data stay
in the repo folder: `embedding.env`, `memory.sqlite`. From a regular install
they live in `~/.local/share/shared-memory/` (`$XDG_DATA_HOME/shared-memory`).
`SHARED_MEMORY_HOME` moves both; `SHARED_MEMORY_ENV` / `SHARED_MEMORY_DB` set
single files.
```

Optionally copy the config and tweak it (defaults are fine to start):

```bash
cp embedding.env.example embedding.env
```

The first run downloads the local embedding model (a few hundred MB) once.
With a remote provider (`voyage`, `openai_compatible`) nothing is downloaded.

## Quick start

Index some markdown and search it:

```bash
# 1. ingest a folder of notes
shared-memory-index ./notes --agent notes

# 2. run the MCP server (stdio transport)
shared-memory-mcp
# or over HTTP on 127.0.0.1:8000 (read the Security section first)
shared-memory-mcp --transport http
```

Or use the library directly:

```python
from shared_memory import Embedder, get_client, get_collection

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
| `memory_search` | `(query, top_k=5, agent_filter=None, source_filter=None, since=None)` | `[{id, text, metadata, score}, ...]` sorted by relevance |
| `memory_add` | `(text, metadata=None)` | id of the added chunk |
| `memory_update` | `(id, text=None, metadata=None, replace_metadata=False)` | `{id, changed, metadata}`; the id is kept, unknown ids raise |
| `memory_forget` | `(ids=None, agent_filter=None, source_filter=None, dry_run=True, expect_count=None)` | `{matched, deleted, dry_run, sample}` |
| `memory_stats` | `()` | `{total, by_agent, by_source, collection, backend}` |

`memory_forget` is a dry run unless you pass `dry_run=False`; deleting by filter
also requires `expect_count` equal to the number of matches, and a call with no
ids and no filters is refused.

Example `memory_search` result:

```json
[
  {
    "id": "notes::markdown::decisions.md::003",
    "text": "decisions.md › Release: ship the beta on Friday, feature-flag the new UI.",
    "metadata": {"agent": "notes", "source": "markdown", "kind": "note",
                 "timestamp": "2026-10-01T09:12:00+00:00"},
    "score": 0.7134
  }
]
```

Filters:
- `agent_filter` — restrict to a single `metadata.agent` value.
- `source_filter` — restrict to a `metadata.source` value (e.g. `markdown`, `sessions`).
- `since` — ISO timestamp lower bound on `metadata.timestamp`. On the SQLite
  backend it is applied before ranking, so a narrow window on a large store
  still returns the matches inside it.

### Connect to Claude Code / any MCP client

Add the server to your client config (paths are examples):

```json
{
  "mcpServers": {
    "shared-memory": {
      "command": "/path/to/shared-memory/.venv/bin/shared-memory-mcp"
    }
  }
}
```

## Security

**The server has no authentication.** Anyone who can reach it can read every
memory, add to it, rewrite it and delete it. Run it over stdio, or bind the
HTTP transport to `127.0.0.1` (FastMCP's default), and never expose it to the
internet or to a network you do not fully control. If several machines need it,
put it behind a VPN or an authenticating reverse proxy.

Memories tend to collect secrets that were pasted into chats. Treat the
database file like a password store: restrictive file permissions (`chmod 600`)
and encrypted backups. With a remote embedding provider every chunk is sent to
that provider; filter secrets out in your indexer before that if it matters.

## Configuration

All settings are read from `embedding.env` (see `embedding.env.example`). Paths
are overridable by environment variable:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SHARED_MEMORY_ENV` | `./embedding.env` | location of the env file |
| `SHARED_MEMORY_BACKEND` | `sqlite` | `sqlite` or `chroma` (see below) |
| `SHARED_MEMORY_DB` | `./memory.sqlite` | SQLite database file |
| `SHARED_MEMORY_DATA_DIR` | `./chroma_data` | ChromaDB persistence directory |
| `EMBEDDING_PROVIDER` | `local` | `local` (fastembed), `openai_compatible` or `voyage` |
| `EMBEDDING_MODEL` | MiniLM multilingual (`voyage-4` for voyage) | embedding model id |
| `EMBEDDING_COLLECTION` | `shared_memory` | collection name (display only on SQLite) |

To route embeddings through a remote OpenAI-compatible `/v1/embeddings`
endpoint, set `EMBEDDING_PROVIDER=openai_compatible` and fill in
`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`. For Voyage AI set
`EMBEDDING_PROVIDER=voyage` and `EMBEDDING_API_KEY` (or `VOYAGE_API_KEY`).

Vectors are tagged with the model that produced them and search ignores vectors
of any other model, so changing the provider or model means re-indexing
(`shared-memory-migrate --reembed` does this for a Chroma store).

### Upgrading from 0.1.x (Chroma)

If an existing `chroma_data` directory is found and no `memory.sqlite`, 0.2.0
keeps using Chroma and logs a warning, so nothing disappears after an upgrade.
To move to SQLite:

```bash
pip install -e ".[chroma]"
shared-memory-migrate   # reuses stored vectors, no API calls
# check memory_stats / a few searches, then set SHARED_MEMORY_BACKEND=sqlite
```

## Search quality

On the SQLite backend a chunk's score is its cosine similarity times a weight
for `metadata.kind` — `note` 1.0, `doc` 0.97, `transcript` 0.90, anything else
1.0 — plus 0.3 when the query contains an exact token (a number of 4+ digits, a
word with a digit, `ABC-123`) that the chunk contains verbatim. Chunks with
`metadata.superseded: true` are multiplied by 0.9: still findable, just below
the current version. Without a query vector (the embedding API is down) search
falls back to BM25 full-text ranking instead of failing.

Measured on a private 32-question benchmark over a real 25k-chunk multi-agent
memory (right file among the top 3 results):

| setup | top-3 |
| --- | --- |
| ChromaDB + local MiniLM (0.1.x) | 16 / 32 |
| Voyage `voyage-4`, vectors only | 26 / 32 |
| Voyage + kind weights + exact tokens + superseded (0.2.0) | 30 / 32 |

The numbers come from one corpus; treat them as a direction, not a guarantee.

### Upgrading from 0.2.x

The `lib` package is now `shared_memory`: change `from lib.store import ...` to
`from shared_memory.store import ...` (or `from shared_memory import ...`).
Client configs that run `mcp-server/server.py` keep working through a shim;
switch them to the `shared-memory-mcp` command when convenient. Data files in
a clone stay where they were.

## Agents without access to the server: a file bus

The MCP server is meant to stay on a private machine (see Security). Agents
that cannot reach it, such as a chat assistant on a phone or a hosted agent
with only a Google Drive or Dropbox connector, can still share work through a
plain shared folder used as a message bus. One setup that works in practice:

```
Agent Bus/
├── 00 Welcome          # what this is, which files to read first
├── 01 Rules            # task and answer formats, what never goes here (secrets, personal data)
├── 01_Inbox            # one file per task: task_id, from, to, priority, goal, expected result
├── 02_Outbox           # one file per answer: task_id, agent, status done|blocked|needs_review
├── 03_Shared Memory    # confirmed facts and decisions only, one file per decision
└── 05_Logs             # traces, daily snapshots
```

Rules that keep it sane: every task and answer carries a `task_id`; silence
from another agent is never treated as agreement; a record is a proposal, not
permission to act on anything irreversible; secrets stay on the server and the
bus carries only references. An agent that does have server access (for
example Claude Code on the same machine) bridges the two: it reads the inbox,
does the work with `memory_search` / `memory_add`, and writes confirmed facts
back as files. shared-memory does not ship a bus; the folder layout above is a
convention, not code.

## Writing your own indexer

`shared_memory/index_markdown.py` is the template. An adapter needs to:

1. discover source items,
2. turn each into `(text, metadata)` chunks — always set `agent` and `source`,
   an ISO `timestamp` if you want `since` filtering, and `kind`
   (`note` / `doc` / `transcript`) if you want the ranking weights,
3. call `add_chunks(collection, embedder, ids, texts, metadatas, upsert=True)`.

Use a stable, deterministic id scheme so re-runs upsert instead of duplicating.

## Design notes

- **Embeddings are computed by us, not by the store.** Vectors are passed
  explicitly on `add`/`query`, so the store stays decoupled from embedder
  availability.
- **No vector index on SQLite.** Search is a brute-force cosine over an
  in-memory numpy matrix: 25k chunks x 1024 dims is ~100 MB and a few
  milliseconds per query, with nothing to corrupt or rebuild. The matrix is
  reloaded when another process writes to the file.
- **Chunking (markdown):** split by `##` headings; fall back to paragraphs;
  soft-cap 8000 chars at word/newline boundaries. Paths in ids/metadata use
  POSIX form for cross-OS stability.
- **Timestamps** are ISO-UTC strings throughout; the SQLite backend normalizes
  them on write, so `Z`, `+02:00`, naive and unix-epoch inputs compare correctly.
- **Search-time hygiene:** results are de-duplicated by content hash and short
  known-error strings are dropped. Scores are clamped to `[0, 1]`.
- **Stats:** `memory_stats` pages through metadatas to bound peak memory.

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
