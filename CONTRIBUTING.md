# Contributing

Thanks for your interest! Contributions of all sizes are welcome — from typo
fixes to new indexer adapters.

## Getting started

```bash
git clone https://github.com/fazerluga-creator/shared-memory.git
cd shared-memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest ruff
```

## Running checks

```bash
ruff check .    # lint
pytest          # tests (no model download needed)
```

Both run in CI on every PR; please make sure they pass locally first.

## What to work on

- Check the [open issues](https://github.com/fazerluga-creator/shared-memory/issues) —
  ones labeled `good first issue` are self-contained starters.
- New **indexer adapters** (chat logs, issue trackers, docs) are especially
  welcome: copy `examples/index_markdown.py` as a template, keep the id scheme
  deterministic, and always set `agent`, `source`, and an ISO `timestamp` in
  chunk metadata.

## Guidelines

- Keep the core small: `lib/` and `mcp-server/` should stay free of
  source-specific logic — that belongs in adapters.
- The default configuration must stay **local-only** (no network calls with an
  empty config).
- Add a test for any behavior change in chunking, filtering, or config parsing.

## Reporting bugs

Open an issue with your Python version, OS, the command you ran, and the full
traceback. If it's a search-quality problem, include the query, the expected
chunk, and what came back instead.
