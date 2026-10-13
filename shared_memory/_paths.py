"""Where configuration and data live.

From a source checkout (``pip install -e .`` or running from the repo) everything
stays next to the code, as in 0.1/0.2: ``./embedding.env``, ``./memory.sqlite``,
``./chroma_data``. From a regular install the package lives in site-packages,
which is no place for user data, so the defaults move to
``$XDG_DATA_HOME/shared-memory`` (``~/.local/share/shared-memory``).
``SHARED_MEMORY_HOME`` overrides both; the per-file variables
(``SHARED_MEMORY_ENV``, ``SHARED_MEMORY_DB``, ``SHARED_MEMORY_DATA_DIR``) still
win over everything.
"""
from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def source_checkout() -> Path | None:
    """The repo root when running from a checkout, else None."""
    root = _PKG.parent
    return root if (root / "pyproject.toml").exists() and (root / ".git").exists() else None


def home() -> Path:
    if os.environ.get("SHARED_MEMORY_HOME"):
        return Path(os.environ["SHARED_MEMORY_HOME"]).expanduser()
    checkout = source_checkout()
    if checkout:
        return checkout
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "shared-memory"
