#!/usr/bin/env python3
"""Compatibility shim for 0.1/0.2 client configs that run ``mcp-server/server.py``.

The server moved into the package in 0.3.0; prefer the ``shared-memory-mcp``
command (or ``python -m shared_memory.server``). This file will be removed in 1.0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_memory.server import main, mcp  # noqa: E402,F401

if __name__ == "__main__":
    main()
