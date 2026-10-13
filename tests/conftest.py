"""Shared test fixtures and module loaders.

Some tests load a private copy of a module by file path so module-level state
(singletons, BACKEND) does not leak between test files.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_module(rel_path: str, name: str):
    """A fresh, independent copy of a package module (own globals per test file).

    Loaded as ``shared_memory.<name>`` so its relative imports resolve.
    """
    import shared_memory  # noqa: F401  parent package must be importable

    name = f"shared_memory.{name}"
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
