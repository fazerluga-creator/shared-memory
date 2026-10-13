"""Tests for shared_memory/embedder.py that don't require downloading a model."""
from __future__ import annotations

import pytest
from conftest import ROOT  # noqa: F401  (ensures repo root on sys.path)

from shared_memory.embedder import (
    Embedder,
    EmbeddingConfigError,
    _l2_normalize,
    _load_env,
    _providers_for_device,
)


def test_load_env_missing_file_returns_empty(tmp_path):
    assert _load_env(tmp_path / "absent.env") == {}


def test_load_env_parses_and_strips(tmp_path):
    p = tmp_path / "e.env"
    p.write_text(
        "# comment\n"
        "\n"
        "KEY=value\n"
        'QUOTED="hello"\n'
        "SPACED =  padded  \n"
        "NOEQUALS\n",
        encoding="utf-8",
    )
    env = _load_env(p)
    assert env == {"KEY": "value", "QUOTED": "hello", "SPACED": "padded"}


def test_l2_normalize():
    out = _l2_normalize([3.0, 4.0])
    assert out == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_unchanged():
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_providers_for_device():
    assert _providers_for_device("cpu") == ["CPUExecutionProvider"]
    assert _providers_for_device("cuda") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert _providers_for_device("gpu") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert _providers_for_device("weird") is None


def test_unsupported_provider_raises(tmp_path):
    env = tmp_path / "e.env"
    env.write_text("EMBEDDING_PROVIDER=bogus\n", encoding="utf-8")
    with pytest.raises(EmbeddingConfigError, match="unsupported EMBEDDING_PROVIDER"):
        Embedder(env_path=env)


def test_matryoshka_dim_cannot_exceed_dim(tmp_path):
    env = tmp_path / "e.env"
    env.write_text(
        "EMBEDDING_DIM=384\nEMBEDDING_MATRYOSHKA_DIM=512\n", encoding="utf-8"
    )
    with pytest.raises(EmbeddingConfigError, match="cannot exceed"):
        Embedder(env_path=env)


def test_openai_compatible_requires_credentials(tmp_path):
    env = tmp_path / "e.env"
    env.write_text("EMBEDDING_PROVIDER=openai_compatible\n", encoding="utf-8")
    with pytest.raises(EmbeddingConfigError, match="not set"):
        Embedder(env_path=env)


def _bare_embedder(dim: int, matryoshka: int | None) -> Embedder:
    """Construct an Embedder without running __init__ (no model download)."""
    e = Embedder.__new__(Embedder)
    e.dim = dim
    e.matryoshka_dim = matryoshka
    e.final_dim = matryoshka or dim
    return e


def test_finalize_vector_passthrough():
    e = _bare_embedder(4, None)
    assert e._finalize_vector([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]


def test_finalize_vector_dim_mismatch_raises():
    e = _bare_embedder(4, None)
    with pytest.raises(RuntimeError, match="dim mismatch"):
        e._finalize_vector([1.0, 2.0])


def test_finalize_vector_matryoshka_truncates_and_normalizes():
    e = _bare_embedder(4, 2)
    out = e._finalize_vector([3.0, 4.0, 100.0, 100.0])
    assert out == pytest.approx([0.6, 0.8])
