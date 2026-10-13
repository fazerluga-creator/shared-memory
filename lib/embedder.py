"""Provider-aware embedding wrapper for shared-memory.

Default provider is local fastembed, so no external API calls are made and the
stack works out of the box (the first run downloads the model, a few hundred MB).
Optional providers send text to a remote API: ``openai_compatible`` (any
``/v1/embeddings`` endpoint) and ``voyage`` (Voyage AI, query/document-aware). Configuration is read from an env file resolved in
this order:

  1. ``$SHARED_MEMORY_ENV``
  2. ``embedding.env`` at the repository root (optional)

If no env file is found, sensible local defaults are used. See
``embedding.env.example`` for the full list of keys.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import requests

LOG = logging.getLogger(__name__)


def _default_env_path() -> Path:
    override = os.environ.get("SHARED_MEMORY_ENV")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "embedding.env"


ENV_PATH = _default_env_path()
DEFAULT_BATCH = 32
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2.0

Provider = Literal["local", "openai_compatible", "voyage"]
PROVIDERS = ("local", "openai_compatible", "voyage")

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VOYAGE_URL = "https://api.voyageai.com/v1"
VOYAGE_DEFAULT_MODEL = "voyage-4"
VOYAGE_DIMS = {"voyage-4": 1024, "voyage-4-lite": 1024, "voyage-4-large": 1024}
VOYAGE_BATCH = 128            # API limit is 1000 inputs per request; stay well below
VOYAGE_BATCH_TOKENS = 100_000  # rough estimate at ~3 chars per token, under the API limit


def resolve_model(env: dict[str, str] | None = None, provider: str | None = None) -> tuple[str, str]:
    """Return ``(provider, model)`` exactly as ``Embedder`` would pick them.

    The SQLite store tags every vector with this model name and ignores vectors
    of other models, so both sides must agree without loading the model.
    """
    env = _load_env(ENV_PATH) if env is None else env
    prov = (provider or env.get("EMBEDDING_PROVIDER", "local")).strip().lower()
    default = VOYAGE_DEFAULT_MODEL if prov == "voyage" else DEFAULT_LOCAL_MODEL
    return prov, env.get("EMBEDDING_MODEL") or default

# Models that fastembed does not ship in its default registry and must be
# registered at runtime via add_custom_model(). e5 needs MEAN pooling +
# normalization and the query:/passage: prefixes (set in embedding.env).
_CUSTOM_LOCAL_MODELS: dict[str, dict] = {
    "intfloat/multilingual-e5-small": {
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
        "model_file": "onnx/model.onnx",
        "normalization": True,
    },
}


class EmbeddingConfigError(RuntimeError):
    """Raised when embedding configuration is missing or inconsistent."""


def _load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file. Missing file -> empty config."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _as_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0:
        return vec
    return [float(x) / norm for x in vec]


def _providers_for_device(device: str) -> list[str] | None:
    """Map EMBEDDING_DEVICE to ONNX Runtime execution providers for fastembed."""
    d = (device or "cpu").strip().lower()
    if d in ("cpu", ""):
        return ["CPUExecutionProvider"]
    if d in ("cuda", "gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # Unknown device string — let fastembed use its defaults.
    return None


class Embedder:
    """Embedding wrapper used by indexers and the MCP server.

    Providers:
      - local: fastembed TextEmbedding, default, no network calls after model download.
      - openai_compatible: optional /v1/embeddings fallback, disabled by default.
      - voyage: Voyage AI embeddings (EMBEDDING_API_KEY or VOYAGE_API_KEY); sends
        input_type=query/document, which measurably helps retrieval.

    ``embed()`` is document-mode for indexing. ``embed_query()`` / ``embed_one()`` are
    query-mode for searches. Task prefixes are controlled by embedding.env.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        env_path: Path = ENV_PATH,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        env = _load_env(env_path)
        prov, default_model = resolve_model(env, provider)
        self.provider: Provider = prov  # type: ignore[assignment]
        if self.provider not in PROVIDERS:
            raise EmbeddingConfigError(f"unsupported EMBEDDING_PROVIDER={self.provider!r}")

        self.model = model or default_model
        if self.provider == "voyage":
            self.dim = int(env.get("EMBEDDING_DIM") or VOYAGE_DIMS.get(self.model, 1024))
        else:
            self.dim = int(env.get("EMBEDDING_DIM", "384"))
        self.batch_size = int(batch_size or env.get("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH))
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.query_prefix = env.get("EMBEDDING_QUERY_PREFIX", "")
        self.document_prefix = env.get("EMBEDDING_DOCUMENT_PREFIX", "")
        self.matryoshka_dim = _as_int(env.get("EMBEDDING_MATRYOSHKA_DIM"), None)
        self.device = env.get("EMBEDDING_DEVICE", "cpu")
        self.threads = _as_int(env.get("EMBEDDING_THREADS"), None)

        if self.matryoshka_dim:
            if self.matryoshka_dim <= 0:
                raise EmbeddingConfigError("EMBEDDING_MATRYOSHKA_DIM must be positive")
            if self.matryoshka_dim > self.dim:
                raise EmbeddingConfigError("EMBEDDING_MATRYOSHKA_DIM cannot exceed EMBEDDING_DIM")
            self.final_dim = self.matryoshka_dim
        else:
            self.final_dim = self.dim

        self._local_model = None
        self._session: requests.Session | None = None
        self._endpoint = ""

        if self.provider == "local":
            self._init_local()
        elif self.provider == "voyage":
            key = api_key or env.get("EMBEDDING_API_KEY") or os.environ.get("VOYAGE_API_KEY")
            self._init_openai_compatible(env, base_url=base_url or VOYAGE_URL, api_key=key)
            if not batch_size:
                self.batch_size = VOYAGE_BATCH
        else:
            self._init_openai_compatible(env, base_url=base_url, api_key=api_key)

    def _init_local(self) -> None:
        from fastembed import TextEmbedding

        # Register non-registry models (e.g. e5-small) before constructing.
        custom = _CUSTOM_LOCAL_MODELS.get(self.model)
        if custom:
            try:
                registered = {m["model"] for m in TextEmbedding.list_supported_models()}
            except Exception:
                registered = set()
            if self.model not in registered:
                from fastembed.common.model_description import ModelSource, PoolingType

                try:
                    TextEmbedding.add_custom_model(
                        model=self.model,
                        pooling=PoolingType.MEAN,
                        normalization=custom["normalization"],
                        sources=ModelSource(hf=custom["hf"]),
                        dim=custom["dim"],
                        model_file=custom["model_file"],
                    )
                except Exception as exc:  # already registered in-process / race
                    LOG.debug("add_custom_model skipped for %s: %s", self.model, exc)

        kwargs: dict = {"model_name": self.model}
        # fastembed 0.8.0 accepts threads/providers on current builds, but keep
        # constructor tolerant across minor API differences.
        if self.threads:
            kwargs["threads"] = self.threads
        providers = _providers_for_device(self.device)
        if providers:
            kwargs["providers"] = providers
        try:
            self._local_model = TextEmbedding(**kwargs)
        except TypeError:
            kwargs.pop("providers", None)
            try:
                self._local_model = TextEmbedding(**kwargs)
            except TypeError:
                kwargs.pop("threads", None)
                self._local_model = TextEmbedding(**kwargs)

    def _init_openai_compatible(
        self,
        env: dict[str, str],
        *,
        base_url: str | None,
        api_key: str | None,
    ) -> None:
        self.base_url = (base_url or env.get("EMBEDDING_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or env.get("EMBEDDING_API_KEY")
        if not self.base_url or not self.api_key:
            raise EmbeddingConfigError("EMBEDDING_BASE_URL / EMBEDDING_API_KEY not set")
        self._endpoint = f"{self.base_url}/embeddings"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _post(self, payload: dict) -> dict:
        if self._session is None:
            raise RuntimeError("HTTP session is not initialized")
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self._session.post(self._endpoint, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (408, 425, 429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"HTTP {r.status_code}: {r.text[:200]}", response=r
                    )
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:500]}", response=r)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                wait = self.backoff * (2 ** (attempt - 1))
                LOG.warning(
                    "embedding attempt %d/%d failed: %s (sleeping %.1fs)",
                    attempt,
                    self.retries,
                    exc,
                    wait,
                )
                if attempt < self.retries:
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _prepare_texts(self, texts: Iterable[str], *, mode: str) -> list[str]:
        prefix = self.query_prefix if mode == "query" else self.document_prefix
        return [f"{prefix}{t if t else ' '}" for t in texts]

    def _finalize_vector(self, vec: Iterable[float]) -> list[float]:
        out = [float(x) for x in vec]
        if len(out) != self.dim:
            raise RuntimeError(f"embedding dim mismatch: expected {self.dim}, got {len(out)}")
        if self.matryoshka_dim:
            out = out[: self.matryoshka_dim]
            out = _l2_normalize(out)
        if len(out) != self.final_dim:
            raise RuntimeError(
                f"final embedding dim mismatch: expected {self.final_dim}, got {len(out)}"
            )
        return out

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if self._local_model is None:
            raise RuntimeError("local embedding model is not initialized")
        vectors = list(self._local_model.embed(texts, batch_size=self.batch_size))
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedding count mismatch: sent {len(texts)}, got {len(vectors)}")
        return [self._finalize_vector(v) for v in vectors]

    def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {"model": self.model, "input": batch}
            resp = self._post(payload)
            data = resp.get("data") or []
            if len(data) != len(batch):
                raise RuntimeError(f"embedding count mismatch: sent {len(batch)}, got {len(data)}")
            data.sort(key=lambda d: d.get("index", 0))
            for item in data:
                vec = item.get("embedding")
                if not vec:
                    raise RuntimeError("empty embedding returned")
                out.append(self._finalize_vector(vec))
        return out

    def _embed_voyage(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Batch by count and by estimated tokens; long inputs are truncated by the API."""
        out: list[list[float]] = []
        i = 0
        while i < len(texts):
            batch: list[str] = []
            est = 0
            while i < len(texts) and len(batch) < self.batch_size:
                cost = len(texts[i]) // 3 + 1
                if batch and est + cost > VOYAGE_BATCH_TOKENS:
                    break
                batch.append(texts[i])
                est += cost
                i += 1
            resp = self._post({"model": self.model, "input": batch,
                               "input_type": input_type, "truncation": True})
            data = sorted(resp.get("data") or [], key=lambda d: d.get("index", 0))
            if len(data) != len(batch):
                raise RuntimeError(f"embedding count mismatch: sent {len(batch)}, got {len(data)}")
            out.extend(self._finalize_vector(d["embedding"]) for d in data)
        return out

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        prepared = self._prepare_texts(texts, mode="document")
        if not prepared:
            return []
        if self.provider == "voyage":
            return self._embed_voyage(prepared, "document")
        if self.provider == "local":
            return self._embed_local(prepared)
        return self._embed_openai_compatible(prepared)

    def embed_query(self, texts: Iterable[str]) -> list[list[float]]:
        prepared = self._prepare_texts(texts, mode="query")
        if not prepared:
            return []
        if self.provider == "voyage":
            return self._embed_voyage(prepared, "query")
        if self.provider == "local":
            return self._embed_local(prepared)
        return self._embed_openai_compatible(prepared)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Document-mode embeddings. Used by indexers and memory_add."""
        return self.embed_documents(texts)

    def embed_one(self, text: str) -> list[float]:
        """Query-mode single embedding. Used by memory_search."""
        return self.embed_query([text])[0]

    def embed_document_one(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ChromaEmbeddingFunction:
    def __init__(self, embedder: Embedder | None = None):
        self._emb = embedder or Embedder()

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._emb.embed_documents(input)

    def name(self) -> str:
        return f"shared-memory-{self._emb.provider}-{self._emb.model}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = Embedder()
    docs = e.embed_documents(["hello world", "semantic memory search"])
    qry = e.embed_query(["search my memory"])
    print(
        f"provider={e.provider} model={e.model} vectors={len(docs)} "
        f"doc_dim={len(docs[0])} query_dim={len(qry[0])}"
    )
