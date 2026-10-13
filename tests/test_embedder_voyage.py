"""Voyage provider: query/document input types and batching (no network)."""
from __future__ import annotations

from shared_memory.embedder import Embedder, resolve_model


class FakeSession:
    def __init__(self):
        self.calls = []
        self.headers = {}

    def post(self, url, json, timeout):
        self.calls.append(json)

        class R:
            status_code = 200

            @staticmethod
            def json():
                return {"data": [{"index": i, "embedding": [1.0] + [0.0] * 1023}
                                 for i in range(len(json["input"]))]}
        return R()


def make(tmp_path):
    e = Embedder(provider="voyage", api_key="test", env_path=tmp_path / "none.env")
    e._session = FakeSession()
    return e


def test_defaults(tmp_path):
    e = make(tmp_path)
    assert (e.model, e.final_dim, e.batch_size) == ("voyage-4", 1024, 128)
    assert e._endpoint == "https://api.voyageai.com/v1/embeddings"
    assert resolve_model({"EMBEDDING_PROVIDER": "voyage"}) == ("voyage", "voyage-4")


def test_input_type(tmp_path):
    e = make(tmp_path)
    e.embed_one("q")
    e.embed_documents(["d"])
    assert [c["input_type"] for c in e._session.calls] == ["query", "document"]


def test_batches_by_count_and_tokens(tmp_path):
    e = make(tmp_path)
    assert len(e.embed_documents(["a"] * 300)) == 300
    assert [len(c["input"]) for c in e._session.calls] == [128, 128, 44]
    e._session.calls.clear()
    e.embed_documents(["x" * 15000] * 30)  # 5001 estimated tokens each -> 19 per batch
    assert [len(c["input"]) for c in e._session.calls] == [19, 11]
