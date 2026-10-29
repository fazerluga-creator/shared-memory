"""Dry-run backfill and optional OmniRoute confirmation. No cron, no network."""
from __future__ import annotations

import importlib.util

import pytest
from conftest import ROOT

from shared_memory.store_sqlite import Store


def _load():
    spec = importlib.util.spec_from_file_location(
        "link_backfill", ROOT / "scripts" / "link_backfill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


backfill = _load()


def _store(tmp_path):
    store = Store(tmp_path / "m.sqlite", "toy")
    store.upsert(
        ["a", "b", "c"],
        ["kettle zebra", "kettle zebra too", "giraffe alone"],
        [{}, {}, {}],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    return store


def _count(store) -> int:
    return store.con.execute("SELECT count(*) FROM memory_links").fetchone()[0]


def test_source_has_endpoint_and_no_hardcoded_key():
    text = (ROOT / "scripts" / "link_backfill.py").read_text(encoding="utf-8")
    assert "http://127.0.0.1:20128/v1/chat/completions" in text
    assert "sk-" not in text
    assert "pa-" not in text
    assert "crontab" in text


def test_dry_run_reports_and_apply_is_idempotent(tmp_path):
    store = _store(tmp_path)
    dry = backfill.run_backfill(store, apply=False, threshold=0.75, top_k=5)
    assert dry["dry_run"] is True
    assert dry["scanned"] == 3
    assert dry["candidates"] == 1
    assert dry["added"] == 0
    assert _count(store) == 0
    assert dry["samples"][0]["type"] == "similar"
    assert {dry["samples"][0]["from_id"], dry["samples"][0]["to_id"]} == {"a", "b"}

    wrote = backfill.run_backfill(store, apply=True, threshold=0.75, top_k=5)
    assert wrote["added"] == 1
    assert _count(store) == 1
    again = backfill.run_backfill(store, apply=True, threshold=0.75, top_k=5)
    assert again["added"] == 0
    assert again["skipped_existing"] >= 1
    assert _count(store) == 1


def test_llm_confirm_writes_llm_type_and_reject_writes_nothing(tmp_path):
    store = _store(tmp_path)
    dry = backfill.run_backfill(
        store, apply=False, llm=True, confirm=lambda _a, _b: True, threshold=0.75)
    assert dry["dry_run"] is True and dry["llm_confirmed"] == 1 and _count(store) == 0

    wrote = backfill.run_backfill(
        store, apply=True, llm=True, confirm=lambda _a, _b: True, threshold=0.75)
    assert wrote["added"] == 1
    assert store.con.execute("SELECT type FROM memory_links").fetchone()[0] == "llm"

    other = Store(tmp_path / "m2.sqlite", "toy")
    other.upsert(["a", "b"], ["kettle zebra", "kettle zebra too"], [{}, {}],
                 [[1.0, 0.0], [1.0, 0.0]])
    rejected = backfill.run_backfill(
        other, apply=True, llm=True, confirm=lambda _a, _b: False, threshold=0.75)
    assert rejected["added"] == 0 and rejected["llm_rejected"] == 1
    assert _count(other) == 0


def test_llm_without_key_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "load_omniroute_key", lambda: None)
    store = _store(tmp_path)
    report = backfill.run_backfill(store, apply=True, llm=True, threshold=0.75)
    assert report["llm_unavailable"] is True
    assert report["added"] == 0
    assert report["candidates"] == 1
    assert _count(store) == 0


def test_key_from_env_or_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OMNIROUTE_API_KEY_FILE", raising=False)
    path = tmp_path / "omniroute.key"
    path.write_text("local-test-key\n", encoding="utf-8")
    monkeypatch.setenv("OMNIROUTE_API_KEY_FILE", str(path))
    assert backfill.load_omniroute_key() == "local-test-key"
    monkeypatch.setenv("OMNIROUTE_API_KEY", "from-env")
    assert backfill.load_omniroute_key() == "from-env"


def test_llm_posts_to_local_omniroute(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Yes"}}]}

    def fake_post(url, headers, json, timeout):
        seen["url"] = url
        seen["auth"] = headers["Authorization"]
        seen["body"] = json
        seen["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(backfill.requests, "post", fake_post)
    assert backfill.llm_same_topic("alpha", "beta", key="test-key") is True
    assert seen["url"] == "http://127.0.0.1:20128/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert "alpha" in seen["body"]["messages"][1]["content"]

    def no(url, headers, json, timeout):
        class No:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "no"}}]}
        return No()

    monkeypatch.setattr(backfill.requests, "post", no)
    assert backfill.llm_same_topic("alpha", "beta", key="test-key") is False


def test_main_defaults_to_dry_run(tmp_path, capsys):
    db = tmp_path / "empty.sqlite"
    Store(db, "toy")
    assert backfill.main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "added: 0" in out
    store = Store(db, "toy")
    assert _count(store) == 0
    with pytest.raises(SystemExit):
        backfill.main(["--db", str(db), "--apply", "--dry-run"])
