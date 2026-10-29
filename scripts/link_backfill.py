#!/usr/bin/env python3
"""Weekly backfill of ``memory_links`` from hybrid search.

Dry-run by default: prints a report and does not insert link rows. Opening
the database creates the ``memory_links`` table when it is missing. Pass
``--apply`` to write. ``--llm`` asks a local OmniRoute server to confirm each
new pair (type ``llm`` instead of ``similar``).

The API key is read from the environment or a local file. It is never
hardcoded in this script.

This script does not install or edit crontab. Schedule it yourself only after
an explicit yes.

    python scripts/link_backfill.py              # report only
    python scripts/link_backfill.py --apply      # write type=similar
    python scripts/link_backfill.py --apply --llm
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import requests

LOG = logging.getLogger("link-backfill")

# Full chat-completions URL. Override with OMNIROUTE_URL (tests, other hosts).
DEFAULT_OMNIROUTE_URL = "http://127.0.0.1:20128/v1/chat/completions"
_TEXT_CLIP = 500


def _local_key_paths() -> list[Path]:
    from shared_memory._paths import home

    return [
        Path.home() / ".config" / "omniroute" / "api_key",
        Path.home() / ".omniroute" / "api_key",
        home() / "omniroute.key",
    ]


def _read_key_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        line = path.read_text(encoding="utf-8").strip().splitlines()
    except OSError as exc:
        LOG.warning("cannot read OmniRoute key file %s: %s", path, exc)
        return None
    if not line:
        return None
    key = line[0].strip()
    return key or None


def load_omniroute_key() -> str | None:
    """Key from env, ``embedding.env``, or a local file. Never a literal in source."""
    for name in ("OMNIROUTE_API_KEY", "OPENAI_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    file_env = (os.environ.get("OMNIROUTE_API_KEY_FILE") or "").strip()
    if file_env:
        return _read_key_file(Path(file_env).expanduser())

    from shared_memory.embedder import ENV_PATH, _load_env

    env = _load_env(ENV_PATH)
    if (env.get("OMNIROUTE_API_KEY") or "").strip():
        return env["OMNIROUTE_API_KEY"].strip()
    if (env.get("OMNIROUTE_API_KEY_FILE") or "").strip():
        found = _read_key_file(Path(env["OMNIROUTE_API_KEY_FILE"]).expanduser())
        if found:
            return found
    for path in _local_key_paths():
        found = _read_key_file(path)
        if found:
            return found
    return None


def omniroute_url() -> str:
    return (os.environ.get("OMNIROUTE_URL") or DEFAULT_OMNIROUTE_URL).strip()


def llm_same_topic(text_a: str, text_b: str, *, key: str, url: str | None = None,
                   model: str | None = None) -> bool:
    """Cheap yes/no confirm. Raises on HTTP or parse failure; caller counts it."""
    model = model or os.environ.get("OMNIROUTE_MODEL") or "gpt-4o-mini"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 3,
        "messages": [
            {"role": "system",
             "content": "You link memory notes. Reply with exactly yes or no."},
            {"role": "user",
             "content": (
                 "Same topic?\n\n"
                 f"A: {(text_a or '')[:_TEXT_CLIP]}\n\n"
                 f"B: {(text_b or '')[:_TEXT_CLIP]}"
             )},
        ],
    }
    response = requests.post(
        url or omniroute_url(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    word = str(content).strip().lower()
    if word.startswith("y"):
        return True
    if word.startswith("n"):
        return False
    raise ValueError(f"expected yes or no, got {content!r}")


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def run_backfill(store, *, apply: bool = False, llm: bool = False,
                 threshold: float | None = None, top_k: int | None = None,
                 limit: int = 0, confirm=None) -> dict:
    """Propose (and optionally write) links. ``confirm(text_a, text_b) -> bool``.

    Opening ``store`` may already have created the empty ``memory_links`` table.
    Dry-run still inserts no rows. LLM HTTP runs only when ``llm`` is set and
    a confirm function or an API key is available.
    """
    import numpy as np

    from shared_memory.links import DEFAULT_LINK_THRESHOLD, SIMILAR_TOP_K

    if threshold is None:
        threshold = DEFAULT_LINK_THRESHOLD
    if top_k is None:
        top_k = SIMILAR_TOP_K
    top_k = max(1, int(top_k))

    report = {
        "dry_run": not apply,
        "scanned": 0,
        "candidates": 0,
        "added": 0,
        "skipped_existing": 0,
        "below_threshold": 0,
        "llm_confirmed": 0,
        "llm_rejected": 0,
        "llm_errors": 0,
        "llm_unavailable": False,
        "samples": [],
    }

    occupied: set[tuple[str, str]] = set()
    for src, dst, typ in store.con.execute(
            "SELECT from_id, to_id, type FROM memory_links"):
        if typ in ("similar", "llm"):
            occupied.add(_pair_key(src, dst))

    confirm_fn = confirm
    if llm and confirm_fn is None:
        key = load_omniroute_key()
        if not key:
            report["llm_unavailable"] = True
            LOG.error("OMNIROUTE_API_KEY is not set and no local key file was found")
        else:
            url = omniroute_url()

            def confirm_fn(text_a, text_b, _key=key, _url=url):
                return llm_same_topic(text_a, text_b, key=_key, url=_url)

    sql = """SELECT id, doc, vec FROM chunks
             WHERE vec IS NOT NULL AND model=? ORDER BY rowid"""
    rows = store.con.execute(sql, (store.model,)).fetchall()
    if limit and int(limit) > 0:
        rows = rows[:int(limit)]

    pending = []
    for cid, doc, blob in rows:
        report["scanned"] += 1
        qvec = np.frombuffer(blob, dtype=np.float32).copy()
        hits = store.search(doc or "", qvec, top_k + 1)["hits"]
        kept = 0
        for hit in hits:
            other = hit["id"]
            if other == cid:
                continue
            score = float(hit["score"])
            if not math.isfinite(score) or score < threshold:
                report["below_threshold"] += 1
                continue
            if _pair_key(cid, other) in occupied:
                report["skipped_existing"] += 1
                continue
            link_type = "similar"
            if llm and confirm_fn is None:
                # No key: show the pair in the report, do not write it.
                occupied.add(_pair_key(cid, other))
                report["candidates"] += 1
                if len(report["samples"]) < 20:
                    report["samples"].append(
                        {"from_id": cid, "to_id": other, "type": "llm", "score": score})
                kept += 1
                if kept >= top_k:
                    break
                continue
            if llm:
                try:
                    ok = bool(confirm_fn(doc or "", hit["doc"]))
                except Exception:
                    LOG.exception("llm confirm failed for %s -> %s", cid, other)
                    report["llm_errors"] += 1
                    continue
                if not ok:
                    report["llm_rejected"] += 1
                    occupied.add(_pair_key(cid, other))
                    continue
                report["llm_confirmed"] += 1
                link_type = "llm"
            occupied.add(_pair_key(cid, other))
            report["candidates"] += 1
            item = {"from_id": cid, "to_id": other, "type": link_type, "score": score}
            if len(report["samples"]) < 20:
                report["samples"].append(item)
            pending.append((cid, other, link_type, score))
            kept += 1
            if kept >= top_k:
                break

    if apply and pending:
        report["added"] = store.add_links(pending)
    return report


def format_report(report: dict) -> str:
    mode = "dry-run" if report["dry_run"] else "apply"
    lines = [
        f"memory link backfill ({mode})",
        f"scanned: {report['scanned']}",
        f"candidates: {report['candidates']}",
        f"added: {report['added']}",
        f"skipped_existing: {report['skipped_existing']}",
        f"below_threshold: {report['below_threshold']}",
        f"llm_confirmed: {report['llm_confirmed']}",
        f"llm_rejected: {report['llm_rejected']}",
        f"llm_errors: {report['llm_errors']}",
        f"llm_unavailable: {report['llm_unavailable']}",
    ]
    if report["samples"]:
        lines.append("sample:")
        for item in report["samples"]:
            lines.append(
                f"  {item['from_id']} -> {item['to_id']} {item['type']} {item['score']:.4f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from shared_memory.embedder import ENV_PATH, _load_env, resolve_model
    from shared_memory.links import DEFAULT_LINK_THRESHOLD, SIMILAR_TOP_K
    from shared_memory.store import DEFAULT_SQLITE_PATH
    from shared_memory.store_sqlite import Store

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_SQLITE_PATH))
    ap.add_argument("--apply", action="store_true",
                    help="write links (default is a dry-run report)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing (this is the default)")
    ap.add_argument("--llm", action="store_true",
                    help="confirm each new pair via OmniRoute; stored type is llm")
    ap.add_argument("--threshold", type=float, default=DEFAULT_LINK_THRESHOLD)
    ap.add_argument("--top-k", type=int, default=SIMILAR_TOP_K)
    ap.add_argument("--limit", type=int, default=0, help="max chunks to scan (0 = all)")
    args = ap.parse_args(argv)
    if args.apply and args.dry_run:
        ap.error("pass only one of --apply and --dry-run")

    model = "/".join(resolve_model(_load_env(ENV_PATH)))
    store = Store(args.db, model)
    report = run_backfill(
        store,
        apply=bool(args.apply),
        llm=bool(args.llm),
        threshold=args.threshold,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
