"""SQLite storage backend: chunks, FTS5 full-text index and vectors in one file.

Why not a vector database: at agent-memory scale (tens of thousands of chunks)
a brute-force cosine over an in-memory numpy matrix takes milliseconds
(25k x 1024 float32 is ~100 MB) and needs no extra service, no HNSW index to
corrupt and no extension such as sqlite-vec. Vectors are stored as BLOBs next
to the text, so one ``sqlite3 .backup`` / ``VACUUM INTO`` copies everything.

Ranking (measured on a 32-query benchmark over a real 25k-chunk agent memory,
right file in the top 3: vector-only 26/32, this ranking 29-30/32):

* cosine similarity x a per-kind weight. ``metadata.kind`` is ``note``
  (distilled facts, weight 1.0), ``doc`` (documentation, 0.97) or
  ``transcript`` (raw conversation logs, 0.90). Unknown kinds count as 1.0.
* exact tokens from the query (numbers of 4+ digits, words containing a digit,
  ``ABC-123`` ids) add a bonus to chunks that contain them verbatim: dense
  vectors are weak on identifiers, plain text search is not.
* chunks marked ``metadata.superseded = true`` are multiplied by 0.9, so an
  outdated note still shows up, just below the current one.

With no query vector (embedding provider unavailable) search falls back to
FTS5/BM25 and says so in the result (``mode="bm25"``) instead of failing.

``SqliteCollection`` exposes the subset of the Chroma collection API used by
this project (add/upsert/update/get/delete/query/count), so the MCP server and
indexers work unchanged on either backend.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KIND_WEIGHT = {"note": 1.0, "doc": 0.97, "transcript": 0.90}
LITERAL_BOOST = 0.3
SUPERSEDED_WEIGHT = 0.9

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(
  rowid INTEGER PRIMARY KEY,
  id TEXT UNIQUE NOT NULL,
  kind TEXT,
  ts TEXT,                       -- canonical UTC ISO timestamp (metadata.timestamp)
  meta TEXT NOT NULL DEFAULT '{}',
  doc TEXT NOT NULL,
  hash TEXT NOT NULL,            -- sha1 of the text: re-embed only what changed
  model TEXT,                    -- embedding model of `vec`; a model change means re-index
  vec BLOB,
  superseded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chunks_ts ON chunks(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  doc, content='chunks', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO fts(rowid, doc) VALUES (new.rowid, new.doc); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO fts(fts, rowid, doc) VALUES ('delete', old.rowid, old.doc); END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF doc ON chunks BEGIN
  INSERT INTO fts(fts, rowid, doc) VALUES ('delete', old.rowid, old.doc);
  INSERT INTO fts(rowid, doc) VALUES (new.rowid, new.doc); END;
"""

# Separate from SCHEMA so applying or dropping links never rewrites `chunks`.
LINK_TYPES = ("similar", "tag", "llm", "manual")

MEMORY_LINKS_UP = """
CREATE TABLE IF NOT EXISTS memory_links(
  from_id TEXT NOT NULL,
  to_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK(type IN ('similar', 'tag', 'llm', 'manual')),
  score REAL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(from_id, to_id, type)
);
CREATE INDEX IF NOT EXISTS memory_links_from_id ON memory_links(from_id);
CREATE INDEX IF NOT EXISTS memory_links_to_id ON memory_links(to_id);
"""

MEMORY_LINKS_DOWN = "DROP TABLE IF EXISTS memory_links;"


def apply_memory_links(con: sqlite3.Connection) -> None:
    """Create ``memory_links`` (idempotent). Does not change chunks, FTS or triggers."""
    con.executescript(MEMORY_LINKS_UP)


def revert_memory_links(con: sqlite3.Connection) -> None:
    """Drop ``memory_links`` and its indexes. Chunk rows are left in place."""
    con.executescript(MEMORY_LINKS_DOWN)

_OPS = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def text_hash(doc: str) -> str:
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()


def canonical_ts(value) -> str | None:
    """UTC ISO string, so `since` can be compared as text in SQL.

    Indexers write "...Z", "+00:00", naive or unix-epoch timestamps; mixing them
    would make string comparison wrong. Unparseable values are kept as-is.
    """
    if value in (None, ""):
        return None
    s = str(value).strip()
    try:
        if s.isdigit():
            dt = datetime.fromtimestamp(int(s), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(s[:-1] + "+00:00" if s.endswith("Z") else s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return s


def exact_tokens(query: str) -> list[str]:
    """Tokens worth matching verbatim: 4+ digit numbers, words with a digit, ABC-123."""
    toks = re.findall(r"[^\W_][\w\-]*\d[\w\-]*|[A-Za-z]+-\d+", query)
    return sorted({t for t in toks if len(t) >= 4})


def fts_query(query: str) -> str:
    """OR of quoted terms, so FTS5 syntax characters in the query cannot break it."""
    toks = [t.strip(".-") for t in re.findall(r"[\w\-.]+", query.lower())]
    return " OR ".join(f'"{t}"' for t in toks if len(t) > 1) or '""'


def where_sql(where: dict | None, args: list) -> str:
    """Translate a Chroma-style metadata filter into SQL over the JSON `meta` column.

    Supported: ``{"k": v}``, ``{"k": {"$eq|$ne|$gt|$gte|$lt|$lte|$in|$nin": v}}``,
    ``{"$and": [...]}``, ``{"$or": [...]}``.
    """
    if not where:
        return "1"
    parts = []
    for key, val in where.items():
        if key in ("$and", "$or"):
            joiner = " AND " if key == "$and" else " OR "
            parts.append("(" + joiner.join(where_sql(w, args) for w in val) + ")")
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", key):
            raise ValueError(f"unsupported metadata key: {key!r}")
        col = f"json_extract(meta, '$.\"{key}\"')"
        op, arg = ("$eq", val) if not isinstance(val, dict) else next(iter(val.items()))
        if op in ("$in", "$nin"):
            neg = "NOT " if op == "$nin" else ""
            parts.append(f"{col} {neg}IN ({','.join('?' * len(arg))})")
            args.extend(arg)
        elif op in _OPS:
            parts.append(f"{col} {_OPS[op]} ?")
            args.append(arg)
        else:
            raise ValueError(f"unsupported filter operator: {op!r}")
    return " AND ".join(parts)


class Store:
    """Chunks + FTS5 + vectors in one SQLite file. Thread-safe enough for one MCP server."""

    def __init__(self, path: Path | str, model: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        # indexers and the server write from different processes
        self.con.execute("PRAGMA busy_timeout=30000")
        self.con.executescript(SCHEMA)
        apply_memory_links(self.con)
        self._mat = None  # (rowids, matrix, weights); dropped on any write
        self._ver = None

    # --- writes -------------------------------------------------------------
    def upsert(self, ids, documents, metadatas, embeddings=None) -> None:
        rows = []
        for i, (cid, doc, meta) in enumerate(zip(ids, documents, metadatas, strict=True)):
            meta = dict(meta or {})
            vec = None
            if embeddings is not None and embeddings[i] is not None:
                v = np.asarray(embeddings[i], dtype=np.float32)
                vec = (v / max(float(np.linalg.norm(v)), 1e-9)).tobytes()
            rows.append((cid, meta.get("kind"), canonical_ts(meta.get("timestamp")),
                         json.dumps(meta, ensure_ascii=False), doc or "", text_hash(doc or ""),
                         self.model if vec is not None else None, vec,
                         int(bool(meta.get("superseded")))))
        self.con.executemany(
            """INSERT INTO chunks(id, kind, ts, meta, doc, hash, model, vec, superseded)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, ts=excluded.ts,
                 meta=excluded.meta, doc=excluded.doc, hash=excluded.hash,
                 model=excluded.model, vec=excluded.vec, superseded=excluded.superseded""",
            rows)
        self.con.commit()
        self._mat = None

    def delete_ids(self, ids: Iterable[str]) -> int:
        gone = [(c,) for c in ids]
        # Drop dangling edges. No FK: the chunks table itself stays unchanged.
        self.con.executemany(
            "DELETE FROM memory_links WHERE from_id=? OR to_id=?",
            [(c, c) for (c,) in gone],
        )
        self.con.executemany("DELETE FROM chunks WHERE id=?", gone)
        self.con.commit()
        self._mat = None
        return len(gone)

    def add_links(self, rows: Iterable[tuple]) -> int:
        """Insert links. ``(from_id, to_id, type, score=None, created_at=None)``.

        Self-links are skipped. A repeated primary key refreshes ``score`` only.
        """
        now = datetime.now(timezone.utc).isoformat()
        clean = []
        for row in rows:
            if len(row) < 3:
                raise ValueError("link row needs from_id, to_id, type")
            from_id, to_id, typ = row[0], row[1], row[2]
            score = row[3] if len(row) > 3 else None
            created_at = row[4] if len(row) > 4 and row[4] else now
            if not from_id or not to_id or from_id == to_id:
                continue
            if typ not in LINK_TYPES:
                raise ValueError(f"unsupported link type: {typ!r}")
            clean.append((str(from_id), str(to_id), typ,
                          None if score is None else float(score), str(created_at)))
        if not clean:
            return 0
        self.con.executemany(
            """INSERT INTO memory_links(from_id, to_id, type, score, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(from_id, to_id, type) DO UPDATE SET score=excluded.score""",
            clean)
        self.con.commit()
        return len(clean)

    def links_for(self, chunk_id: str, link_type: str | None = None, limit: int = 50) -> list[dict]:
        """Links where ``chunk_id`` is either endpoint, newest first."""
        if link_type is not None and link_type not in LINK_TYPES:
            raise ValueError(f"unsupported link type: {link_type!r}")
        limit = max(1, min(int(limit), 200))
        sql = """SELECT from_id, to_id, type, score, created_at FROM memory_links
                 WHERE (from_id=? OR to_id=?)"""
        args: list = [chunk_id, chunk_id]
        if link_type:
            sql += " AND type=?"
            args.append(link_type)
        sql += " ORDER BY created_at DESC, from_id, to_id LIMIT ?"
        args.append(limit)
        out = []
        for from_id, to_id, typ, score, created_at in self.con.execute(sql, args):
            out.append({
                "from_id": from_id,
                "to_id": to_id,
                "type": typ,
                "score": None if score is None else float(score),
                "created_at": created_at,
            })
        return out

    def docs_containing(self, needle: str, exclude_id: str, limit: int = 200) -> list[tuple[str, str]]:
        """Newest chunks whose text contains ``needle`` inside a ``[[...]]`` span."""
        return self.con.execute(
            """SELECT id, doc FROM chunks
               WHERE id != ? AND instr(doc, '[[') > 0 AND instr(doc, ?) > 0
               ORDER BY rowid DESC
               LIMIT ?""",
            (exclude_id, needle, int(limit)),
        ).fetchall()

    def count(self) -> int:
        return self.con.execute("SELECT count(*) FROM chunks").fetchone()[0]

    # --- search -------------------------------------------------------------
    def _matrix(self):
        # another process (an indexer) may have written since we built the matrix
        ver = self.con.execute("PRAGMA data_version").fetchone()[0]
        if ver != self._ver:
            self._ver, self._mat = ver, None
        if self._mat is None:
            rs = self.con.execute(
                "SELECT rowid, vec, kind, superseded FROM chunks WHERE vec IS NOT NULL AND model=?",
                (self.model,)).fetchall()
            ids = np.array([r[0] for r in rs], dtype=np.int64)
            mat = (np.frombuffer(b"".join(r[1] for r in rs), dtype=np.float32).reshape(len(rs), -1)
                   if rs else np.zeros((0, 1), dtype=np.float32))
            w = np.array([KIND_WEIGHT.get(r[2], 1.0) * (SUPERSEDED_WEIGHT if r[3] else 1.0)
                          for r in rs], dtype=np.float32)
            self._mat = (ids, mat, w)
        return self._mat

    def _allowed(self, where, where_document, since):
        if not (where or where_document or since):
            return None
        args: list = []
        sql = "SELECT rowid FROM chunks WHERE " + where_sql(where, args)
        if where_document:
            sql += " AND instr(doc, ?) > 0"
            args.append(where_document["$contains"])
        if since:
            sql += " AND ts >= ?"
            args.append(canonical_ts(since))
        return {r[0] for r in self.con.execute(sql, args)}

    def search(self, query: str, qvec, top_k: int = 10, *, where: dict | None = None,
               where_document: dict | None = None, since: str | None = None) -> dict:
        """Return ``{"mode": "hybrid"|"bm25", "hits": [...]}``; each hit has id, doc, meta, score.

        ``since`` is a canonical UTC ISO timestamp compared in SQL, before ranking,
        so a narrow time window never starves the result list.
        """
        allow = self._allowed(where, where_document, since)
        if qvec is None:
            return {"mode": "bm25", "hits": self._bm25(query, top_k, allow)}
        ids, mat, w = self._matrix()
        if not len(ids):
            return {"mode": "bm25", "hits": self._bm25(query, top_k, allow)}
        q = np.asarray(qvec, dtype=np.float32)
        s = (mat @ (q / max(float(np.linalg.norm(q)), 1e-9))) * w
        toks = exact_tokens(query or "")
        if toks:
            lit: set[int] = set()
            for t in toks:
                lit |= {r[0] for r in self.con.execute(
                    "SELECT rowid FROM chunks WHERE instr(doc, ?) > 0", (t,))}
            if lit:
                s = s + np.isin(ids, list(lit)).astype(np.float32) * LITERAL_BOOST * w
        if allow is not None:
            s = np.where(np.isin(ids, list(allow)), s, -np.inf)
        hits = []
        for j in np.argsort(-s)[:top_k]:
            if not np.isfinite(s[j]):
                break
            hits.append(self._row(int(ids[j]), float(s[j])))
        return {"mode": "hybrid", "hits": hits}

    def _bm25(self, query, top_k, allow):
        rs = self.con.execute(
            "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
            (fts_query(query or ""), top_k * 10 if allow is None else 5000)).fetchall()
        hits = [self._row(r[0], -float(r[1])) for r in rs if allow is None or r[0] in allow]
        hits.sort(key=lambda h: -h["score"] * KIND_WEIGHT.get(h["meta"].get("kind"), 1.0))
        return hits[:top_k]

    def _row(self, rowid: int, score: float) -> dict:
        cid, meta, doc = self.con.execute(
            "SELECT id, meta, doc FROM chunks WHERE rowid=?", (rowid,)).fetchone()
        return {"id": cid, "meta": json.loads(meta), "doc": doc, "score": score}


class SqliteCollection:
    """The subset of the Chroma collection API this project uses, backed by ``Store``."""

    def __init__(self, store: Store, name: str):
        self.s = store
        self.name = name
        self.metadata = {"backend": "sqlite", "embedder": store.model}

    def add(self, ids, embeddings=None, documents=None, metadatas=None):
        self.s.upsert(ids, documents or [""] * len(ids), metadatas or [{}] * len(ids), embeddings)

    upsert = add

    def update(self, ids, documents=None, metadatas=None, embeddings=None):
        old = self.get(ids=ids, include=["documents", "metadatas"])
        by_id = dict(zip(old["ids"], zip(old["documents"], old["metadatas"], strict=True),
                         strict=True))
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"unknown ids: {missing}")
        docs = documents or [by_id[i][0] for i in ids]
        metas = metadatas or [by_id[i][1] for i in ids]
        if embeddings is None:  # keep the stored vector when the text did not change
            embeddings = []
            for i in ids:
                blob = self.s.con.execute("SELECT vec FROM chunks WHERE id=?", (i,)).fetchone()[0]
                embeddings.append(None if blob is None else np.frombuffer(blob, dtype=np.float32))
        self.add(ids, embeddings, docs, metas)

    def _select(self, ids=None, where=None, where_document=None, limit=None, offset=None):
        args: list = []
        sql = "SELECT id, doc, meta FROM chunks WHERE " + where_sql(where, args)
        if ids is not None:
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args += list(ids)
        if where_document:
            sql += " AND instr(doc, ?) > 0"
            args.append(where_document["$contains"])
        sql += " ORDER BY rowid"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            args += [int(limit), int(offset or 0)]
        return self.s.con.execute(sql, args).fetchall()

    def get(self, ids=None, where=None, include=("documents", "metadatas"), limit=None,
            offset=None, where_document=None):
        rs = [] if ids is not None and not ids else self._select(
            ids, where, where_document, limit, offset)
        return {"ids": [r[0] for r in rs],
                "documents": [r[1] for r in rs] if "documents" in include else None,
                "metadatas": [json.loads(r[2]) for r in rs] if "metadatas" in include else None}

    def delete(self, ids=None, where=None):
        if ids is None:
            ids = [r[0] for r in self._select(None, where)]
        self.s.delete_ids(ids)

    def count(self):
        return self.s.count()

    def query(self, query_embeddings, n_results=10, where=None, where_document=None,
              include=None, query_texts=None, since=None):
        """Chroma-shaped results; distance is ``1 - score`` (score already weighted).

        Extra keywords over Chroma: ``query_texts`` feeds exact-token boosting and
        the BM25 fallback, ``since`` filters by timestamp in SQL.
        """
        out = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        texts = query_texts or [""] * len(query_embeddings)
        for qv, qt in zip(query_embeddings, texts, strict=True):
            hits = self.s.search(qt, qv, n_results, where=where,
                                 where_document=where_document, since=since)["hits"]
            out["ids"].append([h["id"] for h in hits])
            out["documents"].append([h["doc"] for h in hits])
            out["metadatas"].append([h["meta"] for h in hits])
            out["distances"].append([1.0 - h["score"] for h in hits])
        return out
