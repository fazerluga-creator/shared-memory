"""Tests for the markdown chunking logic in examples/index_markdown.py."""
from __future__ import annotations

from conftest import load_module

idx = load_module("examples/index_markdown.py", "index_markdown")


def test_split_by_h2_basic():
    text = "intro line\n\n## First\nbody one\n\n## Second\nbody two"
    sections = idx._split_by_h2(text)
    assert [h for h, _ in sections] == ["<intro>", "First", "Second"]
    assert sections[1][1] == "body one"
    assert sections[2][1] == "body two"


def test_split_by_h2_no_headings_falls_back_to_whole_text():
    text = "just a plain note\nwith two lines"
    sections = idx._split_by_h2(text)
    assert len(sections) == 1
    assert sections[0][0] == "<intro>"
    assert "plain note" in sections[0][1]


def test_split_by_h2_ignores_h1_and_h3():
    text = "# Title\n\n### Sub\nbody"
    sections = idx._split_by_h2(text)
    assert len(sections) == 1
    assert sections[0][0] == "<intro>"


def test_split_paragraphs():
    text = "para one\n\npara two\n\n\npara three"
    assert idx._split_paragraphs(text) == ["para one", "para two", "para three"]


def test_split_paragraphs_empty_returns_original():
    assert idx._split_paragraphs("") == [""]


def test_chunks_for_small_file(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("## Decisions\nShip on Friday.\n\n## Risks\nNone known.", encoding="utf-8")
    md = idx.MdFile(path=p, agent="test")
    chunks = idx._chunks_for(md)
    assert len(chunks) == 2
    text0, meta0 = chunks[0]
    assert "Decisions" in text0 and "Ship on Friday." in text0
    assert meta0["agent"] == "test"
    assert meta0["source"] == "markdown"
    assert meta0["heading"] == "Decisions"
    assert meta0["chunk_index"] == 0
    assert "T" in meta0["timestamp"]  # ISO format


def test_chunks_for_oversized_section_soft_splits(tmp_path):
    big = "word " * 3000  # ~15000 chars, > MAX_CHUNK_CHARS
    p = tmp_path / "big.md"
    p.write_text(f"## Huge\n{big}", encoding="utf-8")
    md = idx.MdFile(path=p, agent="test")
    chunks = idx._chunks_for(md)
    assert len(chunks) >= 2
    for text, meta in chunks:
        assert len(text) <= idx.MAX_CHUNK_CHARS + 100  # header line on top
        assert meta["heading"] == "Huge"
    # chunk_index must be unique and sequential
    indexes = [meta["chunk_index"] for _, meta in chunks]
    assert indexes == list(range(len(chunks)))


def test_discover_walks_recursively(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("y", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("z", encoding="utf-8")
    files = idx._discover(tmp_path, "test")
    names = sorted(f.path.name for f in files)
    assert names == ["a.md", "b.md"]


def test_discover_missing_dir_returns_empty(tmp_path):
    assert idx._discover(tmp_path / "nope", "test") == []


def test_existing_ids_for_path_returns_matching_ids():
    class FakeCollection:
        def get(self, where=None, include=None):
            assert where is not None
            return {"ids": ["a::markdown::note.md::000", "a::markdown::note.md::001"]}

    ids = idx._existing_ids_for_path(FakeCollection(), "a", "note.md")
    assert ids == ["a::markdown::note.md::000", "a::markdown::note.md::001"]


def test_soft_split_prefers_word_boundaries():
    # Build text longer than the soft cap with a clear mid-word risk if sliced blindly.
    words = ["alpha", "beta", "gamma", "delta"] * 600  # plenty > 8000 chars
    text = " ".join(words)
    pieces = idx._soft_split(text, max_chars=200)
    assert len(pieces) >= 2
    for piece in pieces:
        assert len(piece) <= 200
        assert not piece.endswith("alph")  # would indicate mid-word cut of "alpha"


def test_chunks_for_uses_posix_path(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("## One\nbody", encoding="utf-8")
    md = idx.MdFile(path=p, agent="test")
    chunks = idx._chunks_for(md)
    assert chunks[0][1]["path"] == p.as_posix()
    assert "\\" not in chunks[0][1]["path"]
