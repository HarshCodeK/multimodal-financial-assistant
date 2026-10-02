"""Offline tests. No network, no API key.

These prove the two claims that matter: that retrieval really runs on ChromaDB
with real embeddings, and that no module can pin a model Groq has retired.

Run:  python -m pytest tests/ -q
"""
import ast
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import models  # noqa: E402
from src.extract import ExtractionError, extract  # noqa: E402
from src.parser import load  # noqa: E402

RETIRED = ("llama-3.3-70b-versatile", "llama-4-scout-17b-16e-instruct",
           "llama-3.1-8b-instant", "qwen/qwen3-32b", "qwen/qwen3.6-27b")


# ---------------------------------------------------------------------------
# The model registry
# ---------------------------------------------------------------------------
class TestModels:
    def test_default_is_live(self):
        assert models.resolve() in models.MODELS
        assert models.DEFAULT_MODEL in models.VISION_MODELS

    def test_unknown_model_raises(self):
        for dead in RETIRED:
            with pytest.raises(models.UnknownModel):
                models.resolve(dead)

    def test_exactly_one_vision_model(self):
        assert models.VISION_MODELS == ["qwen/qwen3.8-27b"]

    def test_prices_are_sane(self):
        for mid, (_, _, pin, pout) in models.MODELS.items():
            assert pin > 0 and pout > pin, mid

    def test_no_retired_model_used_as_a_value(self):
        """Walk the AST. Docstrings may name a retired model to explain the
        migration; a code value may not."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in {"__pycache__", ".git", "chroma_db", "tests"}]
            for name in filenames:
                if not name.endswith(".py") or name == "models.py":
                    continue
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                docstrings = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                        body = getattr(node, "body", None)
                        if (body and isinstance(body[0], ast.Expr)
                                and isinstance(body[0].value, ast.Constant)
                                and isinstance(body[0].value.value, str)):
                            docstrings.add(id(body[0].value))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Constant)
                            and isinstance(node.value, str)
                            and id(node) not in docstrings):
                        for dead in RETIRED:
                            if dead in node.value:
                                offenders.append(f"{name}:{node.lineno}")
        assert not offenders, f"retired model pinned as a value: {offenders}"


# ---------------------------------------------------------------------------
# ChromaDB retrieval -- the claim on the resume
# ---------------------------------------------------------------------------
class TestVectorStore:
    def test_embedding_model_is_local(self):
        from src.knowledge_base import EMBEDDING_MODEL
        # A local model is what makes retrieval free and offline.
        assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_chunking_is_deterministic(self):
        from src.knowledge_base import chunk
        text = " ".join(f"w{i}" for i in range(250))
        assert chunk(text) == chunk(text)
        assert [len(c.split()) for c in chunk(text)] == [100, 100, 50]

    def test_ingest_then_search_finds_the_right_policy(self, tmp_path):
        from src import knowledge_base as kb

        docs = tmp_path / "policies"
        docs.mkdir()
        (docs / "refund.txt").write_text(
            "Refunds are issued within 30 days of the original purchase date. " * 12,
            encoding="utf-8")
        (docs / "late_fee.txt").write_text(
            "A late fee of 1.5 percent applies to any payment not received by the due date. " * 12,
            encoding="utf-8")

        kb.CHROMA_DIR = str(tmp_path / "chroma")
        kb._client = None
        kb._embedder = None
        try:
            stored = kb.ingest(str(docs))
            assert stored > 0

            hits = kb.search("how long do I have to get my money back")
            assert hits, "retrieval returned nothing"
            assert hits[0]["source"] == "refund.txt", \
                f"expected the refund policy first, got {hits[0]['source']}"

            # Deterministic ids: re-ingesting replaces rather than duplicates.
            kb.ingest(str(docs))
            assert kb.size() == stored
        finally:
            kb._client = None
            kb._embedder = None


# ---------------------------------------------------------------------------
# The vision guard
# ---------------------------------------------------------------------------
class TestVisionGuard:
    def test_text_only_model_refused_for_image(self, tmp_path):
        img = tmp_path / "r.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(ExtractionError, match="cannot read images"):
            extract(load(str(img)), "openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_unsupported_type_rejected(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        load(str(bad))


def test_image_routed(tmp_path):
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    d = load(str(img))
    assert d["type"] == "image" and d["filename"] == "receipt.png"


def test_pdf_text_extracted(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((50, 72), "TOTAL AMOUNT DUE: $4512.78")
    path = tmp_path / "invoice.pdf"
    doc.save(str(path)); doc.close()
    parsed = load(str(path))
    assert parsed["type"] == "pdf" and "4512.78" in parsed["text"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def test_log_records_model_and_sources(tmp_path, monkeypatch):
    from src import log_store
    monkeypatch.setattr(log_store, "DB_PATH", str(tmp_path / "l.db"))
    log_store.record("r.png", "why?", "because policy", {"total": 1},
                     ["refund_policy.txt"], "qwen/qwen3.8-27b", 1234.5)
    rows = log_store.recent(5)
    assert len(rows) == 1
    assert json.loads(rows[0]["sources"]) == ["refund_policy.txt"]
    assert rows[0]["model"] == "qwen/qwen3.8-27b"


class _FakeMessage:
    content = "answer"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeGroq:
    """Stands in for the groq client so no network call happens."""
    def __init__(self, **kwargs):
        self.chat = type(
            "Chat", (), {"completions": type(
                "Completions", (), {"create": staticmethod(lambda **kw: _FakeResponse())})()}
        )()


def test_cited_sources_are_deduplicated(monkeypatch):
    """top-k returns chunks, not documents, so one file can appear twice.

    Citing it twice makes correct retrieval look broken. Caught live: the
    sources list had late_fee_policy.txt on consecutive entries.
    """
    import os as _os
    from src import answer as answer_mod

    monkeypatch.setattr(answer_mod, "search", lambda q, **k: [
        {"text": "a", "source": "refund_policy.txt"},
        {"text": "b", "source": "refund_policy.txt"},
        {"text": "c", "source": "late_fee_policy.txt"},
    ])
    monkeypatch.setattr(answer_mod, "extract", lambda d, m=None: {"vendor": "V"})
    monkeypatch.setitem(_os.environ, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr("groq.Groq", _FakeGroq)

    out = answer_mod.answer({"type": "pdf", "text": "t", "filename": "f.pdf"}, "q?")
    assert out["sources"] == ["refund_policy.txt", "late_fee_policy.txt"]
