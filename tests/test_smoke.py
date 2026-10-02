"""Offline tests — no network, no API key.

These cover the two claims that were previously false:

1. Retrieval really runs on ChromaDB with real embeddings (not TF-IDF).
2. No module can pin a model Groq has retired.

Plus the SQLite migration, because `monitor.init_db` now ALTERs an existing
table and that is exactly the kind of change that silently breaks upgrades.

Run: python -m pytest tests/ -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import models  # noqa: E402
from src.document_parser import load_document  # noqa: E402
from src.vision_extractor import _parse_json  # noqa: E402


# ---------------------------------------------------------------------------
# No retired model may be pinned anywhere
# ---------------------------------------------------------------------------
class TestNoRetiredModels:
    def test_registry_contains_no_retired_model(self):
        for model_id in models.MODELS:
            assert model_id not in models.RETIRED, f"{model_id} is retired"

    def test_every_default_is_live(self):
        assert models.DEFAULT_MODEL in models.MODELS
        assert models.DEFAULT_MODEL in models.FREE_TIER_SAFE

    def test_resolve_rejects_retired_model(self):
        for dead in ("llama-3.3-70b-versatile",
                     "meta-llama/llama-4-scout-17b-16e-instruct",
                     "qwen/qwen3.6-27b"):
            with pytest.raises(models.ModelRetiredError):
                models.resolve(dead)

    def test_resolve_error_names_the_replacement(self):
        with pytest.raises(models.ModelRetiredError, match="qwen3.6-27b"):
            models.resolve("qwen/qwen3.6-27b")

    def test_resolve_passes_live_model(self):
        assert models.resolve("qwen/qwen3.8-27b") == "qwen/qwen3.8-27b"

    def test_resolve_defaults_when_none(self):
        assert models.resolve(None) == models.DEFAULT_MODEL

    def test_no_source_file_pins_a_retired_model(self):
        """The real regression guard: inspect code values, not prose.

        A retired model id may legitimately appear in a docstring — explaining
        what got replaced and why is exactly the history worth keeping. It may
        never appear as a *value* in shipped code.

        So this walks the AST, discards docstrings, and fails on any remaining
        string constant containing a retired id. Scanned: `src/` and `app.py`.
        Skipped: `tests/` (must name retired ids to prove they are rejected)
        and `src/models.py` (the registry itself — holding retired ids as data
        is its whole purpose).
        """
        import ast

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        offenders = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in
                           {".git", "__pycache__", "chroma_db", ".venv", "tests"}]
            for name in filenames:
                if not name.endswith(".py") or name == "models.py":
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue

                # Collect docstring nodes so their text is not scanned.
                docstrings = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = getattr(node, "body", None)
                        if (body and isinstance(body[0], ast.Expr)
                                and isinstance(body[0].value, ast.Constant)
                                and isinstance(body[0].value.value, str)):
                            docstrings.add(id(body[0].value))

                for node in ast.walk(tree):
                    if (isinstance(node, ast.Constant)
                            and isinstance(node.value, str)
                            and id(node) not in docstrings):
                        for dead in models.RETIRED:
                            if dead in node.value:
                                offenders.append(
                                    f"{name}:{node.lineno} pins {dead}")

        assert not offenders, (
            "retired model used as a code value: " + "; ".join(offenders))


# ---------------------------------------------------------------------------
# ChromaDB retrieval — the claim the resume makes
# ---------------------------------------------------------------------------
class TestVectorStore:
    def test_knowledge_base_uses_chromadb(self):
        import src.knowledge_base as kb

        assert callable(kb._get_collection)
        assert "chromadb" in kb.__doc__.lower()

    def test_embedding_model_is_local_and_small(self):
        import src.knowledge_base as kb

        # A local model is what keeps this project offline-capable.
        assert kb.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_chunking_is_deterministic(self):
        import src.knowledge_base as kb

        text = " ".join(f"w{i}" for i in range(250))
        assert kb.chunk_text(text) == kb.chunk_text(text)
        chunks = kb.chunk_text(text)
        assert len(chunks) == 3
        assert len(chunks[0].split()) == 100

    def test_ingest_then_retrieve_is_idempotent(self, tmp_path):
        import src.knowledge_base as kb

        docs = tmp_path / "policy"
        docs.mkdir()
        (docs / "refund.txt").write_text(
            "Refunds are issued within 5 business days. " * 12, encoding="utf-8")
        (docs / "late_fee.txt").write_text(
            "A late fee of 1.5 percent applies after the due date. " * 12,
            encoding="utf-8")

        kb.CHROMA_PATH = str(tmp_path / "chroma")
        kb._client = None
        kb._embedder = None
        try:
            stored = kb.ingest(str(docs))
            assert stored > 0

            hits = kb.retrieve_policy_context("how long do refunds take")
            assert hits
            assert any("refund" in h.lower() for h in hits)

            # Deterministic ids mean re-ingest replaces rather than duplicates.
            kb.ingest(str(docs))
            assert kb.knowledge_size() == stored
        finally:
            kb._client = None
            kb._embedder = None


# ---------------------------------------------------------------------------
# Model capability + pricing
# ---------------------------------------------------------------------------
class TestModelRegistry:
    def test_exactly_one_vision_model_and_it_is_default(self):
        assert models.VISION_MODELS == ["qwen/qwen3.8-27b"]
        assert models.DEFAULT_MODEL in models.VISION_MODELS

    def test_pricing_is_positive_and_output_costs_more(self):
        for mid, meta in models.MODELS.items():
            assert meta["in_per_1m"] > 0, mid
            assert meta["out_per_1m"] > meta["in_per_1m"], mid

    def test_price_scales_across_models(self):
        cheap = models.price("openai/gpt-oss-20b", 1_000_000, 1_000_000)
        dear = models.price("qwen/qwen3.8-27b", 1_000_000, 1_000_000)
        assert dear > cheap

    def test_price_rejects_unknown_model(self):
        with pytest.raises(models.ModelRetiredError):
            models.price("not-a-model", 10, 10)

    def test_list_models_filters_by_vision(self):
        assert len(models.list_models(vision=True)) == 1
        assert all(not m["vision"] for m in models.list_models(vision=False))


# ---------------------------------------------------------------------------
# The vision guard
# ---------------------------------------------------------------------------
class TestVisionGuard:
    def test_text_model_rejected_for_image(self, tmp_path):
        from src.vision_extractor import ExtractionError, extract_fields

        img = tmp_path / "r.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(ExtractionError, match="does not accept image"):
            extract_fields(load_document(str(img)), model="openai/gpt-oss-20b")

    def test_json_repair_strips_code_fences(self):
        assert _parse_json('```json\n{"vendor": "Acme", "total": 100}\n```') == {
            "vendor": "Acme", "total": 100}
        assert _parse_json('  {"vendor": "Acme", "total": 100}  ') == {
            "vendor": "Acme", "total": 100}


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------
def test_parser_rejects_unsupported_format(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(str(bad))


def test_parser_handles_image_path(tmp_path):
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    doc = load_document(str(img))
    assert doc["type"] == "image"
    assert doc["filename"] == "receipt.png"


def test_parser_extracts_pdf_text(tmp_path):
    fitz = pytest.importorskip("fitz")
    d = fitz.open()
    d.new_page().insert_text((50, 72), "TOTAL AMOUNT DUE: $4512.78")
    path = tmp_path / "invoice.pdf"
    d.save(str(path))
    d.close()

    parsed = load_document(str(path))
    assert parsed["type"] == "pdf"
    assert "4512.78" in parsed["text"]


# ---------------------------------------------------------------------------
# SQLite log, including the migration
# ---------------------------------------------------------------------------
class TestMonitor:
    def test_roundtrip_records_model_and_sources(self, tmp_path, monkeypatch):
        import src.monitor as monitor

        monkeypatch.setattr(monitor, "DB_PATH", str(tmp_path / "t.db"))
        monitor.init_db()
        monitor.log_interaction(
            "invoice.png", "Why?", "Because policy X", {"total": 10}, 123.4,
            model="qwen/qwen3.8-27b", sources=["refund_policy.txt"])
        rows = monitor.get_recent_logs(5)
        assert len(rows) == 1
        assert rows[0]["filename"] == "invoice.png"
        assert rows[0]["model"] == "qwen/qwen3.8-27b"
        assert json.loads(rows[0]["sources"]) == ["refund_policy.txt"]

    def test_migration_adds_columns_and_keeps_rows(self, tmp_path, monkeypatch):
        """A logs.db written by v1 must gain the new columns without losing data."""
        import sqlite3

        import src.monitor as monitor

        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE logs (
                id INTEGER PRIMARY KEY, timestamp TEXT, filename TEXT,
                question TEXT, answer TEXT, extracted_fields TEXT,
                latency_ms REAL)"""
        )
        conn.execute(
            "INSERT INTO logs (timestamp, filename, question, answer,"
            " extracted_fields, latency_ms)"
            " VALUES ('2026-01-01', 'old.pdf', 'q', 'a', '{}', 1.0)")
        conn.commit()
        conn.close()

        monkeypatch.setattr(monitor, "DB_PATH", str(db))
        monitor.init_db()

        conn = sqlite3.connect(db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(logs)")}
        count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        conn.close()

        assert {"model", "sources"} <= cols
        assert count == 1, "migration must not drop existing rows"

    def test_init_db_is_idempotent(self, tmp_path, monkeypatch):
        import src.monitor as monitor

        monkeypatch.setattr(monitor, "DB_PATH", str(tmp_path / "i.db"))
        monitor.init_db()
        monitor.init_db()
        assert monitor.get_recent_logs(5) == []

    def test_usage_summary(self, tmp_path, monkeypatch):
        import src.monitor as monitor

        monkeypatch.setattr(monitor, "DB_PATH", str(tmp_path / "u.db"))
        monitor.log_interaction("a.pdf", "q", "a", {}, 100.0, model="m1")
        monitor.log_interaction("b.pdf", "q", "a", {}, 200.0, model="m1")
        summary = monitor.usage_summary()
        assert summary["interactions"] == 2
        assert summary["avg_latency_ms"] == 150.0
        assert summary["by_model"] == {"m1": 2}
