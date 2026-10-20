"""Bounded SQLite log for questions, latency, token usage, and estimated API cost."""

import json
import sqlite3
from pathlib import Path


class InteractionStore:
    """Persist a bounded local history and usage ledger without storing uploaded image bytes."""

    def __init__(self, path: str, max_records: int = 10_000):
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.path = Path(path)
        self.max_records = max_records
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=15.0) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS interactions ("
                "id INTEGER PRIMARY KEY, question TEXT NOT NULL, supported INTEGER NOT NULL, "
                "answer TEXT NOT NULL, latency_ms REAL, extraction_result TEXT, "
                "workflow TEXT NOT NULL DEFAULT 'pdf_rag', model TEXT, prompt_tokens INTEGER, "
                "completion_tokens INTEGER, estimated_cost_usd REAL, input_chars INTEGER, "
                "input_words INTEGER, input_bytes INTEGER)"
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(interactions)")}
            migrations = {
                "latency_ms": "REAL",
                "extraction_result": "TEXT",
                "workflow": "TEXT NOT NULL DEFAULT 'pdf_rag'",
                "model": "TEXT",
                "prompt_tokens": "INTEGER",
                "completion_tokens": "INTEGER",
                "estimated_cost_usd": "REAL",
                "input_chars": "INTEGER",
                "input_words": "INTEGER",
                "input_bytes": "INTEGER",
            }
            for name, sql_type in migrations.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE interactions ADD COLUMN {name} {sql_type}")

    def record(
        self,
        question: str,
        supported: bool,
        answer: str,
        latency_ms: float | None = None,
        extraction_result: dict | None = None,
        *,
        workflow: str = "pdf_rag",
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        input_chars: int | None = None,
        input_words: int | None = None,
        input_bytes: int | None = None,
    ) -> None:
        serialized_extraction = (
            json.dumps(extraction_result, ensure_ascii=False) if extraction_result is not None else None
        )
        with sqlite3.connect(self.path, timeout=15.0) as db:
            db.execute("PRAGMA busy_timeout=15000")
            db.execute(
                "INSERT INTO interactions("
                "question, supported, answer, latency_ms, extraction_result, workflow, model, "
                "prompt_tokens, completion_tokens, estimated_cost_usd, input_chars, input_words, input_bytes"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question, int(supported), answer, latency_ms, serialized_extraction, workflow, model,
                    prompt_tokens, completion_tokens, estimated_cost_usd, input_chars, input_words, input_bytes,
                ),
            )
            db.execute(
                "DELETE FROM interactions WHERE id <= "
                "(SELECT MAX(id) FROM interactions) - ?",
                (self.max_records,),
            )

    def recent_usage(self, limit: int = 100) -> list[dict]:
        """Return recent usage metadata; never includes API keys or uploaded image bytes."""
        safe_limit = min(max(int(limit), 1), 500)
        with sqlite3.connect(self.path, timeout=15.0) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT id, workflow, model, prompt_tokens, completion_tokens, estimated_cost_usd, "
                "latency_ms, input_chars, input_words, input_bytes, question "
                "FROM interactions ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]
