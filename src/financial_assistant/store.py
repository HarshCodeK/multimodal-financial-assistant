import sqlite3
import json
from pathlib import Path


class InteractionStore:
    """Bounded SQLite log for questions, answers, elapsed time, and extraction outcomes."""

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
                "answer TEXT NOT NULL, latency_ms REAL, extraction_result TEXT)"
            )
            # Migrate databases created by earlier versions without deleting history.
            columns = {row[1] for row in db.execute("PRAGMA table_info(interactions)")}
            if "latency_ms" not in columns:
                db.execute("ALTER TABLE interactions ADD COLUMN latency_ms REAL")
            if "extraction_result" not in columns:
                db.execute("ALTER TABLE interactions ADD COLUMN extraction_result TEXT")

    def record(
        self,
        question: str,
        supported: bool,
        answer: str,
        latency_ms: float | None = None,
        extraction_result: dict | None = None,
    ) -> None:
        serialized_extraction = json.dumps(extraction_result, ensure_ascii=False) if extraction_result is not None else None
        with sqlite3.connect(self.path, timeout=15.0) as db:
            db.execute("PRAGMA busy_timeout=15000")
            db.execute(
                "INSERT INTO interactions(question, supported, answer, latency_ms, extraction_result) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, int(supported), answer, latency_ms, serialized_extraction),
            )
            db.execute(
                "DELETE FROM interactions WHERE id <= "
                "(SELECT MAX(id) FROM interactions) - ?",
                (self.max_records,),
            )
