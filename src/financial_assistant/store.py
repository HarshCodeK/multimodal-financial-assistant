import sqlite3
from pathlib import Path


class InteractionStore:
    def __init__(self, path: str, max_records: int = 10_000):
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.path = Path(path)
        self.max_records = max_records
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=15.0) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("CREATE TABLE IF NOT EXISTS interactions (id INTEGER PRIMARY KEY, question TEXT NOT NULL, supported INTEGER NOT NULL, answer TEXT NOT NULL)")

    def record(self, question: str, supported: bool, answer: str) -> None:
        with sqlite3.connect(self.path, timeout=15.0) as db:
            db.execute("PRAGMA busy_timeout=15000")
            db.execute("INSERT INTO interactions(question, supported, answer) VALUES (?, ?, ?)", (question, int(supported), answer))
            db.execute(
                "DELETE FROM interactions WHERE id NOT IN "
                "(SELECT id FROM interactions ORDER BY id DESC LIMIT ?)",
                (self.max_records,),
            )
