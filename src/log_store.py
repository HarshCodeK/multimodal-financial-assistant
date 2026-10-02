"""Record every question asked and how it was answered.

Why: when a financial answer looks wrong, the only useful questions are "what
did the model see" and "what was it given". Logging the model and the cited
sources is what makes that answerable after the fact.
"""
import json
import os
import sqlite3

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY,
    asked_at TEXT DEFAULT (datetime('now')),
    filename TEXT,
    question TEXT,
    answer TEXT,
    fields TEXT,
    sources TEXT,
    model TEXT,
    elapsed_ms REAL
)
"""


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with _conn() as c:
        c.execute(_SCHEMA)


def record(filename, question, answer, fields, sources, model, elapsed_ms):
    init()
    with _conn() as c:
        c.execute(
            "INSERT INTO interactions (filename, question, answer, fields, sources, model, elapsed_ms)"
            " VALUES (?,?,?,?,?,?,?)",
            (filename, question, answer, json.dumps(fields, default=str),
             json.dumps(sources), model, elapsed_ms),
        )


def recent(limit: int = 10):
    init()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
