"""SQLite interaction log — one row per question answered.

Why the schema changed
----------------------
The old table recorded the question, answer, extracted fields and latency, but
not *which model produced it* or *which policy documents were cited*. That makes
the log useless for the question it exists to answer: "why did it say that?"

Two columns were added. The migration is additive and idempotent — an existing
`logs.db` keeps its rows and gains the new columns on the next write.

JSON lives in TEXT columns rather than normalised child tables: this log is read
as whole rows in the UI and never aggregated in SQL, so normalising buys nothing.
"""
import sqlite3
import json
import os

DB_PATH = os.environ.get(
    "MFA_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs.db")
)

_BASE_COLUMNS = """
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    filename TEXT,
    question TEXT,
    answer TEXT,
    extracted_fields TEXT,
    latency_ms REAL
"""

# Added after v1; applied via ALTER so an existing logs.db keeps its rows.
_NEW_COLUMNS = {"model": "TEXT", "sources": "TEXT"}


def _get_conn():
    # Row access by name, so adding a column cannot silently shift indices.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the table, and add newer columns to an existing one.

    Why the ALTER: `CREATE TABLE IF NOT EXISTS` silently does nothing when the
    table already exists with an older shape, so a new column would never appear
    on any machine that already had a logs.db.
    """
    conn = _get_conn()
    conn.execute(f"CREATE TABLE IF NOT EXISTS logs ({_BASE_COLUMNS})")
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(logs)")}
    for column, coltype in _NEW_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE logs ADD COLUMN {column} {coltype}")
    conn.commit()
    conn.close()


def log_interaction(filename, question, answer, extracted_fields, latency_ms,
                    model=None, sources=None):
    init_db()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO logs
           (timestamp, filename, question, answer, extracted_fields, latency_ms, model, sources)
           VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)""",
        (filename, question, answer, json.dumps(extracted_fields, default=str),
         latency_ms, model, json.dumps(sources or [])),
    )
    conn.commit()
    conn.close()


def get_recent_logs(limit=20):
    init_db()
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def usage_summary():
    """Small aggregate for the sidebar."""
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) AS n, AVG(latency_ms) AS avg_ms FROM logs").fetchone()
    by_model = conn.execute(
        "SELECT model, COUNT(*) AS n FROM logs WHERE model IS NOT NULL GROUP BY model"
    ).fetchall()
    conn.close()
    return {
        "interactions": row["n"],
        "avg_latency_ms": round(row["avg_ms"] or 0, 1),
        "by_model": {r["model"]: r["n"] for r in by_model},
    }
