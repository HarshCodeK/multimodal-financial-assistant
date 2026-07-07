import sqlite3
import json

DB_PATH = "logs.db"

def _get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            filename TEXT,
            question TEXT,
            answer TEXT,
            extracted_fields TEXT,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(filename, question, answer, extracted_fields, latency_ms):
    init_db()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO logs (timestamp, filename, question, answer, extracted_fields, latency_ms) VALUES (datetime('now'), ?, ?, ?, ?, ?)",
        (filename, question, answer, json.dumps(extracted_fields), latency_ms),
    )
    conn.commit()
    conn.close()

def get_recent_logs(limit=20):
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, filename, question, answer, extracted_fields, latency_ms FROM logs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows
