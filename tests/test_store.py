import sqlite3

from financial_assistant.store import InteractionStore


def test_interaction_store_retains_only_newest_records(tmp_path):
    path = tmp_path / "runtime" / "interactions.db"
    store = InteractionStore(str(path), max_records=3)

    for index in range(5):
        store.record(f"synthetic question {index}", True, f"synthetic answer {index}")

    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT question, answer FROM interactions ORDER BY id").fetchall()
    assert rows == [
        ("synthetic question 2", "synthetic answer 2"),
        ("synthetic question 3", "synthetic answer 3"),
        ("synthetic question 4", "synthetic answer 4"),
    ]


def test_interaction_store_records_latency_and_extraction_result(tmp_path):
    import json

    path = tmp_path / "interactions.db"
    store = InteractionStore(str(path))
    store.record(
        "PDF extraction: policy.pdf",
        True,
        "Indexed 2 text chunks across 2 pages.",
        latency_ms=18.5,
        extraction_result={"source": "policy.pdf", "chunks_indexed": 2, "pages_extracted": 2},
    )

    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT latency_ms, extraction_result FROM interactions"
        ).fetchone()

    assert row[0] == 18.5
    assert json.loads(row[1]) == {"source": "policy.pdf", "chunks_indexed": 2, "pages_extracted": 2}


def test_interaction_store_migrates_legacy_schema_without_losing_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE interactions (id INTEGER PRIMARY KEY, question TEXT NOT NULL, "
            "supported INTEGER NOT NULL, answer TEXT NOT NULL)"
        )
        db.execute(
            "INSERT INTO interactions(question, supported, answer) VALUES (?, ?, ?)",
            ("old question", 1, "old answer"),
        )

    InteractionStore(str(path))

    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT question, answer, latency_ms, extraction_result FROM interactions"
        ).fetchone()

    assert row == ("old question", "old answer", None, None)


def test_interaction_store_persists_usage_and_cost_metadata(tmp_path):
    store = InteractionStore(str(tmp_path / "usage.db"))
    store.record(
        "What is the total?",
        True,
        "INR 420",
        latency_ms=95.2,
        workflow="image_vision",
        model="qwen/qwen3.6-27b",
        prompt_tokens=2200,
        completion_tokens=24,
        estimated_cost_usd=0.001392,
        input_chars=18,
        input_words=4,
        input_bytes=1024,
    )

    row = store.recent_usage(1)[0]
    assert row["workflow"] == "image_vision"
    assert row["prompt_tokens"] == 2200
    assert row["completion_tokens"] == 24
    assert row["estimated_cost_usd"] == 0.001392
    assert row["input_bytes"] == 1024
    assert "GROQ_API_KEY" not in row
