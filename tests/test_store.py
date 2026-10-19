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
