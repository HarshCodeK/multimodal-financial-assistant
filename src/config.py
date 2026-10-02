"""Paths and retrieval settings."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(ROOT, "data", "policy_docs")
CHROMA_DIR = os.environ.get("MFA_CHROMA", os.path.join(ROOT, "chroma_db"))
DB_PATH = os.environ.get("MFA_DB", os.path.join(ROOT, "logs.db"))

# 100-word chunks. Sentence-aware splitting would be better; at ~30 chunks over
# three short policy documents the measured retrieval is identical, so the
# simpler version stays.
CHUNK_WORDS = 100
TOP_K = 3
