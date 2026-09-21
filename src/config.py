import os

# Auto-load .env so the pipeline finds GROQ_API_KEY without manual setup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

POLICY_DOCS_PATH = "data/policy_docs"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "policies"
