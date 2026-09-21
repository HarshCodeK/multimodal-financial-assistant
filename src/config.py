import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(__file__))
POLICY_DOCS_PATH = os.path.join(ROOT, "data", "policy_docs")
TOP_K = 3
