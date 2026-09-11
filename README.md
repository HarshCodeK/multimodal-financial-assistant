# Multimodal Financial Assistant

A production-style RAG + vision pipeline that reads financial documents (PDF or image) and answers charge-dispute questions grounded in actual policy text — no hallucinated answers.

## What it does

- **Vision + text extraction** — Groq's `meta-llama/llama-4-scout-17b-16e-instruct` vision model extracts structured fields (vendor, line items, totals, date, flagged charge) from uploaded screenshots or PDFs
- **RAG policy grounding** — ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) retrieves relevant refund, subscription, and late-fee policy chunks so answers cite real rules
- **JSON-repair retry loop** — `vision_extractor.py` retries with a stricter prompt if the first extraction fails JSON parsing
- **Query logging** — every interaction (filename, question, answer, extracted fields, latency) persisted to SQLite via `monitor.py`
- **Streamlit UI** — upload, extract, ask, and inspect recent logs in one interface

## Architecture

```
Upload (PDF/Image)
      |
      v
document_parser.py — PyMuPDF for PDF, PIL path for image
      |
      v
vision_extractor.py — Groq vision LLM => structured JSON fields
      |
      v
qa_engine.py — retrieve_policy_context() via ChromaDB
      |
      v
Groq llama-3.3-70b-versatile — synthesize grounded answer
      |
      v
Streamlit UI + SQLite log
```

## Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

`groq` · `pymupdf` · `pillow` · `chromadb` · `sentence-transformers` · `streamlit` · `python-dotenv`

## Quickstart

```bash
git clone https://github.com/HarshCodeK/multimodal-financial-assistant.git
cd multimodal-financial-assistant
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env:
# GROQ_API_KEY=your-key-here

# one-time setup
python -c "from src.knowledge_base import build_knowledge_base; build_knowledge_base()"

streamlit run app.py
```

### Offline test suite

```bash
python -m pytest tests/ -q
```

No API key or network needed — document parsing, JSON-fence repair, and
SQLite logging are covered with the LLM layers stubbed.

## Example

**Upload:** credit card statement screenshot

**Extracted (from `vision_extractor.py`):**
```json
{
  "vendor": "Unknown",
  "line_items": [
    {"description": "Spotify Subscription", "amount": 11.99},
    {"description": "Progressive Insurance", "amount": 145.50},
    {"description": "Grand Hotel (2 nights)", "amount": 340.00}
  ],
  "total": 4512.78,
  "flagged_charge": {"description": "Progressive Insurance", "amount": 145.50}
}
```

**Question:** "Why was this charge deducted?"

**Answer (grounded in `data/policy_docs/subscription_billing.txt`):** "The Progressive Insurance charge of $145.50 was deducted as part of a recurring subscription billing cycle. Per policy, charges are processed on the same calendar day each month..."

## Status / Roadmap

Working prototype. Next steps:

- Better table extraction for complex PDF layouts
- Confidence scoring on extracted fields
- Streaming LLM responses in the UI
