# Multimodal Financial Assistant

A simple financial-document assistant:

PDF/Image -> field extraction -> policy retrieval -> LLM answer -> SQLite logging

## Run on Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq key to `.env`:

```
GROQ_API_KEY=your_key_here
```

Start the app:

```bash
python -m streamlit run app.py
```

Run tests:

```bash
python -m pytest tests/ -q
```

## Structure

- `app.py` — Streamlit interface
- `src/document_parser.py` — PDF/image input
- `src/vision_extractor.py` — extracts fields with Groq
- `src/knowledge_base.py` — simple TF-IDF policy retrieval
- `src/qa_engine.py` — grounded answer generation
- `src/monitor.py` — SQLite logging
- `data/policy_docs/` — policy files
- `tests/` — offline tests

No ChromaDB or separate knowledge-base setup is required.
