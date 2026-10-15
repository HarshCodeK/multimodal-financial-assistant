# Multimodal Financial Assistant

A small, grounded assistant for financial-policy documents. It combines document extraction, vector retrieval, a configurable LLM, and an API/UI boundary. Unsupported questions are refused rather than guessed.

## What it demonstrates

- PDF text extraction with page provenance
- ChromaDB semantic retrieval with source replacement on re-ingestion
- Local sentence-transformer embeddings
- Grounded LLM answers with an explicit refusal path
- Image validation and vision-model input encoding
- FastAPI service + Streamlit demo
- SQLite interaction logging
- Docker and GitHub Actions
- Testable provider boundaries without requiring API credentials

## Architecture

`documents → extraction → embeddings → ChromaDB → retrieval → grounded answer`

The Streamlit UI calls FastAPI. SQLite stores interaction history; it is deliberately not used as the vector database.

See `docs/ARCHITECTURE.md`, `docs/INTERVIEW_QA.md`, `docs/THREAT_MODEL.md`, and `docs/RETRIEVAL_RELIABILITY.md` for design decisions, threat boundaries, and retrieval failure modes.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn financial_assistant.api:app --reload
```

In another shell:

```bash
streamlit run src/financial_assistant/app.py
```

Set `GROQ_API_KEY` for live answers. `GROQ_MODEL` and `GROQ_VISION_MODEL` are configurable.

## Tests

```bash
pytest
```

The Chroma integration tests cover round-trip retrieval, source replacement after re-ingestion, bounded query sizes, and embedder contract failures. The integration tests are skipped locally if ChromaDB is unavailable; CI installs the declared dependencies.

## Honest scope

This is a portfolio engineering project, not financial advice software. It does not execute payments, certify compliance, guarantee hallucination-free output, or provide production document governance. Image questions use a dedicated `/describe-image` vision endpoint. The persistent retrieval corpus remains text-first, so image-derived facts are not automatically added to ChromaDB. Re-ingestion removes stale records for the same source, but writes are serialized only within one process; multiple worker processes need an external ingestion lock or job queue. The image endpoint accepts bounded PNG/JPEG data URLs and sends them to the configured vision model.

## License

MIT. The repository includes the standard MIT license text in `LICENSE`.

## Verification
The default CI workflow runs unit tests, the Chroma retrieval integration test, source compilation, whitespace checks, and a Docker health smoke test.
