# Multimodal Financial Assistant

A grounded assistant for questions about financial-policy documents. It combines PDF extraction, vector retrieval, a configurable LLM, and an API/UI boundary. Unsupported questions should be refused rather than guessed.

**Stack:** Python 3.11+, FastAPI, PyMuPDF, ChromaDB, sentence-transformers, Groq-compatible text/vision APIs, SQLite, Streamlit, Docker.

## What it demonstrates

- PDF text extraction with page provenance
- Semantic retrieval using ChromaDB
- Local sentence-transformer embeddings
- Grounded answers with a structured support/refusal field
- Bounded PNG/JPEG image input for a separate vision endpoint
- FastAPI service and Streamlit demo
- SQLite interaction logging
- Provider boundaries testable without API credentials
- Docker and GitHub Actions checks

## Architecture

`PDF → page-aware chunks → embeddings → ChromaDB → top-k retrieval → grounded answer`

The Streamlit UI calls FastAPI. SQLite stores interaction records; it is not the vector store. Image questions use a separate path:

`validated image data URL + question → vision provider → answer`

Image-derived facts are not automatically indexed in ChromaDB, so the image path does not currently have the same persistent source provenance as PDF retrieval.

See `docs/ARCHITECTURE.md`, `docs/INTERVIEW_QA.md`, and `docs/THREAT_MODEL.md` for design decisions and limitations.

## Quickstart

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn financial_assistant.api:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
streamlit run src/financial_assistant/app.py
```

Set `GROQ_API_KEY` for live answers. `GROQ_MODEL` and `GROQ_VISION_MODEL` are configurable. The first embedding-model use may download model files and consume additional startup time and memory. Review environment variables and storage paths before deployment; do not commit secrets or private financial documents.

## API surface

- `GET /health` — process-level health response.
- `POST /ask` — retrieves policy chunks, requests a grounded answer, and records the interaction.
- `POST /describe-image` — validates and sends a bounded image data URL to the vision provider.

A successful health response does not prove that ChromaDB, the embedding model, the LLM provider, or SQLite can complete a real request. Treat it as liveness, not a full dependency-readiness check.

## Tests and verification

Run:

```bash
pytest
```

The Chroma integration test can be skipped locally if ChromaDB is unavailable; CI installs declared dependencies and exercises a real Chroma round trip. The default CI workflow also runs unit tests, source compilation, whitespace checks, and a Docker health smoke test. Fake providers make core behavior testable without paid credentials.

## Grounding and safety contract

The text answerer receives retrieved evidence and returns structured `supported` and `answer` fields validated by Pydantic. Unsupported answers should not expose retrieved chunks as if they supported the response. This improves testability but is **not a formal hallucination guarantee**.

Image input is size-bounded and validated with Pillow before encoding. Documents and retrieved passages remain untrusted content; prompt wording alone is not a complete prompt-injection boundary. The project is not financial advice software, does not execute transactions, and does not certify compliance or production-grade document governance.

## Capacity and load testing

This repository currently makes no measured concurrent-user or production-capacity claim. A reproducible benchmark should state the exact commit, machine or hosting plan, cold/warm state, embedding model, corpus size, chunk count, top-k value, test duration, concurrency, and whether a live external LLM was used.

Measure p50/p95/p99 latency, throughput, HTTP errors/timeouts, CPU/RAM, retrieval latency, provider latency, refusal rate, and answer quality against a labeled question set. Separate retrieval-only tests using a deterministic answerer from end-to-end tests using a real provider; otherwise network latency and provider quotas obscure the retrieval service's own behavior. Record quality metrics separately from latency, and never describe generated test clients as real users.

Likely bottlenecks to measure include cold embedding-model initialization, CPU-heavy embedding, Chroma query behavior as the corpus grows, synchronous request work, SQLite write contention, and provider rate limits. These are test hypotheses, not measured findings. The default health endpoint is not a dependency-readiness check, and the app does not currently establish a production availability or maximum-concurrency guarantee.

## Known limitations

- Retrieval quality depends on the corpus, chunking, embeddings, and top-k configuration.
- Image-derived facts are not added to the persistent text retrieval index.
- Unsupported-answer behavior reduces risk but cannot guarantee zero hallucinations.
- SQLite is a simple interaction log, not a multi-host analytics store.
- Live provider behavior requires controlled tests with secrets and spend limits.
- No production throughput, accuracy, availability, or concurrency guarantee is claimed.

## License

MIT. See `LICENSE`.
