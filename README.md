# Multimodal Financial Assistant

A small, evidence-grounded assistant for financial-policy documents. Index a text-based PDF, ask a question against retrieved document chunks, inspect returned source evidence, and separately use vision inference to summarize a statement or chart image.

This is a portfolio system, not financial advice software. It does not execute transactions or guarantee error-free financial interpretation.

## What it demonstrates

- Bounded PDF upload and page-level extraction with stable, deterministic chunk IDs.
- Safe re-indexing: new chunks are embedded/upserted before stale chunks for the same source are removed.
- ChromaDB retrieval with local sentence-transformer embeddings.
- Grounded LLM answers with an explicit provider-error and unsupported-answer path.
- Validation for actual PNG/JPEG data URL bytes, declared image type, and size.
- Owner-token-gated PDF ingestion to prevent arbitrary visitors from contaminating the shared document index.
- A process-local shared provider request cap when a Groq key is configured.
- SQLite interaction logging with WAL and a writer wait timeout.
- FastAPI and Streamlit interfaces plus client-side load reports and process-local metrics.

## Architecture

~~~text
PDF upload → bounded extraction → stable chunk IDs → embeddings → ChromaDB
Question → embedding → top-k chunks → grounded answer model → answer + sources
Image upload → bounded image validation → vision request → image description
~~~

The Streamlit interface calls FastAPI. SQLite stores interaction history; it is not the vector database. Indexing a PDF and querying indexed text are separate operations, so a successful upload is not evidence that the answer is correct.

## Quickstart

Requires Python 3.11+.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
INGEST_API_TOKEN=local-owner-token GROQ_API_KEY=your-key uvicorn financial_assistant.api:app --reload
~~~

In a second terminal, start the UI:

~~~bash
streamlit run src/financial_assistant/app.py
~~~

Set ASSISTANT_API_URL if the UI and API do not share a host. The included sample policy is indexed on first initialization when the collection is empty. Without GROQ_API_KEY, API liveness remains available but provider-dependent answers are unavailable.

## API

- GET /health: liveness only.
- GET /ready: initializes dependencies; model initialization may be slow or memory-intensive.
- GET /metrics: process-local request counts, status codes, in-flight count and recent latency percentiles.
- POST /ingest: multipart field named file, PDF only, up to 10 MiB and 100 pages; requires X-Ingest-Token matching INGEST_API_TOKEN.
- POST /ask: accepts {"question":"What is the meal reimbursement limit?"}; returns answer and source chunks.
- POST /describe-image: accepts a validated PNG/JPEG data URL and a question.
- GET /docs: interactive API documentation.

Scanned PDFs are not OCR-processed. A PDF with no extractable text returns a validation error rather than creating an empty index. Image data URLs are validated as base64 and checked against the actual format and size limit.

## Tests

~~~bash
pytest
python -m compileall -q src scripts
~~~

CI uses deterministic fakes for provider calls, tests owner-token gating and ingestion, verifies stable chunk IDs and stale-chunk cleanup, and runs the real Chroma integration tests when installed. It does not require API secrets or incur provider cost.

## Reproducible reliability testing

See docs/RELIABILITY_LAB.md. Example:

~~~bash
python scripts/load_test.py --base-url http://127.0.0.1:8000 --path /ask --payload '{"question":"What is the meal reimbursement limit?"}' --requests 100 --concurrency 10 --output artifacts/ask-10.json
curl http://127.0.0.1:8000/metrics
~~~

The report captures client-observed throughput, p50/p95/p99 latency, error rate, HTTP status codes, transport failures and response bytes. It does not assert production users, model accuracy, or server resource usage. Record the commit, hosting plan, cold/warm state and request mix; retain failure and retest reports together.

## Deployment

The root render.yaml is a free Render API blueprint. Configure GROQ_API_KEY and INGEST_API_TOKEN as secrets in the Render dashboard; do not commit them. Deploy the Streamlit UI separately through Streamlit Community Cloud. Set ASSISTANT_API_URL in the UI service. PDF uploads require the owner-issued token at the UI, so a public user cannot alter the common index without it.

A free host may sleep after inactivity and has ephemeral storage, and the embedding stack may exceed a small instance's memory. Validate startup, first-question behavior and sustained load before sharing performance numbers. Chroma/SQLite local data may be lost on restart or spin-down.

## Security and limitations

The ingestion endpoint cannot be used without a configured owner token. When GROQ_API_KEY exists, /ask and /describe-image share a process-local default cap of 30 requests/minute (tunable with LLM_REQUESTS_PER_MINUTE); this is not distributed or user-specific rate limiting. Prompt injection is not fully solved. The interaction database retains the newest 10,000 question/answer records by default. It stores user-provided questions and generated answers, so use synthetic/demo documents and do not upload private bank statements or sensitive customer material to a public deployment.

Known limitations: retrieval quality depends on the embedding model and source document; no OCR for scanned PDFs; no tenant isolation; provider quota/availability affects answers; local Chroma/SQLite storage is not durable or horizontally shared; no measured production accuracy or capacity claim.

## License

MIT. See LICENSE.
