# Multimodal Financial Assistant

A small, evidence-grounded assistant for financial-policy documents. Index a text-based PDF, ask a question against retrieved document chunks, inspect returned source evidence, and separately use vision inference to summarize a statement or chart image.

This is a portfolio system, not financial advice software. It does not execute transactions or guarantee error-free financial interpretation.

## What it demonstrates

- Bounded PDF upload and page-level text extraction with source provenance.
- Stable chunk IDs so re-indexing identical chunks is idempotent.
- ChromaDB retrieval with local sentence-transformer embeddings.
- Grounded LLM answers with an explicit unsupported-answer path.
- Separate image-description endpoint with bounded image-data URL input.
- FastAPI and Streamlit interfaces.
- SQLite interaction logging with WAL and a writer wait timeout.
- Process-local request metrics, CI tests, Docker support and a free-host demo blueprint.

## Architecture

~~~text
PDF upload → bounded extraction → stable chunk IDs → embeddings → ChromaDB
Question → embedding → top-k retrieved chunks → grounded answer model → answer + sources
Image upload → bounded vision request → image description
~~~

The Streamlit interface calls FastAPI. SQLite stores interaction history; it is not the vector database. Indexing a PDF and querying indexed text are separate operations, so a successful upload is not evidence that the LLM answered correctly.

## Quickstart

Requires Python 3.11+.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn financial_assistant.api:app --reload
~~~

In another terminal, start the UI:

~~~bash
streamlit run src/financial_assistant/app.py
~~~

Set ASSISTANT_API_URL if the UI and API do not share a host. Set GROQ_API_KEY for LLM and vision answers. Without that key, the API can remain live but provider-dependent answers are unavailable. The sample policy is indexed on first initialization when the collection is empty.

## API

- GET /health: liveness only.
- GET /ready: initializes dependencies and reports readiness; model initialization may be slow or memory-intensive.
- GET /metrics: process-local route request counts, status codes, in-flight count and recent latency percentiles.
- POST /ingest: multipart field named file, PDF only, up to 10 MiB and 100 pages; extracts a bounded amount of text.
- POST /ask: accepts {"question":"What is the meal reimbursement limit?"}; returns answer and source chunks.
- POST /describe-image: accepts a PNG/JPEG data URL and a question for the configured vision model.
- GET /docs: interactive API documentation.

Scanned PDFs are not OCR-processed. A PDF with no extractable text receives a validation error rather than silently creating an empty index.

## Run tests

~~~bash
pytest
~~~

The real Chroma integration test runs when dependencies are installed. External Groq calls are not part of CI; tests use deterministic fakes so CI does not require secrets or incur provider cost.

## Reproducible reliability testing

See docs/RELIABILITY_LAB.md. Example:

~~~bash
python scripts/load_test.py --base-url http://127.0.0.1:8000 --path /ask --payload '{"question":"What is the meal reimbursement limit?"}' --requests 100 --concurrency 10 --output artifacts/ask-10.json
curl http://127.0.0.1:8000/metrics
~~~

The report captures client-observed throughput, p50/p95/p99 latency, error rate, HTTP status codes, transport failures and response bytes. It does not assert production users, model accuracy, or server resource usage. Record the commit, hosting plan, cold/warm state and request mix. Keep failure and retest results together.

## Deployment

The root render.yaml is a free Render API demo blueprint. The Streamlit UI can be deployed separately through Streamlit Community Cloud, with ASSISTANT_API_URL pointing at the API URL. A free host may sleep after inactivity and has ephemeral storage, and the embedding stack may exceed a small instance's memory. Validate startup, first-question behavior and sustained load before sharing performance numbers.

## Security and limitations

PDF bytes, page count and extracted text are bounded; filenames are reduced to their basename before source storage. Prompt injection is not fully solved. The interaction database stores questions and answers, so use synthetic/demo documents and do not upload private bank statements or sensitive customer material to a public deployment.

Known limitations: retrieval quality depends on the embedding model and indexed document; no OCR for scanned PDFs; no document access control or tenant isolation; provider quota/availability affects answers; local Chroma/SQLite storage is not durable or horizontally shared; no measured production accuracy or capacity claim.

## License

MIT. See LICENSE.
