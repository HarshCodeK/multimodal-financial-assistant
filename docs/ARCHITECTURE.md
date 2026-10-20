# Architecture

## Goal

Provide a small local demo that answers questions about financial policies using retrieved document evidence and separately reads visible details from receipt or statement images.

## One-process design

Run one command: `python -m streamlit run src/financial_assistant/app.py`.

The Streamlit UI calls `FinancialAssistant` in `services.py` directly. There is no local HTTP hop, separate FastAPI process, API URL setting, or ingestion-token requirement.

## PDF RAG flow

1. PyMuPDF extracts selectable text from each PDF page, with input and extraction limits.
2. Each page becomes a `DocumentChunk` carrying source and page metadata.
3. Sentence Transformers embeds the text locally.
4. ChromaDB persists vectors and text chunks.
5. A user question is embedded and the closest chunks are retrieved.
6. Groq receives the question and retrieved evidence, and returns a structured supported/answer result.
7. Supported answers display their source chunks; unsupported answers do not display retrieved chunks as citations.

The included `data/sample_policy.txt` is indexed when the local Chroma collection is empty. Uploaded PDFs are stored in the local collection by filename; uploading the same filename replaces its old chunks.

## Image flow

The app validates and converts PNG/JPEG uploads to a data URL and sends them to a Groq vision model. The vision model may read receipt text and figures, but this is not a dedicated OCR engine and its output may contain mistakes. Image descriptions are not automatically inserted into the PDF RAG index.

Scanned PDFs are not OCR-processed in this version. A PDF without selectable text returns a clear error instead of silently indexing an empty document.

## Persistence

- **ChromaDB** stores indexed document text, embeddings, and source metadata.
- **SQLite** stores interaction records, latency, and structured PDF ingestion results. It is not the retrieval index.

## Grounding and limitations

The text model is instructed to answer only from retrieved evidence. Its JSON response is validated, with a conservative repair pass for common formatting errors. Missing fields are not invented, but this does not guarantee that every model answer is correct.

The project has no accounts, shared hosted index, tenant isolation, or production data governance. Keep API keys out of source control and use synthetic documents for demonstrations.
