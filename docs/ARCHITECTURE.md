# Architecture

## Design goal

A small local tool that extracts text from financial PDFs and receipt images, chunks document text, embeds it, and retrieves relevant evidence without requiring an API key or separate server.

## One-process flow

Run `python -m streamlit run src/financial_assistant/app.py`. The Streamlit UI calls local Python services directly.

## PDF ingestion and RAG

1. PyMuPDF extracts selectable text page by page.
2. Pages with little or no text are rendered to images and passed through local RapidOCR.
3. Text is split into 180-word chunks with 35-word overlap.
4. Sentence Transformers embeds the chunks locally.
5. ChromaDB persists vectors and source/page metadata.
6. A question is embedded and the closest chunks are retrieved.
7. The app displays retrieved passages, source/page metadata, and cosine distance. This is retrieval with extractive evidence display—not free-form LLM generation.

The embedding model is initialized only after the user first indexes a PDF or asks a question, so the UI can render without waiting for model loading.

## Image OCR

Uploaded images are validated with Pillow and passed to RapidOCR locally. Recognized text lines are preserved and displayed. The user can then ask a question; a small lexical matcher ranks lines by overlap with question terms and displays the matching OCR lines as evidence. It abstains when there is no meaningful overlap. This is deliberately transparent extractive matching, not an LLM and not semantic retrieval. No image is sent to a remote vision API. OCR is text recognition, not general visual reasoning; verify totals and dates.

## Persistence

- ChromaDB stores chunks, embeddings, and source metadata.
- SQLite stores interaction records, latency, and PDF extraction metadata.

## Trade-offs

This version intentionally removes the Groq client, external API calls, FastAPI, Docker, and deployment configuration. It is simpler to run and does not require credentials, but it does not generate free-form LLM answers. It returns retrieved evidence. OCR and embedding models may need initial downloads. The application is a local single-user demo.
