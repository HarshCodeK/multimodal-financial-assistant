# Architecture

## Design goal

A single Streamlit app demonstrates evidence-first PDF Retrieval-Augmented Generation and local OCR for image text. Both workflows use the same Groq text model, `openai/gpt-oss-20b`. There is no vision-model route.

## Interface

Run `python -m streamlit run src/financial_assistant/app.py`.

- **PDF RAG Explorer:** upload/index a PDF, inspect chunks/embeddings, ask questions, and review retrieved evidence before answers.
- **OCR image Q&A:** upload an image, run local OCR, review extracted text, then ask GPT-OSS-20B about that text.
- The left sidebar is removed. Configuration troubleshooting and privacy details live in the main page.

## PDF ingestion and retrieval

1. PyMuPDF extracts selectable text page by page.
2. Pages with no selectable text are rendered locally and passed through RapidOCR; extracted text remains associated with its PDF page.
3. Text is split into 80-word chunks with 16-word overlap.
4. `all-MiniLM-L6-v2` generates 384-dimensional embeddings locally.
5. ChromaDB stores vectors, chunk text and source/page metadata in `financial_assistant_pdf_rag_v5`.
6. Questions are embedded with the same model; up to five passages are retrieved and sent with the question to GPT-OSS-20B.
7. The UI displays the exact retrieved passages, source/page/distance and citations before the answer. Invalid citation IDs are discarded and unsupported answers abstain.

A successful indexing action replaces the active PDF. No bundled document is indexed automatically.

## Image OCR and answering

1. Validate uploaded PNG/JPEG/WebP signature and size (5 MiB maximum).
2. Decode the image locally and run RapidOCR (ONNX Runtime). OCR model files may download on first use and are cached locally.
3. Display OCR text and mean line-confidence estimate to the user.
4. Send only the user's question and extracted text to the same Groq text-model generator used by PDF RAG. The raw image is not sent to Groq, embedded, or stored in ChromaDB.
5. Display the answer, citations to the OCR text, token usage and latency when available.

This works for readable text, not chart comprehension or visual-spatial questions. OCR confidence is not a guarantee of correctness; verify amounts and dates.

## API-key configuration

The app reads the project-root `.env` explicitly. A usable key there is assigned to the process environment and takes precedence over stale inherited Windows/user/system values. If no usable project key exists, the app falls back to ordinary environment loading. Keys are never displayed or logged.

## Data handling and observability

- PDF text extraction, scanned-page OCR, image OCR, embeddings and retrieval run locally.
- PDF questions and retrieved PDF excerpts are sent to Groq.
- Image questions and OCR-extracted text are sent to Groq; the image bytes are not.
- SQLite stores question/answer text, workflow, model, latency, token counts and estimated cost. It does not store API keys or uploaded image bytes.
- Cost is an estimate from configured text-model token rates, not a bill or account balance.
- The app does not train/fine-tune models on uploaded documents.

## Limitations

- OCR can misread low-resolution text, handwriting, unusual fonts and dense layouts.
- Text-only models cannot infer visual details absent from OCR text.
- Chunking can split sentences; retrieval can miss evidence; citations do not mathematically guarantee correctness.
- This is a single-user educational demo, not a production financial system.
