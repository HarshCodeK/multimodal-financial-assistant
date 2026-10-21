# Financial Assistant

A Streamlit demo with two workflows. Both use the same Groq text model, `openai/gpt-oss-20b`; **no vision model is configured or called**.

- **PDF RAG:** extract text locally, chunk it, embed it locally, retrieve evidence from ChromaDB, then send the question and retrieved passages to GPT-OSS-20B.
- **Image Q&A:** run OCR locally, show the extracted text, then send only that text and the question to GPT-OSS-20B. The raw image is not sent to Groq.
- **Scanned PDFs:** selectable text is extracted normally; pages without text are rendered locally and passed through OCR as a fallback.

The app does not train or fine-tune either the embedding model or LLM on uploaded material.

## Run on Windows

Use Python 3.11 or newer, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
notepad .env
```

Set your valid Groq key in the project-root `.env`:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

A usable key in this project's `.env` takes precedence over inherited Windows environment variables. Never commit or share `.env`.

Start the app:

```powershell
python -m streamlit run src/financial_assistant/app.py
```

## PDF RAG

1. Open **PDF RAG Explorer**.
2. Upload a PDF and click **Extract, chunk, embed & index PDF**.
3. Selectable text is extracted page by page. Pages without selectable text use local OCR.
4. Text is split into 80-word chunks with a 16-word overlap.
5. Sentence Transformers `all-MiniLM-L6-v2` generates 384-dimensional vectors; ChromaDB stores chunks, vectors, and source/page metadata locally.
6. Ask a question. Up to five retrieved passages are sent with the question to GPT-OSS-20B. The UI shows the exact evidence and citation IDs before the answer.

A successful new indexing operation replaces the previous PDF index. The bundled fictional sample is never indexed automatically.

## Image OCR + text LLM

1. Open **OCR image Q&A** and upload a PNG, JPEG, or WebP image (up to 5 MiB).
2. Click **Run local OCR and answer**.
3. OCR runs on your computer using RapidOCR. Its extracted text and an approximate confidence indicator are displayed.
4. Only the question and extracted text are sent to Groq's `openai/gpt-oss-20b` text endpoint. The image bytes are not included in the API request.
5. Verify the OCR output against the image, especially financial amounts, dates, and identifiers.

OCR is suitable for legible text in receipts, forms, statements, and screenshots. It does not understand charts, visual relationships, or layouts the way a vision model can, and handwriting/blurry images can be misread. If OCR finds no readable text, the app asks for a clearer image rather than guessing.

RapidOCR's model files may be downloaded on first use and cached locally. PDF extraction, OCR, embeddings and retrieval run locally. PDF questions + retrieved excerpts and image questions + OCR text are sent to Groq. Do not upload sensitive material unless this transfer is acceptable.

## Evidence, logging, and limitations

- The PDF answer UI displays retrieved passages, source filename, page, cosine distance, citation IDs, and the prompt/context for inspection.
- The image UI displays extracted OCR text before the text-model answer.
- A local SQLite usage log records questions/answers, workflow, model, token usage, latency, input lengths, and estimated API cost. It does not store API keys or uploaded image bytes; it does store question/answer text.
- Cost uses default GPT-OSS-20B rates configured in `src/financial_assistant/costing.py`. It is an estimate, not an invoice or account-credit balance.
- Citation and OCR confidence are not guarantees of correctness. Verify important information against the source.
- This is an educational single-user demo, not financial advice or a production financial system.

## Tests

```powershell
python -m pytest -q
```

Tests mock OCR and Groq boundaries and do not require a live API request. Manual smoke testing is still needed for OCR quality and a real Groq response.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md), and [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## License

MIT. See [LICENSE](LICENSE).
