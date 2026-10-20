# Multimodal Financial Assistant

A locally runnable portfolio demo that answers questions about financial-policy documents using retrieval-augmented generation (RAG), and analyzes receipt or statement images with a vision-capable Groq model.

## Features

- **PDF ingestion:** extract selectable text page by page, then index it in ChromaDB.
- **RAG:** create local sentence-transformer embeddings, retrieve relevant chunks, and ask Groq to answer from those passages.
- **Evidence display:** show the source document and page text used for supported answers.
- **Image analysis:** send a validated PNG/JPEG to a Groq vision model to read visible financial information.
- **SQLite logging:** record questions, answers, elapsed time, and structured PDF-ingestion results.
- **Conservative JSON repair:** handle common model formatting issues without inventing missing fields.
- **Local by default:** Streamlit calls the service layer directly. No separate FastAPI server, API URL, or ingestion token is required.

This is a learning/portfolio demo, not financial advice software. It does not execute transactions or guarantee correct interpretation.

## Quickstart (Windows PowerShell)

Use Python 3.11 or newer. From the project root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
$env:GROQ_API_KEY="your-groq-api-key"
python -m streamlit run src/financial_assistant/app.py
~~~

The browser preview opens at http://localhost:8501. Keep only this one terminal running.

For macOS/Linux, activate the virtual environment with `. .venv/bin/activate`, install with `pip install -e .`, set `GROQ_API_KEY` in the shell, and run the same Streamlit command.

The first launch downloads the configured sentence-transformer model and creates a local Chroma collection. On an empty collection, `data/sample_policy.txt` is indexed automatically. The repository also includes [`data/sample_financial_policy.pdf`](data/sample_financial_policy.pdf), a fictional PDF you can upload through the UI to test the full ingestion flow. Re-uploading a PDF with the same filename replaces that source's indexed chunks.

## Try it

Ask about the built-in sample policy:
- What is the meal reimbursement limit?
- What documentation is required for a claim?
- How soon must employees submit expenses?
- What does the policy say about traffic fines?
- Does the policy specify an international travel allowance?

Then upload your own text-based financial policy PDF and ask questions whose answers are present in it. Upload a fictional receipt image and select **Inspect image** to test vision analysis.

## How it works

~~~text
PDF → PyMuPDF text extraction → sentence-transformer embeddings → ChromaDB
Question → retrieve relevant chunks → Groq LLM → grounded answer + evidence
Receipt/image → image validation → Groq vision model → description of visible details
Interactions → SQLite (question/answer, latency, extraction metadata)
~~~

The app and service layer run in the same Streamlit process. SQLite stores interaction logs; ChromaDB stores searchable document vectors. PDF and image paths are deliberately separate: image descriptions are not automatically added to the PDF retrieval index.

## Configuration and local files

| Setting | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | unset | Required for LLM answers and image analysis |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Text-answer model |
| `GROQ_VISION_MODEL` | `qwen/qwen3.8-27b` | Image-analysis model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_DIR` | `.chroma` | Local vector database directory |
| `SQLITE_PATH` | `data/interactions.db` | Interaction log database |

Do not commit API keys or real customer/financial documents. The local Chroma index and SQLite log persist between launches and are ignored by Git.

## Limitations

- **Scanned PDFs are not OCR-processed.** PDFs need selectable text. Image uploads are read by a vision model, which can still misread digits or totals.
- The vision model's output is not a deterministic OCR transcript; verify important figures against the image.
- Retrieval quality depends on the embedding model and the indexed documents. The model can still make mistakes despite grounding instructions.
- No user accounts, multi-tenant isolation, shared cloud index, or production data governance.
- Provider availability and quota affect answers. Groq API usage may incur limits or costs.
- Not financial advice and not suitable for making decisions without human verification.

## Tests

Install the project dependencies, then run:

~~~powershell
python -m pytest -q
~~~

Tests cover PDF extraction, vector retrieval, JSON repair, SQLite logging, and the local service workflow using fakes. They do not prove live Groq model accuracy or live API availability.

## Interview explanation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow and [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md) for concise interview answers.

## License

MIT. See [LICENSE](LICENSE).
