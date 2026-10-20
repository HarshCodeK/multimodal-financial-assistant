# Multimodal Financial Assistant

A **local-only** financial document demo. It extracts text from PDFs and receipt images, splits documents into overlapping chunks, creates sentence embeddings, and retrieves relevant passages with ChromaDB. It does not call Groq or require an API key.

## What it does

- **PDF text extraction:** PyMuPDF extracts selectable text.
- **OCR fallback:** RapidOCR reads scanned PDF pages that have little/no text layer and extracts text from receipt/statement images.
- **Chunking:** text is split into 180-word chunks with 35-word overlap so context near chunk boundaries is retained.
- **Semantic retrieval:** Sentence Transformers creates local embeddings; ChromaDB stores chunks and retrieves semantically similar passages.
- **Policy questions:** questions are embedded and searched against the local ChromaDB index. The UI shows matching passages with source/page metadata; this version does not generate free-form LLM answers.
- **Receipt questions:** local OCR preserves recognized line boundaries. A small extractive matcher selects OCR lines that overlap with the question and shows the evidence. It abstains when no useful overlap is found; it is not a language model and cannot infer facts OCR failed to read.
- **SQLite logs:** records questions, results, elapsed time, and PDF indexing metadata.
- **One local process:** Streamlit calls Python services directly. No FastAPI server, Docker, API URL, ingestion token, Groq key, or cloud model calls.

## Run on Windows

Use Python 3.11 or newer. From the repository folder:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m streamlit run src/financial_assistant/app.py
~~~

Open http://localhost:8501. You do not need a second terminal.

The app itself should open without loading the embedding model. The first time you index a PDF or ask a question, Sentence Transformers downloads `all-MiniLM-L6-v2` if it is not cached; this can take a few minutes depending on the connection. The first OCR action may also initialize/download local OCR model assets. Subsequent use reuses the loaded models.

## Try it

1. Click **Download sample financial policy PDF**.
2. Upload the PDF and click **Index PDF**.
3. Ask `What is the maximum daily meal reimbursement?`
4. Inspect the displayed source passages and page numbers.
5. Upload a receipt image and click **Extract text from image**.

The bundled `data/sample_policy.txt` is also indexed automatically when the local index initializes.

## How it works

~~~text
PDF → PyMuPDF text extraction → OCR fallback for scanned pages → 180-word chunks / 35-word overlap
Question → local sentence embedding → ChromaDB semantic retrieval → relevant source passages
Receipt image → local RapidOCR → recognized text lines → question-term matching → relevant OCR evidence
Interaction metadata → SQLite
~~~

## Technologies

- Python
- Streamlit
- PyMuPDF
- RapidOCR (local OCR)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- SQLite

## Limitations

- Policy Q&A is semantic retrieval, not LLM generation: it returns relevant source passages, not a synthesized answer.
- Receipt Q&A uses lexical matching over OCR lines. It can miss paraphrases and must not be treated as verified financial data.
- OCR can confuse similar characters and numbers. Verify financial figures against the original.
- The embedding model and OCR runtime may need downloads on first use. If model downloads fail, the app shows an error when the relevant feature is used.
- This is a local single-user demo, not a production financial product or financial advice.

## Tests

~~~powershell
python -m pytest -q
~~~

The automated tests use fakes for OCR in unit tests; they do not guarantee OCR accuracy on every real receipt or scanned PDF.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md).

## License

MIT. See [LICENSE](LICENSE).
