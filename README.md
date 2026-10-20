# Multimodal Financial Assistant

A **local-only** financial document demo. It extracts text from PDFs and receipt images, splits documents into overlapping chunks, creates sentence embeddings, and retrieves relevant passages with ChromaDB. It does not call Groq or require an API key.

## What it does

- **PDF text extraction:** PyMuPDF extracts selectable text.
- **OCR fallback:** RapidOCR reads scanned PDF pages that have little/no text layer and extracts text from receipt/statement images.
- **Chunking:** text is split into 80-word chunks with 16-word overlap so context near chunk boundaries is retained.
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
python -m pip install -e ".[test]"
python -m streamlit run src/financial_assistant/app.py
~~~

Open http://localhost:8501. You do not need a second terminal.

The app itself should open without loading the embedding model. The first time you index a PDF or ask a question, Sentence Transformers downloads `all-MiniLM-L6-v2` if it is not cached; this can take a few minutes depending on the connection. The first OCR action may also initialize/download local OCR model assets. Subsequent use reuses the loaded models.

## Try it

1. Click **Download sample financial policy PDF**.
2. Upload the PDF and click **Index PDF**.
3. Ask `What is the maximum daily meal reimbursement?` in the question box below the conversation; submit further questions there. Previous questions and evidence remain visible.
4. Inspect the displayed source passages and page numbers.
5. Open **Receipt image · Q&A**, upload a receipt image, and click **Read image text**.
6. Review and correct any OCR mistakes in the editable recognized-text box, then ask `What is the total amount?` in the question box below the conversation. Previous questions remain visible while the same image is selected.

Only PDFs explicitly uploaded and indexed in the app are added to the policy retrieval collection. The repository's `data/sample_policy.txt` is not indexed or used as hidden evidence. Downloading the sample PDF also does not index it: upload the PDF and click **Index uploaded PDF** first. Short PDFs can legitimately produce one chunk; longer PDFs produce multiple 80-word chunks with 16-word overlap. The app uses a fresh collection name to avoid mixing in records from earlier versions that automatically indexed bundled sample text.

## How it works

~~~text
PDF → PyMuPDF text extraction → OCR fallback for scanned pages → 80-word chunks / 16-word overlap
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
- Receipt Q&A uses lexical matching over OCR line windows and common price/amount synonyms. It can still miss paraphrases or depend on OCR accuracy; treat matches as evidence to verify, not verified financial data. The OCR text is editable so users can correct misread digits before asking questions.
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
