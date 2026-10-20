# PDF RAG Explorer

A small, explainable **PDF-only Retrieval-Augmented Generation (RAG)** demo. Upload one text-based PDF, inspect how its text becomes chunks and vectors, retrieve relevant evidence for a question, and ask an LLM to produce a concise answer with source references.

This project does **not** train an LLM or embedding model on your PDF. It indexes the uploaded document for retrieval.

## What it demonstrates

- **PDF extraction — PyMuPDF:** reads selectable text page by page and keeps page numbers. This simplified version does not include receipt OCR or scanned-page OCR. Image-only/scanned PDFs must be OCR-processed before upload.
- **Chunking:** splits each page into 80-word chunks with a 16-word overlap. Overlap reduces the risk that a rule near a chunk boundary loses surrounding context. Short PDFs can legitimately become one chunk.
- **Embeddings — Sentence Transformers:** the pretrained `all-MiniLM-L6-v2` model converts each chunk into a 384-dimensional numeric vector. The vector is a learned representation useful for semantic similarity, not a summary of the chunk.
- **Vector storage/retrieval — ChromaDB:** stores chunk text, vectors and source/page metadata. The question is embedded with the same model and ChromaDB returns nearby vectors. Cosine distance is a ranking signal, not a guarantee that a passage answers the question.
- **Answer generation — Groq LLM API:** the question and retrieved passages are sent to the configured LLM. It must answer only from those passages and return source IDs. If the passages do not support an answer, the app should abstain.
- **Inspection:** the UI shows extracted chunks, a sample vector, retrieved passages, distances, citations, the prompt/context used, and token usage returned by the provider.
- **SQLite:** stores minimal interaction and indexing metadata locally.

## Pipeline

```text
User uploads one PDF and clicks Index
  → PyMuPDF extracts selectable text and page numbers
  → 80-word chunks are created with 16-word overlap
  → Sentence Transformer converts each chunk to a 384-number embedding
  → ChromaDB stores vectors + original chunk text + source/page metadata

User asks a question
  → the same embedding model converts the question to a vector
  → ChromaDB retrieves the closest passages
  → question + retrieved passages are sent to Groq
  → LLM returns a concise answer and cited evidence IDs
  → UI displays answer, source pages, retrieval distances, prompt and token usage
```

## Run on Windows

Use Python 3.11 or newer. From the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
notepad .env
```

Add your own Groq API key to `.env`:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Do not commit or share `.env`. The repository ignores it. Then run:

```powershell
python -m streamlit run src/financial_assistant/app.py
```

Open http://localhost:8501. The first indexing action downloads the embedding model if it is not already cached. You need a Groq API key to generate answers. Without it, local PDF extraction and indexing can still work, but answer generation will report a configuration error.

## Try it

1. In the sidebar, upload a text-based PDF. You can use the fictional example in `data/sample_financial_policy.pdf`.
2. Click **Extract, chunk, embed & index PDF**. This replaces the previously indexed PDF; it does not append another document.
3. Open the indexing report and inspect the chunks, overlap, model name, vector dimensions and sample vector values.
4. Ask a question such as `What is the maximum meal reimbursement?`
5. Inspect retrieved passages, ChromaDB distances, cited source IDs, and the exact prompt/context sent to Groq.
6. Ask something absent from the document, such as `What is the international travel allowance?`, and check whether the system abstains instead of guessing.

The sample PDF is fictional and is **not** indexed automatically. No bundled text file is silently indexed. Only an explicit upload and indexing action changes the vector index.

## What the tools actually do

| Component | Responsibility | Does not do |
|---|---|---|
| PyMuPDF | Extracts selectable PDF text and page numbers | Does not understand the document or answer questions |
| Chunking | Divides text into manageable overlapping passages | Does not summarize or train a model |
| Sentence Transformers | Maps text to numeric semantic vectors | Does not store the PDF or generate the final answer |
| ChromaDB | Persists vectors/text/metadata and finds nearby vectors | Does not know whether a retrieved passage truly answers the question |
| Retriever | Selects the top passages for the question | Does not write a natural-language answer |
| Groq-hosted LLM | Reads the question and retrieved evidence to draft an answer | Is not trained on the uploaded PDF by this app; it can still make mistakes |
| Streamlit | Presents the workflow and inspection panels | Does not perform the embedding or retrieval itself |
| SQLite | Stores interaction/index metadata locally | Is not the vector database |

## RAG and token cost: what we can claim

Without retrieval, a simple document-Q&A implementation might send the entire document to the LLM on every question. RAG indexes the document once and sends only a few retrieved passages for each question. That can reduce repeated input context and cost for sufficiently large documents.

There are trade-offs: indexing requires extraction and embedding work, retrieval can miss relevant passages, and short documents may not benefit much. The app shows rough word-based token estimates for document/context size and, when returned by Groq, actual input/output token counts for the generation call. **Word-based estimates are not actual tokenizer measurements, and this project does not claim a fixed percentage of cost savings.**

## Data handling and security

- PDF extraction, chunking, embeddings and ChromaDB retrieval run locally.
- The question and retrieved PDF excerpts are sent to Groq for answer generation. Do not use confidential documents unless that data transfer is acceptable under your requirements.
- The uploaded PDF is not used to fine-tune or train the embedding model or LLM.
- The app indexes one PDF at a time; a successful index replaces the previous index in the app's dedicated Chroma collection.
- The app has file size, page-count and extracted-text limits in the PDF extraction code.
- LLM answers and citations must still be checked against the displayed source passages. Prompting and citation validation reduce risk but do not guarantee correctness.

## Tests

```powershell
python -m pytest -q
```

Tests use fakes for the embedding/retrieval/LLM boundaries and do not require a live Groq API call. A passing unit suite does not guarantee answer quality on every PDF.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md).

## License

MIT. See [LICENSE](LICENSE).
