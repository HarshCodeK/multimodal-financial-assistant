# Multimodal Financial Assistant

An educational Streamlit app with two distinct workflows:

- **PDF RAG Explorer:** upload one text-based PDF, inspect extraction/chunking/embeddings/retrieval, and ask questions answered from retrieved passages with source IDs.
- **Image Q&A:** upload a PNG, JPEG, or WebP image and ask a vision-capable Groq model about its visible contents.

Both uploaders are on the **main screen**. The sidebar contains workflow and privacy information, not the primary upload controls.

This project does **not** train an LLM or embedding model on your PDF or image.

## What it demonstrates

### PDF RAG workflow

- **PDF extraction — PyMuPDF:** reads selectable text page by page and keeps page numbers. This path does not run OCR; scanned/image-only PDFs must be OCR-processed before upload.
- **Chunking:** splits each page into 80-word chunks with a 16-word overlap.
- **Embeddings — Sentence Transformers:** the pretrained `all-MiniLM-L6-v2` model maps each chunk to a 384-dimensional vector. The vector is a learned representation useful for semantic similarity, not a summary.
- **Vector storage/retrieval — ChromaDB:** stores chunk text, vectors and source/page metadata. The question is embedded with the same model and nearby vectors are retrieved. Cosine distance is a ranking signal, not proof of correctness.
- **Answer generation — Groq LLM API:** the question and retrieved passages are sent to the configured text model. The model is instructed to answer from the passages and cite source IDs; the app validates citation IDs.
- **Inspection and usage accounting:** the UI shows extracted chunks, a sample vector, retrieved passages, distances, citations, prompt/context, provider token usage when available, and estimated API cost based on published token rates.
- **Local usage ledger:** successful PDF/image questions record model, input/output tokens, latency, approximate request lengths, and estimated USD cost in local SQLite. It never stores API keys or image bytes, but it does store question/answer text.

### Image Q&A workflow

- **Main-screen image uploader:** accepts PNG, JPEG, and WebP images up to 5 MiB.
- **Vision model:** sends the image and the user's question to the configured Groq vision model (default: `qwen/qwen3.6-27b`) and displays its response.
- **Cost comparison:** compares direct vision inference with a hypothetical cloud OCR + text-LLM path. Input/output token prices, OCR price, extracted-text size, and output length are editable. This is an estimate, not an OCR implementation or billing API.
- **Safety/accuracy:** malformed or oversized files are rejected; the model is instructed not to guess unreadable values and to flag uncertainty. Financial amounts and dates still need verification against the original image.
- **Separate from RAG:** image content is not added to ChromaDB, does not replace the PDF index, and is not persisted as an indexed source. This is direct visual question answering, not local OCR or image-RAG.

## PDF RAG pipeline

```text
Upload PDF and explicitly click Index
  → PyMuPDF extracts selectable text and page numbers
  → 80-word chunks with 16-word overlap
  → Sentence Transformer creates a 384-dimensional vector per chunk
  → ChromaDB stores vectors + text + source/page metadata

Ask a question
  → embed the question with the same model
  → retrieve up to five passages
  → send question + retrieved passages to Groq
  → display answer, citations, source pages, distances and token usage
```

## Image Q&A pipeline

```text
Upload PNG/JPEG/WebP + enter a question
  → validate file signature and 5 MiB size limit
  → encode image as a data URL
  → send image + question to Groq vision model
  → display response and provider token usage when available
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
GROQ_MODEL=openai/gpt-oss-20b
GROQ_VISION_MODEL=qwen/qwen3.6-27b
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Do not commit or share `.env`. Then run:

```powershell
python -m streamlit run src/financial_assistant/app.py
```

Open http://localhost:8501. The first PDF indexing action downloads the embedding model if it is not already cached. Both LLM workflows require a valid Groq API key and network access. PDF extraction/indexing can work locally without the key, but answers cannot be generated.

If the UI says HTTP 401 / invalid_api_key despite a key being present, open the Groq API configuration & troubleshooting panel near the top. The app shows whether the project .env and a key are present without revealing the key. A pre-existing Windows/user/system GROQ_API_KEY environment variable takes precedence over .env; remove or update a stale value, then restart Streamlit. The app trims accidental surrounding whitespace and matching quotes. A key being present does not prove Groq accepts it.

## Try it

### Test PDF RAG

1. Open the **PDF RAG Explorer** tab on the main screen.
2. Upload a text-based PDF, or download the fictional sample PDF and then upload it. Downloading the sample does not index it.
3. Click **Extract, chunk, embed & index PDF**. A successful new index replaces the previously indexed PDF.
4. Inspect the chunk boundaries, embedding model/dimensions, and vector preview.
5. Ask a question such as `What is the maximum meal reimbursement?`
6. Inspect retrieved passages, ChromaDB distances, cited source IDs, and the prompt/context sent to Groq.

### Test image Q&A

1. Open the **Ask about an image** tab on the main screen.
2. Upload a PNG, JPEG, or WebP image.
3. Ask a specific question, such as `Read the total amount and date. If either is unclear, say so.`
4. Click **Analyze image** and verify the result against the original image.

The sample PDF is fictional and is **not** indexed automatically. No bundled text is silently indexed.

## What each component does

| Component | Responsibility | Does not do |
|---|---|---|
| PyMuPDF | Extracts selectable PDF text and page numbers | Does not OCR scanned PDFs or answer questions |
| Chunking | Divides PDF text into overlapping passages | Does not summarize or train a model |
| Sentence Transformers | Maps PDF chunks and questions to semantic vectors | Does not store the index or generate answers |
| ChromaDB | Persists PDF vectors/text/metadata and retrieves nearby vectors | Does not prove that a passage supports an answer |
| Text LLM on Groq | Drafts answers from retrieved PDF passages | Is not trained on the uploaded PDF; can still make mistakes |
| Vision model on Groq | Answers questions about the uploaded image | Does not add the image to the PDF vector index |
| Streamlit | Presents upload, question and inspection workflows | Does not perform embedding or retrieval itself |
| SQLite | Stores local interaction/index metadata | Is not the vector database |

## Token, cost, and length logging

The local SQLite usage log records workflow, model, provider-reported input/output tokens (when available), latency, question/prompt length estimates, and estimated API cost. Inspect recent rows in the app or download them as JSON. The estimate uses published rates configured in the costing module; it is not the account's actual invoice, balance, or remaining credits. It excludes local compute/energy, embedding-model hosting costs, taxes, and provider-specific billing nuances.

### Direct vision vs OCR + text LLM

The image tab includes an editable comparison calculator. Default public rates referenced when this code was updated:

- Groq Qwen3.6-27B: $0.60 per 1M input tokens and $3.00 per 1M output tokens. Groq documents one image as 2,048 input tokens for this model.
- Groq GPT-OSS-20B: $0.075 per 1M input tokens and $0.30 per 1M output tokens.
- Google Cloud Vision text detection: $1.50 per 1,000 images after the first 1,000 free monthly units.

Sources: [Groq vision documentation](https://console.groq.com/docs/vision), [Groq supported-model pricing](https://console.groq.com/docs/models), and [Google Cloud Vision pricing](https://cloud.google.com/vision/pricing).

The comparison estimates direct vision from provider-reported tokens when available; otherwise it uses 2,048 image tokens plus rough question/system overhead and an assumed answer length. The OCR + text LLM path estimates OCR cost plus extracted-text tokens, prompt overhead, and answer tokens.

The OCR route is a hypothetical cost comparison only; this app does not call a cloud OCR service. You can set OCR cost to zero to model local OCR, but account for local compute and engineering/quality costs separately. Vision can retain layout/chart context; OCR + text can be cheaper for text-heavy images but may lose spatial structure. Actual costs vary with rates, free-tier usage, image dimensions/model tokenization, and actual prompt/output. No fixed winner is claimed.

### What do “Approx. document tokens” and “Vector dimensions 384” mean?

- **Approx. document tokens:** a rough estimate of full PDF text size using word count × 1.33. It is not exact tokenization and not the number sent on each question. With RAG, only retrieved passages plus the prompt/question are sent per answer.
- **Vector dimensions 384:** the embedding model represents each text chunk as 384 floating-point coordinates. ChromaDB compares these vectors with the question vector to retrieve semantically similar text. The 384 values are not words, model parameters, or a confidence percentage. A higher dimension is not automatically better; model quality, retrieval, chunking and evaluation matter.

## Data handling and security

- PDF extraction, chunking, embeddings and ChromaDB retrieval run locally.
- PDF questions and retrieved PDF excerpts are sent to Groq for text answer generation.
- The uploaded image and image question are sent to Groq for visual analysis.
- Do not upload confidential material unless the relevant data transfer is acceptable for your requirements.
- The PDF and image are not used by this app to train or fine-tune either model.
- Only one PDF is indexed at a time. A successful index replaces the prior PDF index; image Q&A does not alter that index.
- PDF and image uploads have size limits. PDF extraction also limits page count and extracted characters.
- Model outputs and citations are not guaranteed to be correct. Verify generated answers against source material.

## Tests

```powershell
python -m pytest -q
```

Tests mock the Groq boundaries and do not require a live API call. A passing unit suite does not guarantee answer quality on every file. UI layout and a live provider request still need manual smoke testing with a valid API key.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md), and [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## License

MIT. See [LICENSE](LICENSE).
