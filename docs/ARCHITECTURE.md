# Architecture

## Design goal

The app demonstrates two explicit workflows in one Streamlit interface: PDF Retrieval-Augmented Generation (RAG) with inspectable evidence, and direct image question answering through a vision-capable model. The uploaders are on the main screen. The workflows do not share an index.

## Runtime and interface

Run `python -m streamlit run src/financial_assistant/app.py`. Streamlit calls Python services directly; there is no FastAPI service or separate vector database server.

- **PDF RAG Explorer tab:** PDF uploader, explicit indexing action, indexing report, retrieval-grounded chat and inspection panels.
- **Ask about an image tab:** image uploader, question field, analyze action and response.
- **Sidebar:** architecture summary and privacy/data-flow notes only; primary upload controls are not hidden there.

## PDF indexing

1. The user explicitly uploads a text-based PDF and clicks the indexing button.
2. PyMuPDF extracts selectable text page by page. Page numbers are attached to chunks; scanned/image-only PDFs with no text are rejected.
3. Text is divided into 80-word chunks with 16-word overlap. Chunking is page-local.
4. `sentence-transformers/all-MiniLM-L6-v2` creates one 384-dimensional vector per chunk. It is pretrained; the PDF does not train or fine-tune it.
5. ChromaDB persists each chunk's vector, text and source/page metadata in the dedicated `financial_assistant_pdf_rag_v4` collection.
6. A successful indexing operation replaces the previous indexed PDF rather than appending another source. Extraction and embedding happen before the existing collection is cleared.

The UI displays chunk text and boundaries, model name, vector dimensions, and the first few vector values. Individual vector dimensions are not human-readable labels.

## PDF question answering

1. The question is embedded with the same model used for document chunks.
2. ChromaDB returns up to five nearest passages and cosine distances.
3. The generator labels those passages `S1`, `S2`, etc. and sends the question and evidence to the configured Groq-hosted text model (default `openai/gpt-oss-20b`).
4. The LLM is instructed to use only the evidence, return structured JSON, answer concisely and provide source IDs. The application discards invalid citation IDs and does not mark an answer supported unless the model claims support and cites at least one valid ID.
5. The UI displays the answer, cited/uncited retrieved passages, source pages, distances, prompt/context, and provider token usage when available.

These are practical guardrails, not a mathematical guarantee against hallucinations. A model can misinterpret evidence or cite a passage that does not fully justify its answer.

## Image question answering

1. The user uploads a PNG, JPEG, or WebP image and enters a question on the main screen.
2. The service checks the byte signature and rejects empty, unsupported, or larger-than-5-MiB image data.
3. The image bytes are base64-encoded into a data URL and sent with the question to Groq's vision-capable chat-completions API. The default model is `qwen/qwen3.6-27b`, configurable through `GROQ_VISION_MODEL`.
4. The response text and provider input/output token usage (if supplied) are displayed.
5. The image is not OCR-indexed locally, is not persisted to ChromaDB, and does not replace or modify the PDF index. Each image request is direct visual question answering, not image RAG.

The system prompt tells the model not to follow instructions visible inside an image and not to guess unreadable financial values. This is not a guarantee; users must verify dates, amounts, and identifiers against the original.

## Token and cost observability

The PDF tab reports provider input/output token counts when returned by Groq. It also shows rough word-based estimates for document and retrieved-context size; these are not tokenizer measurements and are not a measured percentage saving. RAG may reduce repeated context for large documents, but embedding/indexing has overhead and retrieval can miss evidence. Image Q&A sends image plus question directly to the vision model and is not the token-saving RAG path.

## Data boundaries and persistence

- PyMuPDF, chunking, Sentence Transformers and ChromaDB run locally.
- PDF questions and retrieved text passages are sent to Groq for answer generation.
- Image bytes and the image question are sent to Groq for vision analysis.
- The uploaded PDF/image is not used to train or fine-tune either model by this app.
- No bundled sample PDF or text is automatically indexed.
- ChromaDB stores PDF chunks, vectors and source/page metadata.
- SQLite stores local interaction and indexing metadata.
- The Groq key is read from a local `.env` file, which must not be committed.

## Main trade-offs and limitations

- PDF chunking is word-based and can split sentences.
- The embedding model is English-oriented and can miss exact identifiers, numbers or unusual terminology.
- Top-k retrieval can omit relevant passages; cosine distance is not proof of support.
- LLM answers can still be incorrect despite prompt constraints and citation validation.
- PDF ingestion supports selectable text only; OCR must be performed before upload for scanned PDFs.
- Image Q&A depends on Groq network/API access and may misread small or blurry text.
- Uploaded images are sent to the configured provider; do not use confidential files unless the data transfer is acceptable.
- This is a single-user educational app, not a production financial system.
