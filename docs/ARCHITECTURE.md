# Architecture

## Design goal

Demonstrate the full PDF RAG path with as few moving parts as practical: local text extraction, transparent chunking, pretrained embeddings, vector retrieval, grounded LLM answer generation, and visible evidence. This is a single-user educational application, not a production document platform.

## Runtime

Run `python -m streamlit run src/financial_assistant/app.py`. Streamlit calls the Python services directly. There is no FastAPI service or separate vector database server.

## Indexing

1. The user explicitly uploads a text-based PDF and clicks the indexing button.
2. PyMuPDF extracts selectable text page by page. Page numbers are attached to chunks.
3. Text is divided into 80-word chunks with 16-word overlap. Chunking is page-local.
4. `sentence-transformers/all-MiniLM-L6-v2` creates one 384-dimensional vector per chunk. It is a pretrained model; the PDF does not train or fine-tune it.
5. ChromaDB persists each chunk's vector, text and source/page metadata in the dedicated `financial_assistant_pdf_rag_v4` collection.
6. A successful indexing operation replaces the previous indexed PDF rather than appending another source. Extraction and embedding happen before the existing collection is cleared.

The UI displays chunk text and boundaries, model name, vector dimensions, and the first few vector values. Those values are a sample of a dense semantic representation; individual dimensions are not labels humans can interpret independently.

## Question answering

1. The question is embedded with the same model used for document chunks.
2. ChromaDB returns up to five nearest passages and cosine distances.
3. The generator labels those passages `S1`, `S2`, etc. and sends the question and evidence to the configured Groq-hosted LLM.
4. The LLM is instructed to use only the evidence, return JSON, answer concisely and provide source IDs. The application discards invalid citation IDs and does not mark an answer supported unless the model claims support and cites at least one valid ID.
5. The UI displays the answer, cited/uncited retrieved passages, source pages, distances, prompt/context, and provider token usage when available.

This is a practical guardrail, not a mathematical guarantee against hallucinations. The model can misinterpret evidence or cite a passage that does not fully justify its answer; the displayed evidence is available for review.

## Token and cost observability

The app reports actual input/output token counts when the provider returns usage data. It also shows word-based rough estimates for document and retrieved-context size; the estimate uses a simple multiplier and is not a tokenizer. These estimates illustrate the idea of sending selected passages rather than a whole document, but they are not a measured percentage saving. Embedding/indexing work, retrieval misses and LLM prompt overhead are trade-offs.

## Data boundaries

- PyMuPDF, chunking, Sentence Transformers and ChromaDB run locally.
- The question and retrieved text passages are sent to Groq for answer generation.
- The PDF is not used to train either model.
- No bundled sample text is automatically indexed.
- This version deliberately removes receipt-image Q&A and OCR to keep the learning path focused. Scanned/image-only PDFs with no selectable text are rejected with an actionable message.

## Persistence

- ChromaDB stores vectors, chunk text and source/page metadata.
- SQLite stores interaction metadata, answer support status and latency.
- The user-provided Groq key is read from a local `.env` file, which must not be committed.

## Main trade-offs and limitations

- Word-based chunks are easy to understand but do not respect sentence/section boundaries as well as a structure-aware chunker.
- The embedding model is English-oriented and can miss exact identifiers, numbers or unusual terminology; cosine distance alone is not a proof of support.
- Top-k retrieval can omit relevant passages when a document is long or repetitive.
- The LLM may still produce unsupported claims despite prompt constraints and citation validation.
- Scanned PDFs require OCR outside this simplified version.
- Local model downloads can take time; Groq answer generation requires network access and a valid API key.
