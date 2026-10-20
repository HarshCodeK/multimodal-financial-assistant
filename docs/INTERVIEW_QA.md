# Interview Q&A — PDF RAG Explorer

## 1. What does the project do?

It lets a user upload one PDF, indexes its text as overlapping chunks and semantic vectors, retrieves relevant passages for a question, and asks an LLM to answer using those passages with source references.

## 2. What is the full pipeline?

PDF → PyMuPDF extraction → overlapping chunks → Sentence Transformer embeddings → ChromaDB → question embedding → top-k retrieval → Groq LLM generation → concise answer + cited evidence.

## 3. What is RAG?

Retrieval-Augmented Generation combines retrieval from an external knowledge source with LLM generation. The LLM receives selected evidence at answer time instead of relying only on its pretrained knowledge or receiving the entire document every time.

## 4. Does the PDF train the LLM or embedding model?

No. The embedding model and LLM are already pretrained. This app extracts and indexes the uploaded PDF. It does not fine-tune either model on the document.

## 5. What does PyMuPDF do?

It extracts selectable text from each PDF page and lets the app preserve page numbers. It does not summarize or interpret the document. This simplified project does not contain OCR, so image-only/scanned PDFs need OCR preprocessing before upload.

## 6. Why do we chunk documents?

An entire document can be too large or wasteful to include in every LLM request. Chunks let retrieval select a smaller relevant subset. This project uses 80-word chunks with 16-word overlap. Overlap helps preserve local context across boundaries; page-local chunking keeps page references clear.

## 7. What exactly is an embedding?

An embedding is a fixed-length list of numbers produced by a pretrained neural network. The numbers are a learned representation of a text's semantic features, useful for comparing texts. The chosen `all-MiniLM-L6-v2` model outputs 384-dimensional vectors. A vector is not a summary and individual coordinates usually do not map to human-readable concepts.

## 8. How is the embedding created?

The Sentence Transformers library tokenizes the text and passes it through the pretrained transformer network. The model's contextual token representations are pooled into a single sentence/passage vector. This app normalizes the resulting vector before storing it. We reuse the same embedding model for chunks and user questions so their vectors are comparable.

## 9. What is ChromaDB responsible for?

ChromaDB stores vectors, original chunk text and metadata, and retrieves nearest vectors for the question vector. It is the vector store—not the embedding model and not the LLM. The app displays cosine distances; lower distance means closer in vector space, but it does not prove the passage answers the question.

## 10. What does the retriever do?

It embeds the question and asks ChromaDB for the closest five chunks. Those passages are the context supplied to the generator. Retrieval can miss relevant passages, so top-k is a trade-off between context size and recall.

## 11. What does the LLM do?

It receives the question and retrieved evidence, then generates a concise answer and cites source IDs. The prompt instructs it not to use outside knowledge and to abstain if the evidence is insufficient. The app validates citation IDs, but this cannot guarantee that the answer is factually correct; reviewers can inspect the exact passages and prompt.

## 12. Why did the old version show paragraphs instead of direct answers?

The old version performed retrieval and displayed retrieved passages; it had no LLM answer-generation step. The new version adds a separate generation stage after retrieval.

## 13. Why does this reduce tokens?

Without RAG, an implementation may send the entire document for each question. This app sends the question and a few retrieved passages instead. The document is embedded/indexed once and the vector index is reused. This can reduce repeated input context on larger documents, but indexing has a cost, and a short PDF may show little or no benefit. The app reports provider token usage when available and labels rough word-based estimates as estimates.

## 14. Is it always cheaper?

No. Savings depend on document size, number of questions, retrieval size, model pricing, and indexing overhead. Retrieval can also miss evidence. This project does not claim a fixed cost reduction.

## 15. Why keep page numbers and citations?

They let a reviewer trace an answer back to source evidence instead of trusting an unverified generated paragraph. The UI also shows retrieved but uncited passages and cosine distances.

## 16. What happens if a question is not in the PDF?

The LLM is instructed to say the document does not specify the answer. Citation validation requires at least one valid source ID for an answer marked supported. This reduces, but does not eliminate, hallucinations.

## 17. Why one PDF at a time?

It makes evidence boundaries explicit for a learning demo. Each successful indexing operation replaces the previous index so the user can tell which document is being questioned. No bundled sample text is silently included.

## 18. Why Groq? Is data sent outside the machine?

Groq is used for hosted LLM answer generation, configured with `GROQ_API_KEY` and `GROQ_MODEL`. Extraction, chunking, embeddings and ChromaDB retrieval run locally, but the question and retrieved PDF excerpts are sent to Groq. Do not use confidential documents unless that data transfer is acceptable.

## 19. What are the limitations?

Word-based chunks may split sentences; embeddings can miss exact numbers or identifiers; top-k retrieval can miss evidence; cosine distance is not proof of relevance; and the LLM can still misinterpret evidence. The simple version does not support scanned PDFs without preprocessing. All generated answers should be checked against their displayed evidence.
