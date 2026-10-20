# Interview Q&A — Multimodal Financial Assistant

## 1. What does the project do?

It offers two distinct workflows. PDF RAG indexes one text-based PDF into overlapping text chunks and vectors, retrieves passages for a question, and asks an LLM to answer from that evidence with source IDs. Image Q&A sends an uploaded image and a question directly to a vision-capable LLM.

## 2. What is the PDF RAG pipeline?

PDF → PyMuPDF extraction → overlapping chunks → Sentence Transformer embeddings → ChromaDB → question embedding → top-k retrieval → Groq text LLM → concise answer + cited evidence.

## 3. What is the image Q&A pipeline?

PNG/JPEG/WebP upload → signature and size validation → base64 data URL → image + user question sent to Groq vision model → answer and provider usage displayed. This is direct visual question answering, not OCR-indexed image RAG.

## 4. Why keep PDF and image workflows separate?

They solve different tasks. The PDF path builds a persistent searchable index and can show chunk boundaries, vectors, retrieval results and citations. The image path sends one image directly to a vision model per request. Images do not enter or replace the PDF index.

## 5. Does the uploaded PDF or image train either model?

No. The embedding model, text LLM and vision model are pretrained. This app extracts and indexes PDF text or sends an image as inference input. It does not fine-tune models on these uploads.

## 6. What does PyMuPDF do?

It extracts selectable text from each PDF page and preserves page numbers. It does not summarize or interpret the document. This PDF workflow does not run OCR; image-only/scanned PDFs need OCR preprocessing before upload.

## 7. Why do we chunk documents?

Chunks let retrieval select a smaller subset instead of sending the whole PDF for every question. This project uses 80-word chunks with 16-word overlap. Overlap preserves some local context across chunk boundaries; page-local chunking helps keep page references clear.

## 8. What exactly is an embedding?

An embedding is a fixed-length numeric vector produced by a pretrained neural network. Its learned representation is useful for comparing text semantically; it is not a human-readable summary. `all-MiniLM-L6-v2` produces 384-dimensional vectors in this app.

## 9. What does ChromaDB do?

It stores PDF chunk vectors, original chunk text and metadata, and retrieves nearby vectors for the question vector. It is the vector store—not the embedding model or the LLM. Cosine distance indicates vector proximity, not proof that a passage answers the question.

## 10. What does the PDF LLM do?

It receives the question and retrieved evidence, then generates a concise answer and source IDs. The prompt instructs it not to use outside knowledge and to abstain if the evidence is insufficient. Citation validation rejects invalid source IDs, but neither prompting nor citation validation guarantees correctness.

## 11. Why does RAG potentially reduce token usage?

Without retrieval, an implementation may send the whole PDF with every question. This app indexes the document once and sends a small number of retrieved passages for each question. This can reduce repeated input context for large documents, but indexing has overhead and short PDFs may not benefit. The project does not claim a fixed cost saving.

## 12. Is the image workflow RAG?

No. The image is passed directly to a vision-capable model along with the user's question. The image is not chunked, embedded, stored in ChromaDB, or retrieved later. A future image-RAG implementation would require a separate explicit design.

## 13. What is sent outside the machine?

PDF extraction, chunking, embeddings and ChromaDB retrieval run locally. For PDF answers, the question and retrieved excerpts are sent to Groq. For image Q&A, the image and question are sent to Groq. Do not upload confidential files unless this provider transfer is acceptable.

## 14. What happens when the model cannot read a value?

The vision model is instructed to flag unclear details rather than guess. This is not guaranteed behavior. Users must verify financial amounts, dates and identifiers against the original image.

## 15. What are the limitations?

PDF chunks can split sentences; embeddings can miss exact identifiers; top-k retrieval can miss evidence; cosine distance does not prove support; text and vision models can make mistakes; scanned PDFs need external OCR; and image analysis requires network access and a valid Groq API key. This is an educational single-user app, not a production financial system.
