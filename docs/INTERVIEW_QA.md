# Interview Q&A — Financial Assistant

## 1. What does the project do?

It offers two text-first workflows. PDF RAG indexes a PDF into overlapping chunks and vectors, retrieves passages for a question, and asks GPT-OSS-20B to answer from that evidence with source IDs. Image Q&A runs local OCR, shows the extracted text, then asks the same text model to answer from that text. No vision model is used.

## 2. What is the PDF RAG pipeline?

PDF → PyMuPDF selectable-text extraction (local OCR fallback for image-only pages) → overlapping chunks → Sentence Transformer embeddings → ChromaDB → question embedding → top-k retrieval → GPT-OSS-20B → concise answer + cited evidence.

## 3. What is the image Q&A pipeline?

PNG/JPEG/WebP upload → validate signature/size → local RapidOCR → inspect extracted text → send question + extracted text to GPT-OSS-20B → answer and token usage. Raw image bytes are not sent to Groq.

## 4. Why keep PDF and image workflows separate?

The PDF path builds a persistent searchable index and exposes chunk boundaries, vectors, retrieval results and citations. The image path performs local OCR for one uploaded image and sends extracted text to the text LLM; it does not alter the PDF index.

## 5. Does the uploaded PDF or image train either model?

No. The embedding model, OCR model and text LLM are pretrained. The app performs inference and does not fine-tune models on uploads.

## 6. What does PyMuPDF do?

It extracts selectable PDF text and page numbers. For pages containing images but no selectable text, the app renders the page locally and uses OCR as a fallback.

## 7. Why use OCR instead of a vision model?

The design deliberately avoids vision-model configuration and image-input API calls. OCR plus a text LLM is suitable for receipts, forms, statements and screenshots with readable text, and keeps image bytes local. The trade-off is that OCR can lose layout, handwriting detail, charts and spatial relationships; those tasks would require a different architecture.

## 8. What is the embedding model doing?

`all-MiniLM-L6-v2` maps chunks and questions into 384-dimensional vectors so ChromaDB can retrieve semantically similar passages. It is not the answer-generating LLM.

## 9. Does a citation guarantee correctness?

No. The app checks that citation IDs refer to retrieved chunks and displays the evidence. It does not mathematically prove the answer is entailed by the cited text.

## 10. What data leaves the machine?

PDF questions and retrieved PDF excerpts go to Groq. For images, OCR runs locally and only the question plus extracted text go to Groq. Local SQLite records question/answer text, model usage and latency; it does not store API keys or image bytes.

## 11. What are the main limitations?

OCR may misread blurry text, handwriting and dense layouts. Text-only models cannot understand visual details absent from OCR. Retrieval can miss passages, and LLM answers can still be wrong. This is an educational single-user demo, not a production financial system.
