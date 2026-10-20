# Security and Failure Considerations

- **Sensitive files:** use synthetic policy documents and receipts for demos. Files are processed locally and are not uploaded to a model provider.
- **Oversized or malformed uploads:** PDFs are limited by file size, page count, and extracted character count. Images are limited by file size and pixel count and validated before OCR.
- **OCR errors:** OCR may confuse digits, dates, decimal separators, or currency symbols. Verify financial figures against the original image.
- **Prompt injection:** there is no LLM in this simplified version, so document text is not passed to a generative model as instructions.
- **Unsupported questions:** the retriever uses a semantic-distance threshold. If no sufficiently close passage is found, the app reports that instead of inventing an answer.
- **Local persistence:** ChromaDB stores document chunks and embeddings; SQLite stores questions, results, latency, and extraction metadata. Treat these local files as sensitive if you use private documents.
- **Model downloads:** Sentence Transformers and OCR runtime may download model assets on first use. If a download fails, the relevant action will show an error.
- **Single-user scope:** this is a local portfolio demo, not a multi-user service or production financial system.
