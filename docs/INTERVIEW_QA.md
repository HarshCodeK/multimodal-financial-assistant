# Interview Q&A

1. **What does the project do?** It extracts text from financial policies and receipt images, indexes policy text, and retrieves relevant passages for a question.
2. **What is the pipeline?** PyMuPDF extracts PDF text; RapidOCR handles scanned pages and images; chunking splits text into overlapping windows; Sentence Transformers embeds chunks; ChromaDB retrieves semantic matches.
3. **Why overlap chunks?** Overlap reduces the chance that a sentence or rule split at a boundary loses important context.
4. **Why use embeddings?** They retrieve semantically related passages even when the query does not use exactly the same words as the document.
5. **Why ChromaDB?** It stores vectors and metadata and supports nearest-neighbor retrieval without operating a separate database server.
6. **What is OCR here?** RapidOCR is a local OCR engine. It recognizes visible text and preserves line boundaries; it is not a general vision-language model and may misread numbers.
7. **How does receipt Q&A work?** It tokenizes the question and OCR lines, ranks lines by term overlap, and returns matching text as evidence. This transparent baseline is not semantic search or LLM reasoning, and can miss paraphrases.
8. **Does it use an LLM?** No. To remove API keys, authentication failures, and external dependencies, this simplified version returns retrieved passages instead of generating free-form answers.
9. **Why lazy-load the embedding model?** The UI renders immediately; model initialization happens when indexing or searching is first requested.
10. **What are the limitations?** OCR errors, embedding quality, and irrelevant nearest-neighbor matches. The app uses a similarity-distance threshold and displays source passages for verification.
11. **Where is data stored?** ChromaDB stores the vector index locally and SQLite stores interaction logs. No data is sent to a model provider.
