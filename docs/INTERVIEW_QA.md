# Interview Q&A

1. **What problem does the project solve?** It lets a user ask questions about a financial policy and see the source text used to answer, instead of relying only on the model's parametric knowledge.

2. **Why RAG?** Relevant passages are retrieved from the supplied policy and sent with the question, which limits context and makes evidence inspection possible.

3. **What is the pipeline?** PyMuPDF extracts text; Sentence Transformers creates embeddings; ChromaDB stores and retrieves chunks; Groq generates an answer constrained by retrieved evidence.

4. **Why ChromaDB and SQLite?** ChromaDB is the vector store for semantic search. SQLite is a relational log for questions, answers, latency, and extraction metadata.

5. **Why local embeddings?** The embedding model runs locally, so the project does not need a separate embedding API.

6. **How does multimodality work?** PDF text is indexed for RAG. Receipt/statement images use a separate Groq vision call. Image outputs are not automatically treated as indexed document evidence.

7. **Is the image feature OCR?** It is vision-model-based reading, not a dedicated OCR engine. The model may misread numbers, so important figures must be verified.

8. **Does it support scanned PDFs?** No. PyMuPDF extracts selectable text; scanned PDFs need OCR, which is not included in this version.

9. **Why run Streamlit and services in one process?** This is a local portfolio demo, so a separate HTTP API and second terminal added setup cost without helping the core demonstration. The UI calls the service layer directly.

10. **How are answers grounded?** The model receives retrieved passages and is instructed to refuse when evidence is insufficient. JSON is validated and common formatting issues are repaired conservatively. This reduces risk but does not guarantee correctness.

11. **What do tests prove?** Unit tests exercise extraction, retrieval, JSON repair, logging, and service behavior with fakes. They do not prove live Groq accuracy, provider availability, or OCR performance.

12. **What are the main limitations?** Retrieval depends on the source and embedding model; image reading can be wrong; scanned PDFs are unsupported; the demo has no multi-user isolation or production governance.
