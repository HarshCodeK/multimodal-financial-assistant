# Interview Q&A

1. **Why RAG?** Policy answers should be grounded in the organization's supplied documents rather than relying on model memory.
2. **Why ChromaDB instead of SQLite?** Retrieval is vector similarity; SQLite is used for structured logs.
3. **Why not send the whole document to the LLM?** Retrieval reduces context size and makes source attribution practical.
4. **Why local embeddings?** They reduce external dependencies and keep the embedding layer replaceable.
5. **How do you avoid hallucination?** The answerer receives retrieved evidence and is instructed to refuse when evidence is absent; this is not a formal guarantee.
6. **How do you test RAG without a live LLM?** Fake embeddings, fake answerers, and a real Chroma round trip isolate retrieval from provider behavior.
7. **Why dependency injection?** It lets API tests run without downloading a model or requiring credentials.
8. **Why SQLite?** Interaction logs are relational records; they need simple durable writes, not similarity search.
9. **How is multimodality handled?** PDF text is indexed, while image input is validated and encoded for a configurable vision model. The two paths stay explicit because image-derived facts do not currently have the same retrieval provenance as PDF chunks.
10. **Why structured LLM output and JSON repair?** The provider is asked for a JSON object with `supported` and `answer`; Pydantic validates the types and required fields. A conservative repair pass handles common formatting mistakes such as Markdown fences and trailing commas. It does not invent missing fields, and malformed responses still fail closed.
11. **Main limitation?** Retrieval quality depends on the embedding model and corpus; the project is a demonstrator, not a regulated financial system.

12. **What is logged in SQLite?** Successful question/answer and image interactions include elapsed request time; successful PDF ingestion records the source, indexed chunk count, page count, and elapsed time as structured extraction metadata. The store retains a bounded number of rows and migrates older databases by adding nullable columns without dropping existing history.
