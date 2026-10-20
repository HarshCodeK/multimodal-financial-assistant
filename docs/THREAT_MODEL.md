# Security and Failure Considerations

- **Provider data transfer:** PDF extraction, chunking, embeddings and vector retrieval run locally. PDF questions plus retrieved excerpts are sent to Groq for text generation. Image bytes plus the image question are sent to Groq for vision inference. Do not use confidential files unless this transfer is acceptable.
- **Upload validation:** PDF extraction enforces a 10 MiB limit, page-count limit and extracted-character limit. Image Q&A accepts PNG/JPEG/WebP with recognized byte signatures and a 5 MiB limit. These checks reduce accidental misuse but are not a complete malware scanner.
- **Image interpretation:** the vision model may misread amounts, dates, small print, or low-quality images. It is instructed to identify uncertainty and not guess, but this is not guaranteed. Verify financial figures against the original.
- **Prompt injection:** visible text in an image or PDF can contain instructions. The vision prompt explicitly says not to follow instructions in the image, but prompt wording is not a security boundary. The PDF text LLM is instructed to use retrieved passages as evidence, but prompt injection and hallucination remain possible.
- **Unsupported PDF questions:** the text model is instructed to abstain when retrieved evidence is insufficient. Valid citation IDs are checked, but the app does not prove semantic entailment; review cited passages.
- **Persistence:** ChromaDB stores PDF chunks, embeddings and source metadata; SQLite stores interaction/index metadata. Image bytes are sent to Groq for the request and are not added to ChromaDB by this app. Treat local data and provider submissions as sensitive.
- **API key:** store `GROQ_API_KEY` in the ignored local `.env` file. Never commit or share it.
- **Model/API availability:** text and image answering need network access, a valid Groq key and supported model access. A model/provider outage should produce an error rather than a fabricated result.
- **Single-user scope:** this is a local portfolio/learning app, not a multi-user service or production financial system.
