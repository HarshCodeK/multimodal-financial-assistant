# Security and Failure Considerations

- **Provider data transfer:** PDF extraction, OCR, chunking, embeddings and retrieval run locally. PDF questions plus retrieved excerpts are sent to Groq for text generation. Image bytes are processed locally; only the image question and OCR-extracted text are sent to Groq. Do not use confidential files unless this transfer is acceptable.
- **Upload validation:** PDF extraction enforces a 10 MiB limit, page-count limit and extracted-character limit. Image OCR accepts PNG/JPEG/WebP with recognized byte signatures and a 5 MiB limit. These checks are not a complete malware scanner.
- **OCR accuracy:** local OCR may misread amounts, dates, small print, handwriting, or low-quality images. The UI shows extracted text so users can inspect it. Verify financial figures against the original.
- **Text-only limitation:** the app does not use a vision model. It cannot reliably interpret charts, spatial relationships, or visual details that OCR does not capture.
- **Prompt injection:** visible text in a PDF/image can contain instructions. The text LLM is instructed to treat source content as untrusted evidence, but prompt wording is not a complete security boundary.
- **Unsupported PDF questions:** the text model is instructed to abstain when retrieved evidence is insufficient. Valid citation IDs are checked, but the app does not prove semantic entailment; review cited passages.
- **Persistence:** ChromaDB stores PDF chunks, embeddings and source metadata; SQLite stores interaction/index metadata and question/answer text. Image bytes are not stored in ChromaDB or sent to Groq by this workflow.
- **API key:** a usable `GROQ_API_KEY` in the project-root `.env` takes precedence over inherited environment variables. Never commit or share the key.
- **Model/API availability:** text answering needs network access, a valid Groq key and access to `openai/gpt-oss-20b`. Failures should be shown rather than replaced with fabricated results.
- **Single-user scope:** this is a local learning app, not a multi-user service or production financial system.
