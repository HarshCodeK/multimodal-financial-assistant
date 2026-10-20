# Security and Failure Considerations

- **Prompt injection in documents:** retrieved text is supplied as evidence, not as instructions. The model is told to answer only from retrieved evidence, but this is not a complete prompt-injection defense.
- **Oversized or malformed uploads:** PDFs are limited by file size, page count, and extracted text length. Images are limited by file size and pixel count and validated by Pillow before conversion.
- **Secret leakage:** the Groq API key is read from the `GROQ_API_KEY` environment variable. Never commit API keys or real financial documents.
- **Unsupported questions:** if retrieval returns no evidence, the text model returns an explicit insufficient-evidence answer without calling Groq.
- **Provider failure:** Groq HTTP errors surface in the Streamlit UI; the app does not silently replace a failed model response with a fabricated answer.
- **Image-reading errors:** vision models can misread digits, dates, tax, and totals. Verify extracted figures against the original image.
- **Sensitive logs:** SQLite stores submitted questions and generated answers. Use synthetic documents for demos and delete local `data/interactions.db` if it contains data you should not retain.
- **Local-only scope:** this version is designed for a single-user local demo. It has no authentication, tenant isolation, or production data governance and should not be exposed as a shared public service.
- **Scanned PDFs:** no OCR engine is included; a PDF without selectable text is rejected with an explanatory message.
