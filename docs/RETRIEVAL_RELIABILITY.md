# Retrieval reliability notes

## Ingestion semantics

Re-ingesting a source replaces its indexed records rather than blindly appending a second version. Embeddings are computed before the old source records are deleted, and collection writes are serialized within this process. This prevents ordinary repeat-ingestion from leaving stale pages, but it is not a cross-process transaction: concurrent writers in multiple worker processes need an external ingestion lock or a job queue.

Chunk IDs are deterministic for source, type, page and ordinal. Source identifiers must be canonicalized by the caller; moving a file to a new path is treated as a different source.

## Query bounds

The retriever rejects blank queries and caps the requested result count at 10. This is an application-level guard, not a guarantee of retrieval quality. Evaluate retrieval precision and recall against a labeled query set before making accuracy claims.

## Failure cases to test

- Re-ingesting a source with fewer pages removes stale pages from its previous version.
- Embedding failures happen before source deletion.
- Empty queries and excessive result counts are rejected.
- Multi-process ingestion is not atomic; do not claim it is.
- Scanned PDFs still need OCR; text extraction alone does not make them searchable.
