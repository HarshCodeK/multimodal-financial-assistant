# Interview Q&A — Multimodal Financial Assistant

## The 30-second pitch

"It reads a receipt or invoice, extracts structured fields with a vision
model, retrieves the governing policy from a ChromaDB vector store, and
answers only from what it retrieved — citing the source document, and
refusing to answer when the corpus doesn't cover the question."

## Q: Why embeddings instead of keyword search?

A: The user asks "why was this charge deducted" while the policy says "fees
assessed after the due date". Those sentences share almost no words, so keyword
matching fails unless the query uses the document's own vocabulary. Embeddings
place both in the same neighbourhood. That is the whole reason a vector store
belongs here instead of a TF-IDF index.

## Q: Why ChromaDB over FAISS?

A: Two concrete reasons. First, ChromaDB persists to disk, so the index
survives a restart; FAISS would require building the persistence layer
myself. Second, ChromaDB's query API returns source metadata with each hit —
which is what lets the answer cite where it came from. FAISS is faster and
would be the right call at a scale this corpus never reaches.

## Q: Why a local embedding model?

A: `all-MiniLM-L6-v2` is 22M parameters and runs on CPU, so retrieval is
free, offline, and deterministic. An embedding API would be marginally better
in quality but needs a network round-trip and an API key for something a local
model does adequately. For three policy documents, the trade clearly favours
local.

## Q: Why is the refusal to answer a feature?

A: Asked why a compute charge was high with no pricing policy in the corpus,
the system says the documents don't explain it, and quotes what they *do* say.
An LLM handed three irrelevant chunks will happily invent a plausible-sounding
reason if allowed to, and an invented reason about money is worse than no
answer. Grounded refusal is what makes this safe to demo and honest to use.

## Q: How does extraction stay structured?

A: `response_format={"type": "json_object"}` makes the provider guarantee the
JSON shape — asking in prose is a coin flip. One retry remains for a model that
still wraps output in a markdown fence; on retry the instruction is restated
more forcefully. An unparseable response raises `ExtractionError` with the
real cause, never a silent half-answer.

## Q: Images vs PDFs — how does that work?

A: An image is base64-encoded and sent to the vision model as an image_url. A
PDF goes through PyMuPDF which extracts the text layer; the model then reads
text. A model without the vision flag is rejected up front with a list of
models that can read images, rather than failing mid-request.

## Q: What about the 100-word chunks?

A: Word-count chunking can split a sentence across two chunks. Sentence-aware
splitting would be better in general; at six chunks over three documents the
measured retrieval is identical, so the simple version stays. An interviewer
asking this gets: I measured, the fancier option did not change the result,
so I kept the simple one.

## Q: What did you learn from the 404 bug?

A: The original project pinned a model id the provider had retired; every call
returned 404. In megaproject the circuit breaker masked it. Here, extraction
maps a 404 to `ExtractionError` naming the model and pointing at
`src/models.py`, so the failure is distinguishable from an offline one. The
general lesson: a generic "something failed" is a bug farm; classify failures.

## Q: What does the SQLite log store record?

A: Question, answer, extracted fields, sources cited, model id, and latency —
one row per interaction. It makes runs auditable and diffable offline, without
sending anything anywhere.

## Q: Where is this unsafe to run in production?

A: There is no confidence threshold on extraction — fields are taken as the
model returned them, so a misread number is trusted. A production version
flags low-confidence fields for human review. Document text goes into the
prompt unescaped, so a malicious document is a prompt-injection vector. And
retrieval quality over a real policy corpus is unmeasured; three short
documents do not prove anything about recall at scale.

## Q: How is this tested?

A: 14 offline tests: chunking, deterministic ids mean re-ingestion replaces
rather than duplicates, metadata carries the source filename, grounded-answer
citations, extraction error classification, PDF parsing, log-store round trip.
No test touches the network or a real model.
