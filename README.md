# Multimodal Financial Assistant

Reads a receipt or invoice, extracts structured fields, retrieves relevant policy context from a ChromaDB vector store using the user question plus document fields, and generates a policy-grounded answer with source citations. Images use the vision model; text-based PDFs are parsed with PyMuPDF.

```
PDF or image
   |
   v
parser.py / extract.py
   | image -> vision model
   | PDF   -> PyMuPDF text layer -> model
   v
structured fields -> {vendor, line_items, subtotal, tax, total, date}
   |
   v
knowledge_base.py   ChromaDB top-3 policy chunks, with source filenames
   |
   v
answer.py     grounded answer + the documents it cited
   |
   v
log_store.py  SQLite: question, answer, fields, sources, model, latency
```

**Python · RAG · LLMs · ChromaDB · sentence-transformers · Groq · Streamlit**

---

## The idea worth explaining

**The answer refuses to guess.**

Asked why a compute charge was high, with no pricing policy in the corpus, the
system answered:

> "The provided policy documents do not explain why the compute charge was
> high. [subscription_billing.txt] states that overage charges are assessed
> after the included allowance is exhausted, but it does not provide the
> specific allowance limits or unit rates..."

The refusal is a prompt-level grounding rule: the model is instructed to answer only from retrieved policy context. That is useful behavior, not a hard security boundary; a malicious document can still create a prompt-injection risk, which is listed under the limitations.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY
streamlit run app.py
```

On the included sample corpus, the first question builds the persistent vector index; later questions reuse it.

---

## Why embeddings, not keywords

The user asks *"why was this charge deducted"*. The policy says *"fees assessed
after the due date"*. Those two sentences share almost no words, so keyword
search fails unless the query happens to use the document's own vocabulary.

Embeddings put both in the same neighbourhood. That is the whole reason a vector
store is here rather than a TF-IDF index — and it is the honest answer if asked
why ChromaDB is worth the dependency.

**Why a local embedding model:** `all-MiniLM-L6-v2` is 22M parameters, runs on
CPU, and costs nothing. An embedding API would be marginally better and would
require a network round-trip for something deterministic.

---

## Measured

| Step | Result |
|---|---|
| Included image extraction sample | 1355 ms; sample fields matched the expected values |
| ChromaDB ingest | 6 chunks from 3 policy documents |
| Retrieval | top-3 with filenames attached and the user question included in the retrieval query |
| Grounded answer | cites sources, admits gaps |

---

## Layout

| File | What it does |
|---|---|
| `src/models.py` | Model ids, prices, which can read images |
| `src/parser.py` | PDF/image -> the dict the extractor expects |
| `src/extract.py` | Vision model -> structured fields |
| `src/knowledge_base.py` | ChromaDB + embeddings |
| `src/answer.py` | Grounded answer with citations |
| `src/log_store.py` | SQLite interaction log |
| `app.py` | Streamlit UI |

---

## Why these choices

**JSON mode over "please return JSON".** Asking for JSON and then parsing it is
a coin flip. `response_format={"type": "json_object"}` makes the provider
guarantee the shape. One retry remains as a backstop.

**A local embedding model over an API.** Offline, free, deterministic.

**ChromaDB over FAISS.** It persists to disk, so the index survives a restart,
and returns source metadata with each hit — which is what lets an answer cite
where it came from. FAISS is faster and would leave the persistence layer to me.

**Vision-capable model required for images.** `src/models.py` marks which configured model entries can accept images; text-based PDFs use the same configured model after PyMuPDF extraction.

---

## Interview Q&A

`docs/INTERVIEW_QA.md` — the pitch, the trust decisions, and the questions an
interviewer will actually ask, with answers grounded in this code.

## Known limits

- **No confidence threshold on extraction.** Fields are taken as returned. A
  production version would flag low-confidence fields for human review — this is
  the main gap.
- **The corpus is three short documents.** Retrieval quality on a real policy set
  would need measuring.
- **Document text is untrusted prompt data.** A malicious document can attempt prompt injection; a production version should isolate document content from system instructions.
- **100-word chunks can split a sentence.** Sentence-aware splitting would be better; the current sample corpus is too small to establish retrieval quality at scale.


## License

MIT.
