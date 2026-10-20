"""Streamlit UI for a transparent, single-PDF RAG demo."""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from financial_assistant.services import create_default_assistant


st.set_page_config(page_title="PDF RAG Explorer", page_icon="📄", layout="wide")
st.title("PDF RAG Explorer")
st.caption(
    "Upload one PDF, inspect extraction/chunking/embeddings/retrieval, then ask an LLM for a concise answer "
    "grounded in the retrieved passages."
)


@st.cache_resource(show_spinner=False)
def get_assistant():
    return create_default_assistant(PROJECT_ROOT)


def submit_question():
    question = st.session_state.get("question_input", "").strip()
    if not question:
        return
    try:
        with st.spinner("Embedding your question, retrieving evidence, and generating a grounded answer..."):
            result = get_assistant().ask(question)
        st.session_state.setdefault("chat_history", []).append({"question": question, "result": result})
        st.session_state["question_input"] = ""
        st.session_state.pop("question_error", None)
    except Exception as exc:
        st.session_state["question_error"] = str(exc)


with st.sidebar:
    st.header("1. Choose the source")
    sample_pdf_path = PROJECT_ROOT / "data" / "sample_financial_policy.pdf"
    if sample_pdf_path.exists():
        st.download_button(
            "Download fictional sample PDF",
            data=sample_pdf_path.read_bytes(),
            file_name="sample_financial_policy.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.caption("Downloading the sample does not index it. Upload the PDF below and click the indexing button.")
    uploaded_pdf = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        key="pdf_upload",
        help="This simple version extracts selectable PDF text. It does not include receipt OCR or scanned-page OCR.",
    )
    if st.button("Extract, chunk, embed & index PDF", disabled=uploaded_pdf is None, use_container_width=True):
        try:
            with st.spinner("Extracting text, creating chunks and embeddings, then replacing the current index..."):
                summary = get_assistant().index_pdf(uploaded_pdf.name, uploaded_pdf.getvalue())
            st.session_state["last_index_result"] = summary
            st.session_state["chat_history"] = []
            st.success(
                f"Indexed {summary['chunks_indexed']} chunks from {summary['pages_extracted']} pages. "
                "The previous PDF index was replaced."
            )
        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")

    st.divider()
    st.header("2. What this demo does")
    st.markdown(
        "- **PyMuPDF:** extracts selectable text and page numbers.\n"
        "- **Chunking:** 80-word chunks with 16-word overlap.\n"
        "- **Sentence Transformers:** maps each chunk to a fixed-length vector (384 dimensions with the default model).\n"
        "- **ChromaDB:** stores vectors, text and source metadata.\n"
        "- **Retriever:** selects the closest passages for each question.\n"
        "- **LLM:** answers from those passages and cites source IDs."
    )
    st.warning(
        "LLM privacy: the question and retrieved PDF excerpts are sent to Groq to generate the answer. "
        "PDF extraction, chunking, embeddings and ChromaDB retrieval run locally."
    )

assistant = get_assistant()
indexed_sources = assistant.indexed_sources()
if indexed_sources:
    st.info("Currently indexed PDF: " + ", ".join(indexed_sources))
else:
    st.info("No PDF is indexed yet. Upload a PDF in the sidebar and click the indexing button.")

summary = st.session_state.get("last_index_result")
if summary:
    st.subheader("Indexing report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages with text", summary["pages_extracted"])
    c2.metric("Chunks indexed", summary["chunks_indexed"])
    c3.metric("Approx. document tokens", f"~{summary['estimated_document_tokens']:,}")
    c4.metric("Vector dimensions", summary["embedding_dimensions"] or "unknown")
    st.caption(
        f"Embedding model: {summary['embedding_model']} · "
        f"Chunk size: {summary['chunk_words']} words · "
        f"Overlap: {summary['chunk_overlap_words']} words. "
        "Token counts are rough word-based estimates, not tokenizer measurements."
    )
    with st.expander("Inspect extracted chunks and chunk boundaries"):
        st.write(
            "Each chunk is embedded independently. Adjacent chunks share 16 words to reduce context loss at boundaries. "
            "Short PDFs may correctly produce only one chunk."
        )
        for chunk in summary["chunks"]:
            page = f" · page {chunk['page']}" if chunk["page"] else ""
            st.markdown(f"**Chunk {chunk['number']}{page} — {chunk['words']} words**")
            st.code(chunk["text"], language="text")
    with st.expander("Inspect the embedding vector"):
        st.write(
            "The pretrained embedding model turns a chunk into a fixed-length numeric vector. "
            "The vector is a learned semantic representation, not a human-readable summary and not training on this PDF."
        )
        st.write(f"Dimensions: {summary['embedding_dimensions']}")
        st.code(str(summary["embedding_preview"]) + (" ..." if summary["embedding_dimensions"] and summary["embedding_dimensions"] > 8 else ""))

st.subheader("Ask the indexed PDF")
for entry in st.session_state.get("chat_history", []):
    result = entry["result"]
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        if result.supported:
            st.markdown(result.answer)
            if result.citations:
                st.caption("Cited evidence: " + ", ".join(f"[{item}]" for item in result.citations))
        else:
            st.warning(result.answer)

        stats = []
        if result.model:
            stats.append(f"LLM: {result.model}")
        if result.prompt_tokens is not None:
            stats.append(f"actual input tokens: {result.prompt_tokens}")
        if result.completion_tokens is not None:
            stats.append(f"actual output tokens: {result.completion_tokens}")
        if result.context_token_estimate:
            stats.append(f"retrieved-context estimate: ~{result.context_token_estimate} tokens")
        if stats:
            st.caption(" · ".join(stats))

        with st.expander("Inspect retrieved passages and similarity distances"):
            if not result.sources:
                st.write("No passages were retrieved.")
            for index, source in enumerate(result.sources, start=1):
                source_id = f"S{index}"
                cited = source_id in result.citations
                label = f"[{source_id}] {Path(source.source).name}"
                if source.page:
                    label += f" · page {source.page}"
                label += " · CITED" if cited else " · retrieved, not cited"
                st.markdown(f"**{label}**")
                st.write(source.text)
                st.caption(f"ChromaDB cosine distance: {source.distance:.4f} (lower means closer)")
        if result.prompt_preview:
            with st.expander("Inspect exact prompt and retrieved context sent for generation"):
                st.code(result.prompt_preview, language="text")

if st.session_state.get("question_error"):
    st.error(st.session_state["question_error"])
    st.caption("If the error says GROQ_API_KEY is missing, copy .env.example to .env and add your Groq API key.")

with st.form("question_form", clear_on_submit=True):
    st.text_input(
        "Ask a question",
        placeholder="What is the maximum meal reimbursement?",
        key="question_input",
        disabled=not bool(indexed_sources),
    )
    st.form_submit_button("Ask", on_click=submit_question, disabled=not bool(indexed_sources))

st.divider()
with st.expander("How token savings work (and what this demo measures)"):
    st.markdown(
        "Without RAG, an application might send the entire document on every question. With RAG, this app sends "
        "the question plus the top retrieved passages instead. The index is built once and reused, so repeated "
        "questions can avoid repeatedly including the entire document. Embedding/indexing still has a cost, and "
        "retrieval can miss relevant passages. Word-based token estimates are approximate; when available, the "
        "LLM provider's reported input/output token counts are the actual usage for that call. No fixed cost "
        "saving is assumed."
    )

st.caption(
    "One indexed PDF at a time. No hidden sample text is added. Uploaded PDFs are not used to train the embedding "
    "model or the LLM. This educational demo is not financial advice."
)
