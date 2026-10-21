"""Streamlit UI for PDF RAG and local-OCR image question answering."""

import json
import os
import sys
from time import perf_counter
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Prefer a usable key in this project's .env over stale inherited Windows variables.
_project_values = dotenv_values(PROJECT_ROOT / ".env")
_project_key = str(_project_values.get("GROQ_API_KEY") or "").strip().strip("'\"")
if _project_key and _project_key != "replace_with_your_groq_api_key":
    os.environ["GROQ_API_KEY"] = _project_key
else:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from financial_assistant.costing import DEFAULT_TEXT_RATES, token_cost_usd
from financial_assistant.diagnostics import friendly_groq_error, groq_key_diagnostics
from financial_assistant.ocr import answer_ocr_text, extract_image_text
from financial_assistant.services import create_default_assistant

st.set_page_config(page_title="Financial Assistant", page_icon="📄", layout="wide")
st.title("Financial Assistant")
st.caption(
    "Inspect PDF retrieval evidence and analyze text in images with local OCR. "
    "Both workflows use the same Groq text model: openai/gpt-oss-20b."
)

key_status = groq_key_diagnostics(PROJECT_ROOT)
with st.expander("Groq API configuration & troubleshooting", expanded=False):
    st.write(f"Project .env file found: {'yes' if key_status['env_file_exists'] else 'no'}")
    st.write(f"Key configured: {'yes' if key_status['configured'] else 'no'}")
    st.write(f"Effective key source: {key_status['source']}")
    st.write(f"Key length: {key_status['key_length'] or 'not configured'} characters (the key itself is never displayed)")
    if key_status["has_outer_whitespace"]:
        st.warning("The configured key has surrounding whitespace. The app trims it before use.")
    if not key_status["configured"]:
        st.warning(
            f"Expected key location: {PROJECT_ROOT / '.env'}. Add GROQ_API_KEY=your_actual_key_here, "
            "save, and restart Streamlit. .env.example is only a template."
        )
    st.caption(
        "A usable key in this project's .env takes precedence over inherited Windows/user/system variables. "
        "Restart Streamlit after changing configuration. Key validity is checked by Groq when a request is made."
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
        st.session_state["question_error"] = friendly_groq_error(exc, "PDF answer")


pdf_tab, image_tab = st.tabs(["📄 PDF RAG Explorer", "🧾 OCR image Q&A"])

with pdf_tab:
    st.subheader("1. Upload and index a PDF")
    st.write(
        "Upload one PDF with selectable text. Indexing a new PDF replaces the current PDF index. "
        "Scanned/image-only PDFs are not OCR-processed in this workflow."
    )
    upload_col, sample_col = st.columns([3, 2], gap="large")
    with upload_col:
        uploaded_pdf = st.file_uploader(
            "Choose a text-based PDF",
            type=["pdf"],
            key="pdf_upload_main",
            help="PDF only, up to the extraction service's 10 MiB limit. Scanned pages may require local OCR.",
        )
        if st.button(
            "Extract, chunk, embed & index PDF",
            disabled=uploaded_pdf is None,
            type="primary",
            use_container_width=True,
        ):
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
    with sample_col:
        sample_pdf_path = PROJECT_ROOT / "data" / "sample_financial_policy.pdf"
        st.markdown("**Want a safe test document?**")
        if sample_pdf_path.exists():
            st.download_button(
                "Download fictional sample PDF",
                data=sample_pdf_path.read_bytes(),
                file_name="sample_financial_policy.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("Fictional policy for testing only. Downloading does not index it; upload it above.")
        st.caption("No bundled document is indexed automatically.")

    assistant = get_assistant()
    indexed_sources = assistant.indexed_sources()
    if indexed_sources:
        st.success("Currently indexed PDF: " + ", ".join(indexed_sources))
        if st.button("Clear indexed PDF", help="Permanently remove the current local PDF index from ChromaDB. This does not delete your original PDF file."):
            assistant.clear_index()
            st.session_state.pop("last_index_result", None)
            st.session_state["chat_history"] = []
            st.rerun()
    else:
        st.info("No PDF is indexed yet. Upload a PDF above and click the indexing button.")

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
        with st.expander("What do document tokens and 384 vector dimensions mean?"):
            st.markdown(
                "**Approx. document tokens** estimates full-document text size using words × 1.33. It is not an "
                "exact tokenizer count, and it is not the number sent on every question. RAG retrieves only a few "
                "passages, which can reduce repeated input context.\n\n"
                "**Vector dimensions = 384** means each chunk is represented by 384 numeric values. ChromaDB "
                "compares these vectors to find semantically similar text. They are not 384 words or a confidence "
                "score. Every chunk and query must use compatible vectors from the same embedding model."
            )
        with st.expander("Inspect the embedding vector"):
            st.write(
                "The pretrained embedding model turns a chunk into a fixed-length numeric vector. "
                "The vector is a learned semantic representation, not a human-readable summary and not training on this PDF."
            )
            st.write(f"Dimensions: {summary['embedding_dimensions']}")
            st.code(
                str(summary["embedding_preview"])
                + (" ..." if summary["embedding_dimensions"] and summary["embedding_dimensions"] > 8 else "")
            )

    st.subheader("2. Ask the indexed PDF")
    for entry in st.session_state.get("chat_history", []):
        result = entry["result"]
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown("**Retrieved passages selected for this question**")
            st.caption(
                "These are the exact chunks retrieved from the indexed PDF and supplied to the LLM as evidence. "
                "Review them first; the answer below must be grounded in this evidence."
            )
            if not result.sources:
                st.info("No passages were retrieved, so no document-grounded answer can be generated.")
            for index, source in enumerate(result.sources, start=1):
                source_id = f"S{index}"
                cited = source_id in result.citations
                label = f"[{source_id}] {Path(source.source).name}"
                if source.page:
                    label += f" · page {source.page}"
                label += " · cited by answer" if cited else " · retrieved"
                st.markdown(f"**{label}**")
                st.write(source.text)
                st.caption(f"ChromaDB cosine distance: {source.distance:.4f} (lower ranks as closer)")

            st.divider()
            st.markdown("**LLM answer based on the passages above**")
            if result.supported:
                st.markdown(result.answer)
                if result.citations:
                    st.caption("Answer citations: " + ", ".join(f"[{item}]" for item in result.citations))
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
            if result.prompt_tokens is not None and result.completion_tokens is not None:
                estimated_usd = token_cost_usd(result.prompt_tokens, result.completion_tokens, DEFAULT_TEXT_RATES)
                stats.append(f"estimated API cost: ${estimated_usd:.8f} USD")
            if stats:
                st.caption(" · ".join(stats))
                st.caption(
                    "Cost uses the default GPT-OSS-20B published rates ($0.075/M input, $0.30/M output); "
                    "it is an estimate, not an invoice or account-credit balance."
                )

            if result.prompt_preview:
                with st.expander("Inspect exact prompt and retrieved context sent for generation"):
                    st.code(result.prompt_preview, language="text")

    if st.session_state.get("question_error"):
        st.error(st.session_state["question_error"])
        st.caption("Open the Groq API configuration & troubleshooting panel above. API keys are never printed in the UI.")

    with st.form("question_form", clear_on_submit=True):
        st.text_input(
            "Ask a question",
            placeholder="What is the maximum meal reimbursement?",
            key="question_input",
            disabled=not bool(indexed_sources),
        )
        st.form_submit_button("Ask", on_click=submit_question, disabled=not bool(indexed_sources))

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

with image_tab:
    st.subheader("Ask a question about text in an image")
    st.write(
        "OCR runs locally on your computer. Review the extracted text before the question and extracted text are sent "
        "to Groq's openai/gpt-oss-20b text model. The raw image is never sent to Groq and is not added to ChromaDB. "
        "This text-only workflow cannot reliably interpret charts or visual layouts."
    )
    image_col, question_col = st.columns([1, 1], gap="large")
    with image_col:
        uploaded_image = st.file_uploader(
            "Choose an image (PNG, JPEG, or WebP; max 5 MiB)",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_upload_main",
            help="The image is processed locally by OCR. Only extracted text and your question are sent to Groq.",
        )
        if uploaded_image is not None:
            st.image(uploaded_image.getvalue(), caption=uploaded_image.name, use_container_width=True)
    with question_col:
        image_question = st.text_area(
            "What do you want to know about the extracted text?",
            placeholder="What is the total amount and transaction date?",
            key="image_question",
            height=120,
        )
        analyze_clicked = st.button(
            "Run local OCR and answer",
            type="primary",
            disabled=uploaded_image is None or not image_question.strip(),
            use_container_width=True,
        )
        if analyze_clicked and uploaded_image is not None:
            started = perf_counter()
            st.session_state.pop("image_error", None)
            st.session_state.pop("last_image_result", None)
            ocr_complete = False
            try:
                with st.spinner("Reading text locally from the image..."):
                    ocr_result = extract_image_text(uploaded_image.getvalue(), uploaded_image.name)
                ocr_complete = True
                st.session_state["last_image_result"] = {
                    "filename": uploaded_image.name,
                    "question": image_question.strip(),
                    "ocr_text": ocr_result["text"],
                    "ocr_confidence": ocr_result["mean_confidence"],
                }
                with st.spinner("Sending only the question and extracted text to GPT-OSS-20B..."):
                    answer_result = answer_ocr_text(
                        image_question,
                        ocr_result["text"],
                        uploaded_image.name,
                    )
                answer_result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
                st.session_state["last_image_result"].update(answer_result)
                text_cost = (
                    token_cost_usd(
                        answer_result["prompt_tokens"],
                        answer_result["completion_tokens"],
                        DEFAULT_TEXT_RATES,
                    )
                    if answer_result.get("prompt_tokens") is not None
                    and answer_result.get("completion_tokens") is not None
                    else None
                )
                get_assistant().store.record(
                    question=image_question.strip(),
                    supported=answer_result["supported"],
                    answer=answer_result["answer"],
                    latency_ms=answer_result["latency_ms"],
                    workflow="image_ocr_text",
                    model=answer_result.get("model"),
                    prompt_tokens=answer_result.get("prompt_tokens"),
                    completion_tokens=answer_result.get("completion_tokens"),
                    estimated_cost_usd=text_cost,
                    input_chars=len(ocr_result["text"]),
                    input_words=len(ocr_result["text"].split()),
                    input_bytes=len(uploaded_image.getvalue()),
                )
            except Exception as exc:
                if ocr_complete:
                    st.session_state["image_error"] = friendly_groq_error(exc, "OCR image answer")
                else:
                    st.session_state["image_error"] = f"Local OCR failed: {exc}"
    if st.session_state.get("image_error"):
        st.error(st.session_state["image_error"])
        st.caption("If OCR text was extracted, it remains visible below even if Groq answer generation failed.")

    image_result = st.session_state.get("last_image_result")
    if image_result:
        st.divider()
        st.markdown(f"**Image:** {image_result['filename']}")
        st.markdown("### Extracted text (OCR)")
        st.code(image_result["ocr_text"], language="text")
        if image_result.get("ocr_confidence") is not None:
            st.caption(f"Mean OCR confidence: {image_result['ocr_confidence']:.0%} (estimate, not a correctness guarantee)")
        if image_result.get("answer"):
            st.markdown(f"**Question:** {image_result['question']}")
            st.markdown("### GPT-OSS-20B answer")
            if image_result.get("supported"):
                st.write(image_result["answer"])
                if image_result.get("citations"):
                    st.caption("Grounded in extracted text: " + ", ".join(f"[{item}]" for item in image_result["citations"]))
            else:
                st.warning(image_result["answer"])
            stats = [f"Model: {image_result.get('model', 'openai/gpt-oss-20b')}"]
            if image_result.get("prompt_tokens") is not None:
                stats.append(f"input tokens: {image_result['prompt_tokens']}")
            if image_result.get("completion_tokens") is not None:
                stats.append(f"output tokens: {image_result['completion_tokens']}")
            if image_result.get("latency_ms") is not None:
                stats.append(f"total OCR + answer time: {image_result['latency_ms']:.0f} ms")
            st.caption(" · ".join(stats))
            st.warning("OCR can misread amounts, dates, and identifiers. Verify the extracted text against the original image.")

with st.expander("Local usage and estimated-cost log"):
    st.caption(
        "Recent requests are recorded in the local SQLite database. It stores question/answer text, latency, token "
        "usage, estimated USD cost and input lengths, but never the API key or uploaded image bytes."
    )
    usage_rows = get_assistant().store.recent_usage(100)
    if usage_rows:
        st.dataframe(usage_rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Download usage log as JSON",
            data=json.dumps(usage_rows, indent=2, ensure_ascii=False),
            file_name="financial_assistant_usage_log.json",
            mime="application/json",
        )
    else:
        st.info("No request records yet. Ask a PDF question or run OCR image Q&A to create a usage record.")
