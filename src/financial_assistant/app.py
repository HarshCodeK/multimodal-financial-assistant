"""Streamlit UI for PDF RAG exploration and separate image question answering."""

import json
import os
import sys
from time import perf_counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROQ_KEY_WAS_SET_BEFORE_DOTENV = bool(os.environ.get("GROQ_API_KEY"))
load_dotenv(PROJECT_ROOT / ".env")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from financial_assistant.costing import (
    DEFAULT_TEXT_RATES,
    DEFAULT_VISION_RATES,
    GOOGLE_VISION_OCR_USD_PER_IMAGE,
    GROQ_QWEN_IMAGE_INPUT_TOKENS,
    TokenRates,
    estimate_ocr_then_text_cost_usd,
    estimate_vision_cost_usd,
    token_cost_usd,
)
from financial_assistant.diagnostics import friendly_groq_error, groq_key_diagnostics
from financial_assistant.services import create_default_assistant
from financial_assistant.vision import answer_image_question

st.set_page_config(page_title="Multimodal Financial Assistant", page_icon="📄", layout="wide")
st.title("Multimodal Financial Assistant")
st.caption(
    "Explore text-based PDF RAG, or ask a separate vision model about an uploaded image. "
    "The two workflows are intentionally separate."
)

key_status = groq_key_diagnostics(PROJECT_ROOT, GROQ_KEY_WAS_SET_BEFORE_DOTENV)
with st.expander("Groq API configuration & troubleshooting", expanded=False):
    st.write(f"Project .env file found: {'yes' if key_status['env_file_exists'] else 'no'}")
    st.write(f"Key configured: {'yes' if key_status['configured'] else 'no'}")
    st.write(f"Effective key source: {key_status['source']}")
    st.write(f"Key length: {key_status['key_length'] or 'not configured'} characters (the key itself is never displayed)")
    if key_status["has_outer_whitespace"]:
        st.warning("The environment-variable key has surrounding whitespace. The app trims it before use.")
    if not key_status["configured"]:
        st.warning(
            f"Expected key location: {PROJECT_ROOT / '.env'}. Add GROQ_API_KEY=your_actual_key_here, "
            "save, and restart Streamlit. .env.example is only a template."
        )
    st.caption(
        "A configured key is not necessarily valid. If Groq returns HTTP 401, check for an old Windows/user/system "
        "GROQ_API_KEY variable (it overrides .env), recreate the key in Groq Console if needed, and restart the app."
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


with st.sidebar:
    st.header("How it works")
    st.markdown(
        "**PDF RAG**\n"
        "1. Extract selectable PDF text locally.\n"
        "2. Chunk text and create embeddings locally.\n"
        "3. Retrieve relevant passages from ChromaDB.\n"
        "4. Send the question and retrieved passages to Groq for an answer."
    )
    st.divider()
    st.markdown(
        "**Image Q&A**\n"
        "The uploaded image and question are sent directly to a Groq vision model. "
        "Images are not added to the PDF index."
    )
    st.warning(
        "Privacy: PDF extraction, embeddings and retrieval run locally. PDF questions plus retrieved excerpts, "
        "and any image plus its question, are sent to Groq. Do not upload sensitive material unless that transfer is acceptable."
    )

pdf_tab, image_tab = st.tabs(["📄 PDF RAG Explorer", "🖼️ Ask about an image"])

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
            help="PDF only, up to the extraction service's 10 MiB limit. Images are uploaded in the separate image tab.",
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
            if result.prompt_tokens is not None and result.completion_tokens is not None:
                estimated_usd = token_cost_usd(result.prompt_tokens, result.completion_tokens, DEFAULT_TEXT_RATES)
                stats.append(f"estimated API cost: ${estimated_usd:.8f} USD")
            if stats:
                st.caption(" · ".join(stats))
                st.caption(
                    "Cost uses the default GPT-OSS-20B published rates ($0.075/M input, $0.30/M output); "
                    "it is an estimate, not an invoice or account-credit balance."
                )

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
    st.subheader("Ask a question about an image")
    st.write(
        "Upload a receipt, statement screenshot, chart, or other image and ask a question about its visible contents. "
        "This uses a vision-capable LLM directly; it does not run OCR locally and does not add the image to the PDF RAG index."
    )
    image_col, question_col = st.columns([1, 1], gap="large")
    with image_col:
        uploaded_image = st.file_uploader(
            "Choose an image (PNG, JPEG, or WebP; max 5 MiB)",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_upload_main",
            help="The image is sent to Groq for visual analysis when you click Analyze image.",
        )
        if uploaded_image is not None:
            st.image(uploaded_image.getvalue(), caption=uploaded_image.name, use_container_width=True)
    with question_col:
        image_question = st.text_area(
            "What do you want to know about this image?",
            placeholder="Read the total amount and date. If either is unclear, say so.",
            key="image_question",
            height=120,
        )
        analyze_clicked = st.button(
            "Analyze image",
            type="primary",
            disabled=uploaded_image is None or not image_question.strip(),
            use_container_width=True,
        )
        if analyze_clicked and uploaded_image is not None:
            try:
                started = perf_counter()
                with st.spinner("Sending the image and question to the Groq vision model..."):
                    vision_result = answer_image_question(
                        image_bytes=uploaded_image.getvalue(),
                        filename=uploaded_image.name,
                        question=image_question,
                    )
                vision_result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
                st.session_state["last_image_result"] = {
                    "filename": uploaded_image.name,
                    "question": image_question.strip(),
                    **vision_result,
                }
                image_cost = estimate_vision_cost_usd(
                    vision_result.get("prompt_tokens"), vision_result.get("completion_tokens")
                )
                get_assistant().store.record(
                    question=image_question.strip(),
                    supported=True,
                    answer=vision_result["answer"],
                    latency_ms=vision_result["latency_ms"],
                    workflow="image_vision",
                    model=vision_result.get("model"),
                    prompt_tokens=vision_result.get("prompt_tokens"),
                    completion_tokens=vision_result.get("completion_tokens"),
                    estimated_cost_usd=image_cost,
                    input_chars=len(image_question),
                    input_words=len(image_question.split()),
                    input_bytes=len(uploaded_image.getvalue()),
                )
            except Exception as exc:
                st.error(friendly_groq_error(exc, "image analysis"))

    image_result = st.session_state.get("last_image_result")
    if image_result:
        st.divider()
        st.markdown(f"**Question:** {image_result['question']}")
        st.markdown(f"**Image:** {image_result['filename']}")
        st.markdown("### Vision model response")
        st.write(image_result["answer"])
        token_stats = []
        if image_result.get("model"):
            token_stats.append(f"Model: {image_result['model']}")
        if image_result.get("prompt_tokens") is not None:
            token_stats.append(f"input tokens: {image_result['prompt_tokens']}")
        if image_result.get("completion_tokens") is not None:
            token_stats.append(f"output tokens: {image_result['completion_tokens']}")
        if token_stats:
            st.caption(" · ".join(token_stats))
        st.warning(
            "Vision-model output is not guaranteed to be correct. Verify amounts, dates, account details, and other "
            "financial values against the original image."
        )

        st.subheader("Usage and estimated cost")
        vision_usage_cost = estimate_vision_cost_usd(
            image_result.get("prompt_tokens"), image_result.get("completion_tokens")
        )
        if vision_usage_cost is not None:
            st.metric("Estimated vision API cost for this call", f"\${vision_usage_cost:.8f} USD")
            st.caption(
                "Calculated from provider-reported input/output tokens and default Qwen3.6-27B rates "
                "($0.60/M input, $3.00/M output). Published-rate estimate, not an invoice."
            )
        else:
            st.info(
                f"The provider did not return usage. A rough estimate starts at {GROQ_QWEN_IMAGE_INPUT_TOKENS:,} "
                "image input tokens plus the question and answer; adjust assumptions in the comparison below."
            )

        with st.expander("Compare direct vision vs OCR + text LLM", expanded=True):
            st.write(
                "This is a cost calculator, not a live OCR implementation. Direct vision sends the image to Qwen. "
                "The alternative assumes cloud OCR extracts text first, then GPT-OSS-20B answers from that text. "
                "OCR can lose layout, handwriting, chart structure, or visual context."
            )
            rate_col1, rate_col2 = st.columns(2)
            with rate_col1:
                vision_input_rate = st.number_input(
                    "Vision input price (USD / 1M tokens)", min_value=0.0,
                    value=DEFAULT_VISION_RATES.input_per_million, step=0.05, format="%.4f",
                    key="vision_input_rate",
                )
                vision_output_rate = st.number_input(
                    "Vision output price (USD / 1M tokens)", min_value=0.0,
                    value=DEFAULT_VISION_RATES.output_per_million, step=0.25, format="%.4f",
                    key="vision_output_rate",
                )
                text_input_rate = st.number_input(
                    "Text LLM input price (USD / 1M tokens)", min_value=0.0,
                    value=DEFAULT_TEXT_RATES.input_per_million, step=0.01, format="%.4f",
                    key="text_input_rate",
                )
                text_output_rate = st.number_input(
                    "Text LLM output price (USD / 1M tokens)", min_value=0.0,
                    value=DEFAULT_TEXT_RATES.output_per_million, step=0.05, format="%.4f",
                    key="text_output_rate",
                )
            with rate_col2:
                ocr_price = st.number_input(
                    "Cloud OCR price per image (USD)", min_value=0.0,
                    value=GOOGLE_VISION_OCR_USD_PER_IMAGE, step=0.0005, format="%.6f",
                    key="ocr_price",
                )
                ocr_text_tokens = st.number_input(
                    "Estimated OCR text tokens", min_value=0, max_value=200000, value=300, step=50,
                    key="ocr_text_tokens",
                )
                assumed_output_tokens = st.number_input(
                    "Assumed answer output tokens", min_value=1, max_value=8000,
                    value=int(image_result.get("completion_tokens") or 100), step=10,
                    key="comparison_output_tokens",
                )
                prompt_overhead = st.number_input(
                    "Text LLM instructions/question overhead tokens", min_value=0, max_value=10000,
                    value=250, step=25, key="comparison_prompt_overhead",
                )
                free_ocr_tier = st.checkbox(
                    "Assume Google Cloud Vision's first 1,000 OCR units/month are still free",
                    value=False, key="ocr_free_tier",
                )
            vision_rates = TokenRates(vision_input_rate, vision_output_rate)
            text_rates = TokenRates(text_input_rate, text_output_rate)
            vision_input_for_estimate = (
                image_result.get("prompt_tokens")
                if image_result.get("prompt_tokens") is not None
                else GROQ_QWEN_IMAGE_INPUT_TOKENS + round(len(image_result["question"].split()) * 1.33)
            )
            vision_output_for_estimate = (
                image_result.get("completion_tokens")
                if image_result.get("completion_tokens") is not None
                else int(assumed_output_tokens)
            )
            vision_estimate = token_cost_usd(
                int(vision_input_for_estimate), int(vision_output_for_estimate), vision_rates
            )
            ocr_estimate = estimate_ocr_then_text_cost_usd(
                int(ocr_text_tokens),
                int(assumed_output_tokens),
                text_rates=text_rates,
                ocr_cost_usd=0.0 if free_ocr_tier else ocr_price,
                prompt_overhead_tokens=int(prompt_overhead),
            )
            compare_a, compare_b = st.columns(2)
            compare_a.metric("Direct vision estimate", f"\${vision_estimate:.8f} USD")
            compare_b.metric("OCR + text LLM estimate", f"\${ocr_estimate:.8f} USD")
            if ocr_estimate < vision_estimate:
                st.success(
                    f"Under these assumptions, OCR + text LLM is about "
                    f"{vision_estimate / max(ocr_estimate, 1e-12):.2f}× cheaper."
                )
            elif vision_estimate < ocr_estimate:
                st.success(
                    f"Under these assumptions, direct vision is about "
                    f"{ocr_estimate / max(vision_estimate, 1e-12):.2f}× cheaper."
                )
            else:
                st.info("The two estimates are equal under these assumptions.")
            st.caption(
                "Default rates: Groq Qwen3.6-27B $0.60/M input and $3/M output; Groq GPT-OSS-20B "
                "$0.075/M input and $0.30/M output; Google Cloud Vision OCR $1.50 per 1,000 images after its "
                "monthly free tier. These are editable published-rate assumptions, not actual billing records. "
                "Image tokenization and OCR free-tier usage can change the real bill."
            )

st.caption(
    "PDF RAG and image Q&A are separate workflows: PDF content is indexed for retrieval; images are analyzed directly "
    "per request and are not persisted in ChromaDB."
)

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
        st.info("No request records yet. Ask a PDF question or analyze an image to create a usage record.")
