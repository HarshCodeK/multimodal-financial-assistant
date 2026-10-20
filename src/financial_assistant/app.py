import hashlib
import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from financial_assistant.services import answer_image_question, create_default_assistant, read_image_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Multimodal Financial Assistant", page_icon="💰", layout="centered")
st.title("Multimodal Financial Assistant")
st.caption("Ask questions about financial policies and receipt images. Extraction and retrieval run locally.")


@st.cache_resource(show_spinner=False)
def get_assistant():
    return create_default_assistant(PROJECT_ROOT)


def submit_policy_question():
    question = st.session_state.get("policy_question", "").strip()
    if not question:
        return
    try:
        with st.spinner("Searching the uploaded policy documents..."):
            result = get_assistant().ask(question)
        st.session_state.setdefault("policy_history", []).append({"question": question, "result": result})
        st.session_state["policy_question"] = ""
        st.session_state.pop("policy_error", None)
    except Exception as exc:
        st.session_state["policy_error"] = f"Could not search the documents: {exc}"


def submit_image_question():
    question = st.session_state.get("image_question", "").strip()
    text = st.session_state.get("ocr_text_editor", "").strip()
    if not question:
        return
    try:
        result = answer_image_question(question, text)
        st.session_state.setdefault("image_history", []).append({"question": question, "result": result})
        st.session_state["image_question"] = ""
        st.session_state.pop("image_error", None)
    except Exception as exc:
        st.session_state["image_error"] = f"Could not answer from the image text: {exc}"


policy_tab, receipt_tab = st.tabs(["Policy PDF · Q&A", "Receipt image · Q&A"])

with policy_tab:
    st.subheader("Ask a financial policy")
    st.write("Upload and index the policy PDF you want to question. Answers can only cite documents you have indexed here; bundled sample text is not automatically included.")
    sample_pdf = PROJECT_ROOT / "data" / "sample_financial_policy.pdf"
    if sample_pdf.exists():
        st.download_button(
            "Download sample policy PDF",
            data=sample_pdf.read_bytes(),
            file_name="sample_financial_policy.pdf",
            mime="application/pdf",
        )
        st.caption("The downloadable PDF is a fictional example. Downloading it does not index it; upload it below and click Index uploaded PDF to use it.")

    policy_pdf = st.file_uploader(
        "Upload a financial policy PDF",
        type=["pdf"],
        key="policy_pdf",
        help="Text PDFs are extracted directly. Scanned pages use local OCR.",
    )
    if st.button("Index uploaded PDF", disabled=policy_pdf is None, key="index_pdf"):
        try:
            with st.spinner("Extracting text, chunking, embedding, and saving passages locally..."):
                result = get_assistant().index_pdf(policy_pdf.name, policy_pdf.getvalue())
            st.session_state["last_index_result"] = result
            st.success(
                f"Indexed {result['chunks_indexed']} chunks from {result['pages_extracted']} pages "
                f"({result['source']})."
            )
        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")

    for entry in st.session_state.get("policy_history", []):
        result = entry["result"]
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            if result.supported and result.sources:
                st.write("Relevant evidence found. This demo returns source passages rather than generated summaries.")
                for index, source in enumerate(result.sources, start=1):
                    label = Path(source.source).name
                    if source.page:
                        label += f" · page {source.page}"
                    st.markdown(f"**Passage {index} — {label}**")
                    st.write(source.text)
                    st.caption(f"Cosine distance: {source.distance:.3f} (lower is closer)")
            else:
                st.warning(result.answer)

    if st.session_state.get("policy_error"):
        st.error(st.session_state["policy_error"])
    with st.form("policy_question_form", clear_on_submit=True):
        st.text_input(
            "Ask your next policy question",
            placeholder="What is the maximum meal reimbursement?",
            key="policy_question",
        )
        st.form_submit_button("Ask about policy", on_click=submit_policy_question)
    st.caption("The next-question box stays below the conversation. Earlier questions and evidence remain visible.")

with receipt_tab:
    st.subheader("Ask about a receipt or statement")
    st.write("Upload an image, extract the visible text, then ask questions about the recognized text.")
    image = st.file_uploader(
        "Upload a receipt or statement image",
        type=["png", "jpg", "jpeg"],
        key="receipt_image",
    )

    if image is not None:
        raw_image = image.getvalue()
        image_digest = hashlib.sha256(raw_image).hexdigest()
        if st.session_state.get("ocr_image_digest") != image_digest:
            st.session_state["ocr_image_digest"] = image_digest
            st.session_state["ocr_text"] = ""
            st.session_state["ocr_text_editor"] = ""
            st.session_state["image_question"] = ""
            st.session_state["image_history"] = []
        st.image(image, caption="Selected image", use_container_width=True)

        if st.button("Read image text", key="read_image"):
            try:
                with st.spinner("Reading visible text with local OCR..."):
                    extracted = read_image_text(raw_image, PROJECT_ROOT)
                st.session_state["ocr_text"] = extracted
                st.session_state["ocr_text_editor"] = extracted
                if extracted:
                    st.success("Text extracted. Check financial figures against the image and correct any OCR mistakes below.")
                else:
                    st.warning("No readable text was detected. Try a sharper, brighter image.")
            except Exception as exc:
                st.error(f"Could not read this image: {exc}")

        if st.session_state.get("ocr_text"):
            st.text_area(
                "Recognized text (editable — correct OCR mistakes before asking)",
                key="ocr_text_editor",
                height=180,
                help="OCR can confuse digits, commas, and currency symbols. If the receipt says 5,020.00 but OCR says 55,020.00, correct this text to match the image before asking.",
            )

            for entry in st.session_state.get("image_history", []):
                result = entry["result"]
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    if result["supported"]:
                        st.write(result["answer"])
                        st.caption("Answer method: extractive matching over the editable OCR text; no LLM is used.")
                        with st.expander("Evidence used"):
                            for passage in result["evidence"]:
                                st.write(passage)
                    else:
                        st.warning(result["answer"])

            if st.session_state.get("image_error"):
                st.error(st.session_state["image_error"])
            with st.form("image_question_form", clear_on_submit=True):
                st.text_input(
                    "Ask your next question about this image",
                    placeholder="What is the total amount?",
                    key="image_question",
                )
                st.form_submit_button("Answer from image text", on_click=submit_image_question)
            st.caption("The next-question box stays below the conversation. Earlier answers stay visible until you change the image.")
    else:
        st.info("Upload a receipt image to begin.")

st.divider()
st.caption(
    "Local-first demo: PDF text extraction, overlapping chunks, sentence embeddings, ChromaDB retrieval, "
    "and OCR. OCR can misread digits; review and correct recognized text against the original. Not financial advice."
)
