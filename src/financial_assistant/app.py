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
st.caption("Ask questions about financial policies and receipt images. All extraction and retrieval run locally.")

@st.cache_resource(show_spinner=False)
def get_assistant():
    return create_default_assistant(PROJECT_ROOT)

policy_tab, receipt_tab = st.tabs(["Policy PDF · Q&A", "Receipt image · Q&A"])

with policy_tab:
    st.subheader("Ask a financial policy")
    st.write(
        "The bundled sample policy text is indexed when the local assistant initializes. "
        "Upload and index a PDF to search its contents too."
    )
    sample_pdf = PROJECT_ROOT / "data" / "sample_financial_policy.pdf"
    if sample_pdf.exists():
        st.download_button(
            "Download sample policy PDF",
            data=sample_pdf.read_bytes(),
            file_name="sample_financial_policy.pdf",
            mime="application/pdf",
        )
        st.caption(
            "This compact PDF may produce one chunk. The built-in sample_policy.txt is longer and is indexed "
            "automatically as multiple overlapping chunks, so you can test retrieval even before uploading a PDF."
        )

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

    question = st.text_input(
        "Your policy question",
        placeholder="What is the maximum meal reimbursement?",
        key="policy_question",
    )
    if st.button("Ask about policy", disabled=not question.strip(), key="ask_policy"):
        try:
            with st.spinner("Embedding your question and searching the local ChromaDB index..."):
                result = get_assistant().ask(question)
            if result.supported and result.sources:
                st.success("Relevant evidence found. This demo returns source passages rather than generated summaries.")
                for index, source in enumerate(result.sources, start=1):
                    label = Path(source.source).name
                    if source.page:
                        label += f" · page {source.page}"
                    st.markdown(f"**Passage {index} — {label}**")
                    st.write(source.text)
                    st.caption(f"Cosine distance: {source.distance:.3f} (lower is closer)")
            else:
                st.warning(result.answer)
        except Exception as exc:
            st.error(f"Could not search the documents: {exc}")

with receipt_tab:
    st.subheader("Ask about a receipt or statement")
    st.write("Upload an image, extract the visible text, then ask a question about that text.")
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
            st.session_state["image_question"] = ""
        st.image(image, caption="Selected image", use_container_width=True)

        if st.button("Read image text", key="read_image"):
            try:
                with st.spinner("Reading visible text with local OCR..."):
                    extracted = read_image_text(raw_image, PROJECT_ROOT)
                st.session_state["ocr_text"] = extracted
                if extracted:
                    st.success("Text extracted. Ask a question below.")
                else:
                    st.warning("No readable text was detected. Try a sharper, brighter image.")
            except Exception as exc:
                st.error(f"Could not read this image: {exc}")

        if st.session_state.get("ocr_text"):
            st.text_area(
                "Recognized text (OCR)",
                value=st.session_state["ocr_text"],
                height=160,
                disabled=True,
                key="ocr_text_display",
            )
            image_question = st.text_input(
                "Your question about this image",
                placeholder="What is the total amount?",
                key="image_question",
            )
            if st.button(
                "Answer from image text",
                disabled=not image_question.strip(),
                key="ask_image",
            ):
                try:
                    result = answer_image_question(image_question, st.session_state["ocr_text"])
                    if result["supported"]:
                        st.success(result["answer"])
                        st.caption("Answer method: extractive keyword matching over OCR text; no LLM is used.")
                        with st.expander("Evidence used"):
                            for passage in result["evidence"]:
                                st.write(passage)
                    else:
                        st.warning(result["answer"])
                except Exception as exc:
                    st.error(f"Could not answer from the image text: {exc}")
    else:
        st.info("Upload a receipt image to begin.")

st.divider()
st.caption(
    "Local-first demo: PDF text extraction, overlapping chunks, sentence embeddings, ChromaDB retrieval, "
    "and OCR. OCR can misread digits; verify financial figures against the original. Not financial advice."
)
