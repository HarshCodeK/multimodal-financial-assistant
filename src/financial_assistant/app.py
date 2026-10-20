import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from financial_assistant.extract import image_bytes_to_text
from financial_assistant.services import create_default_assistant

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Multimodal Financial Assistant", page_icon="💰", layout="centered")
st.title("Multimodal Financial Assistant")
st.caption(
    "Local PDF text extraction, overlapping RAG chunks, sentence embeddings, and OCR. "
    "No API key or separate server is required."
)


@st.cache_resource(show_spinner=False)
def get_assistant():
    return create_default_assistant(PROJECT_ROOT)


st.info(
    "The embedding model loads only when you first index a PDF or ask a question. "
    "That first action can take a few minutes while the model downloads; later actions reuse it."
)

st.subheader("1. Add a financial policy PDF")
st.write(
    "The included sample policy is indexed automatically when you first ask a question or index a PDF. "
    "Uploads are split into overlapping chunks so relevant passages can be retrieved."
)
sample_pdf = PROJECT_ROOT / "data" / "sample_financial_policy.pdf"
if sample_pdf.exists():
    st.download_button(
        "Download sample financial policy PDF",
        data=sample_pdf.read_bytes(),
        file_name="sample_financial_policy.pdf",
        mime="application/pdf",
    )
policy_pdf = st.file_uploader("Choose a PDF (text-based or scanned)", type=["pdf"], key="policy_pdf")
if st.button("Index PDF", disabled=policy_pdf is None):
    try:
        with st.spinner("Loading the local embedding model if needed, extracting text/OCR, chunking, and indexing..."):
            assistant = get_assistant()
            result = assistant.index_pdf(policy_pdf.name, policy_pdf.getvalue())
        st.success(
            f"Indexed {result['chunks_indexed']} overlapping chunks from "
            f"{result['pages_extracted']} pages in {result['source']}."
        )
    except Exception as exc:
        st.error(f"Could not index this PDF: {exc}")

st.divider()
st.subheader("2. Ask a question")
question = st.text_input(
    "Question about the indexed documents",
    placeholder="What is the maximum meal reimbursement?",
    key="question",
)
if st.button("Ask question", disabled=not question.strip()):
    try:
        with st.spinner("Loading local embeddings if needed and retrieving relevant passages..."):
            assistant = get_assistant()
            result = assistant.ask(question)
        st.write(result.answer)
        if result.supported and result.sources:
            st.subheader("Retrieved evidence")
            for source in result.sources:
                label = Path(source.source).name
                if source.page:
                    label += f" — page {source.page}"
                st.caption(label)
                st.write(source.text)
        else:
            st.info("No sufficiently relevant evidence was found. Try a more specific question or index another document.")
    except Exception as exc:
        st.error(f"Could not search the documents: {exc}")

st.divider()
st.subheader("3. Read a receipt or statement image")
st.caption("OCR runs locally. It extracts visible text; it does not invent missing figures or require a vision API.")
image = st.file_uploader(
    "Choose a PNG or JPEG image",
    type=["png", "jpg", "jpeg"],
    key="image",
)
if image:
    st.image(image, caption="Image selected", use_container_width=True)
    if st.button("Extract text from image"):
        try:
            with st.spinner("Loading local OCR if needed and reading the image..."):
                text = image_bytes_to_text(image.getvalue())
            if text:
                st.subheader("Recognized text")
                st.text_area("OCR output", text, height=220)
            else:
                st.warning("No readable text was detected. Try a sharper image with better lighting.")
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")

st.divider()
st.caption(
    "All extraction and retrieval run locally. First-time downloads may be needed for the embedding and OCR models. "
    "OCR can misread digits; verify totals against the original. This demo is not financial advice."
)
