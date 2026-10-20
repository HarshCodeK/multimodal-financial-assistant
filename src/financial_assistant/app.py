import sys
from pathlib import Path
import os

import streamlit as st

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from financial_assistant.services import create_default_assistant

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Multimodal Financial Assistant", page_icon="💰", layout="centered")
st.title("Multimodal Financial Assistant")
st.caption(
    "Ask questions about financial-policy PDFs and inspect receipts or statements. "
    "This demo runs locally in one Streamlit process."
)


@st.cache_resource(show_spinner="Loading the embedding model and local document index...")
def get_assistant():
    return create_default_assistant(PROJECT_ROOT)


with st.sidebar:
    st.subheader("Configuration")
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key detected")
    else:
        st.warning("Set GROQ_API_KEY before asking questions or inspecting images.")
    st.caption("PDF extraction uses selectable text. Scanned PDFs are not OCR-processed.")
    st.caption("Indexed documents and interaction logs are stored locally.")

try:
    assistant = get_assistant()
except Exception as exc:
    st.error(f"Could not initialize the local assistant: {exc}")
    st.stop()

st.subheader("1. Add a financial policy PDF")
st.write(
    "A small sample policy is indexed automatically on first launch. Upload a PDF to add "
    "or replace a document in your local index. Re-uploading the same filename replaces its chunks."
)
sample_pdf = PROJECT_ROOT / "data" / "sample_financial_policy.pdf"
if sample_pdf.exists():
    st.download_button(
        "Download sample financial policy PDF",
        data=sample_pdf.read_bytes(),
        file_name="sample_financial_policy.pdf",
        mime="application/pdf",
    )
policy_pdf = st.file_uploader("Choose a text-based PDF", type=["pdf"], key="policy_pdf")
if st.button("Index PDF", disabled=policy_pdf is None):
    try:
        with st.spinner("Extracting text and updating the local vector index..."):
            result = assistant.index_pdf(policy_pdf.name, policy_pdf.getvalue())
        st.success(
            f"Indexed {result['chunks_indexed']} text chunks from "
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
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            result = assistant.ask(question)
        st.write(result.answer)
        if result.supported and result.sources:
            st.subheader("Evidence used")
            for source in result.sources:
                label = Path(source.source).name
                if source.page:
                    label += f" — page {source.page}"
                st.caption(label)
                st.write(source.text)
        else:
            st.info("No supporting document evidence was returned.")
    except Exception as exc:
        st.error(f"Could not answer this question: {exc}")

st.divider()
st.subheader("3. Read a receipt or statement image")
image = st.file_uploader(
    "Choose a PNG or JPEG image",
    type=["png", "jpg", "jpeg"],
    key="image",
)
image_question = st.text_input(
    "What should the assistant look for? (optional)",
    placeholder="Extract the date, line items, tax, and total; flag uncertain text.",
    key="image_question",
)
if image:
    st.image(image, caption="Image selected", use_container_width=True)
    if st.button("Inspect image"):
        try:
            with st.spinner("Sending the image to the Groq vision model..."):
                answer = assistant.inspect_image(image.getvalue(), image_question)
            st.write(answer)
        except Exception as exc:
            st.error(f"Could not inspect this image: {exc}")

st.divider()
st.caption(
    "Portfolio demo only; not financial advice. Vision models can misread receipt text and totals. "
    "Scanned PDFs need OCR, which is not included in this version."
)
