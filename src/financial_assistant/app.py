import os
from pathlib import Path
import sys
import base64

import requests
import streamlit as st

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Financial Assistant", page_icon="💰")
st.title("Multimodal Financial Assistant")
st.caption("Index a policy PDF, ask a question, and inspect the evidence used for the answer.")
api_url = os.getenv("ASSISTANT_API_URL", "http://localhost:8000").rstrip("/")


def show_api_error(response):
    if not response.ok:
        try:
            detail = response.json().get("detail", "The API could not complete this request.")
        except ValueError:
            detail = "The API could not complete this request."
        st.error(f"API error {response.status_code}: {detail}")
        return True
    return False


st.subheader("1. Index a policy document")
policy_pdf = st.file_uploader("Upload a text-based PDF policy", type=["pdf"], key="policy_pdf")
if st.button("Index policy PDF", disabled=policy_pdf is None):
    try:
        with st.spinner("Extracting pages and indexing chunks..."):
            response = requests.post(
                f"{api_url}/ingest",
                files={"file": (policy_pdf.name, policy_pdf.getvalue(), "application/pdf")},
                timeout=90,
            )
        if not show_api_error(response):
            result = response.json()
            st.success(f"Indexed {result['chunks_indexed']} chunks from {result['pages_extracted']} text pages.")
    except requests.RequestException:
        st.error("Could not reach the assistant API. Check ASSISTANT_API_URL and service status.")

st.divider()
st.subheader("2. Ask a grounded question")
question = st.text_input("Question", placeholder="What is the meal reimbursement limit?")
if st.button("Ask") and question.strip():
    try:
        response = requests.post(f"{api_url}/ask", json={"question": question}, timeout=60)
        if not show_api_error(response):
            data = response.json()
            st.write(data["answer"])
            if data["sources"]:
                st.subheader("Evidence")
                for source in data["sources"]:
                    label = Path(source["source"]).name + (f" — page {source['page']}" if source.get("page") else "")
                    st.caption(label)
                    st.write(source["text"])
            else:
                st.info("No supporting document evidence was returned.")
    except requests.RequestException:
        st.error("Could not reach the assistant API. Check ASSISTANT_API_URL and service status.")

st.divider()
st.subheader("3. Inspect an image")
image = st.file_uploader("Upload a statement or chart image", type=["png", "jpg", "jpeg"], key="image")
if image:
    st.image(image)
    if st.button("Inspect image"):
        try:
            raw = image.getvalue()
            encoded = base64.b64encode(raw).decode("ascii")
            mime = image.type or "image/jpeg"
            response = requests.post(
                f"{api_url}/describe-image",
                json={"question": "Read the financial information in this image and summarize the relevant figures.", "image_data_url": f"data:{mime};base64,{encoded}"},
                timeout=60,
            )
            if not show_api_error(response):
                st.write(response.json()["answer"])
        except requests.RequestException:
            st.error("Could not reach the assistant API. Check ASSISTANT_API_URL and service status.")

st.caption("This is a portfolio demo, not financial advice. Scanned PDFs are not OCR-processed.")
