import os
import tempfile
import streamlit as st
from src.document_parser import load_document
from src.vision_extractor import extract_fields
from src.qa_engine import answer_question
from src.monitor import get_recent_logs

st.set_page_config(page_title="Financial Assistant", layout="wide")
st.title("Multimodal Financial Assistant")

with st.sidebar:
    st.header("Recent Queries")
    rows = get_recent_logs(5)
    for row in rows:
        st.text(f"[{row[1]}] {row[3][:40]}...")
        if st.button(f"View", key=f"log_{row[0]}"):
            st.session_state["log_detail"] = row

if "log_detail" in st.session_state:
    r = st.session_state["log_detail"]
    with st.expander("Last viewed log", expanded=True):
        st.write(f"**File:** {r[2]} | **Question:** {r[3]} | **Latency:** {r[6]:.0f}ms")
        st.write(f"**Answer:** {r[4]}")

uploaded = st.file_uploader("Upload a financial document", type=["pdf", "jpg", "jpeg", "png"])

if uploaded is not None:
    if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.session_state["tmp_path"] = tmp_path
        st.session_state["uploaded_name"] = uploaded.name
        st.session_state["doc"] = load_document(tmp_path)

    doc = st.session_state["doc"]
    if doc["type"] == "image":
        st.image(doc["path"], caption=uploaded.name, width=400)
    else:
        st.text_area("Extracted text (PDF)", doc["text"][:2000], height=200)

    if st.button("Extract Fields"):
        with st.spinner("Extracting fields..."):
            fields = extract_fields(doc)
        st.session_state["fields"] = fields

    if "fields" in st.session_state:
        st.json(st.session_state["fields"])

    st.text_input("Question", value="Why was this charge deducted?", key="question")
    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            result = answer_question(doc, st.session_state.question)
        st.subheader("Answer")
        st.write(result["answer"])
        with st.expander("Policy sources used"):
            for i, chunk in enumerate(result["policy_sources_used"]):
                st.text(f"[{i}] {chunk}")
