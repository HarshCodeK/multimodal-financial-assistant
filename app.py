"""Streamlit front end.

Run:  streamlit run app.py
"""
import json
import os
import tempfile

import streamlit as st

from src import models
from src.answer import answer
from src.extract import ExtractionError, extract
from src.knowledge_base import stats
from src.log_store import recent
from src.parser import load

ACCENT, INK, MUTED, RULE = "#1F7A5C", "#12181F", "#5A6673", "#DCE3E8"

st.set_page_config(page_title="Financial Review", layout="wide", page_icon="◈")
st.markdown(f"""
<style>
  .block-container {{ padding-top:2.2rem; max-width:1060px; }}
  h1, h2, h3, h4 {{ letter-spacing:-0.02em; color:{INK}; }}
  h1 {{ font-size:1.6rem; margin-bottom:.1rem; }}
  .sub {{ color:{MUTED}; font-size:.9rem; margin-bottom:1.4rem; }}
  .rule {{ height:1px; background:{RULE}; border:0; margin:1.3rem 0; }}
  .stat {{ border-left:2px solid {RULE}; padding-left:.85rem; margin-top:.7rem; }}
  .stat .v {{ font-size:.95rem; font-weight:600; }}
  .stat .l {{ font-size:.7rem; color:{MUTED}; text-transform:uppercase;
              letter-spacing:.06em; margin-top:.1rem; }}
  .cite {{ font-size:.82rem; color:{MUTED}; border-left:2px solid {ACCENT};
           padding-left:.7rem; margin:.3rem 0; }}
  .stButton>button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}
</style>""", unsafe_allow_html=True)

st.title("Financial document review")
st.markdown('<div class="sub">A vision model extracts the fields, ChromaDB retrieves the '
            "governing policy, and the answer must cite the document it came from. "
            "When the policy does not cover the question, it says so.</div>",
            unsafe_allow_html=True)

has_key = bool(os.environ.get("GROQ_API_KEY"))
ids = list(models.MODELS)
with st.sidebar:
    st.markdown("#### Model")
    chosen = st.selectbox("model", ids, index=ids.index(models.DEFAULT_MODEL),
                          format_func=lambda m: f"{models.MODELS[m][0]} — {m}",
                          label_visibility="collapsed")
    label, vision, pin, pout = models.MODELS[chosen]
    st.caption(f"${pin:.3f} in / ${pout:.2f} out per 1M tokens")
    st.caption("Reads images — required for photo receipts." if vision
               else "Text only. PDFs only, not images.")
    if not has_key:
        st.error("No GROQ_API_KEY. Copy .env.example to .env and add your key.")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("#### Policy index")
    try:
        s = stats()
        st.markdown(f'<div class="stat"><div class="v">{s["chunks"]} chunks</div>'
                    f'<div class="l">{s["store"]}</div></div>'
                    f'<div class="stat"><div class="v" style="font-size:.8rem">'
                    f'{s["embeddings"]}</div><div class="l">embeddings</div></div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"Not built yet ({type(e).__name__}). Builds on first question.")

upload = st.file_uploader("Drop a PDF, JPG or PNG", type=["pdf","jpg","jpeg","png"],
                          label_visibility="collapsed")
if upload is None:
    st.markdown('<div class="sub" style="margin-top:1.6rem">PDF text is parsed locally with '
                "PyMuPDF; images go to the model as base64. Nothing is stored beyond the "
                "local SQLite log.</div>", unsafe_allow_html=True)
    st.stop()

if st.session_state.get("fname") != upload.name:
    suffix = os.path.splitext(upload.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.read()); tmp.close()
    try:
        st.session_state["doc"] = load(tmp.name)
        st.session_state["fname"] = upload.name
        st.session_state.pop("fields", None)
        st.session_state.pop("result", None)
    except ValueError as e:
        st.error(str(e)); st.stop()

doc = st.session_state["doc"]
left, right = st.columns([1,1], gap="large")

with left:
    if doc["type"] == "image":
        st.image(doc["path"], width=360)
    else:
        st.text_area("Extracted PDF text", doc["text"][:2000], height=220,
                     disabled=True, label_visibility="collapsed")

with right:
    if st.button("Extract fields", type="primary", use_container_width=True, disabled=not has_key):
        try:
            with st.spinner(f"Extracting with {chosen}..."):
                st.session_state["fields"] = extract(doc, chosen)
        except ExtractionError as e:
            st.error(str(e))

    fields = st.session_state.get("fields")
    if fields:
        st.markdown(f"**{fields.get('vendor') or 'Unknown vendor'}**")
        st.markdown(f'<div class="sub" style="margin:.1rem 0 .6rem">'
                    f'{fields.get("date") or "no date"} · total <strong>{fields.get("total")}</strong></div>',
                    unsafe_allow_html=True)
        with st.expander("All extracted fields"):
            st.json(fields)

    question = st.text_input("Ask about this charge", value="Why was this charge deducted?",
                             label_visibility="collapsed")
    if st.button("Ask", use_container_width=True, disabled=not has_key):
        from src import log_store
        try:
            with st.spinner("Retrieving policy context and answering..."):
                r = answer(doc, question, chosen)
            log_store.record(doc["filename"], question, r["answer"], r["fields"],
                             r["sources"], r["model"], r["elapsed_ms"])
            st.session_state["result"] = r
        except ExtractionError as e:
            st.error(str(e))

res = st.session_state.get("result")
if res:
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("#### Answer")
    st.write(res["answer"])
    if res["sources"]:
        st.markdown("**Cited from**")
        for s in res["sources"]:
            st.markdown(f'<div class="cite">{s}</div>', unsafe_allow_html=True)
    st.caption(f"`{res['model']}` · {res['elapsed_ms']} ms")
