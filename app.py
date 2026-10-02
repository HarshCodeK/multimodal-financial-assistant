"""Streamlit UI — upload a financial document, ask about it.

UI decisions worth being able to explain in an interview:

* **One accent colour** (green), used only for the primary action and live
  status. Two accents would mean one of them is decoration.
* **No card-inside-card.** Flat sections separated by whitespace and a single
  hairline rule, not stacked boxes.
* **System font stack, no web font.** A remote font is a render-blocking request
  and the payoff is invisible at this text size.
* **The model picker is a real dropdown**, populated from `src.models`, with
  each model's price and speed shown. Retired models are not hidden — they are
  absent because they 404, and `src/models.py` records why.

Rerun note: Streamlit re-executes this whole script on every interaction, so
document parsing is guarded on filename to avoid re-uploading on each click.
"""
import json
import os
import tempfile

import streamlit as st

from src import models
from src.document_parser import load_document
from src.knowledge_base import index_stats
from src.monitor import get_recent_logs
from src.qa_engine import AnswerError, answer_question
from src.vision_extractor import ExtractionError, extract_fields

st.set_page_config(page_title="Financial Assistant", layout="wide", page_icon="◈")

ACCENT = "#1F7A5C"
INK = "#12181F"
MUTED = "#5A6673"
HAIRLINE = "#DCE3E8"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1060px; }}
      h1, h2, h3, h4 {{ letter-spacing: -0.02em; color: {INK}; }}
      h1 {{ font-size: 1.6rem; margin-bottom: 0.1rem; }}
      .sub {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 1.4rem; }}
      .rule {{ height:1px; background:{HAIRLINE}; border:0; margin:1.3rem 0; }}
      .stat {{ border-left:2px solid {HAIRLINE}; padding-left:0.85rem; margin-top:0.8rem; }}
      .stat .v {{ font-size:0.95rem; font-weight:600; }}
      .stat .l {{ font-size:0.7rem; color:{MUTED}; text-transform:uppercase;
                  letter-spacing:0.06em; margin-top:0.1rem; }}
      .cite {{ font-size:0.82rem; color:{MUTED}; border-left:2px solid {ACCENT};
               padding-left:0.7rem; margin:0.3rem 0; }}
      .stButton>button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Financial document review")
st.markdown(
    '<div class="sub">Upload a PDF or image. A vision model extracts the fields, '
    "a ChromaDB vector store retrieves the governing policy, and every answer is "
    "logged with the model that produced it and the documents it cited.</div>",
    unsafe_allow_html=True,
)

# --- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Model")

    available = models.list_models()
    by_id = {m["id"]: m for m in available}
    ids = list(by_id)
    default_idx = ids.index(models.DEFAULT_MODEL) if models.DEFAULT_MODEL in ids else 0
    chosen = st.selectbox(
        "Model",
        ids,
        index=default_idx,
        format_func=lambda mid: f"{by_id[mid]['label']} — {mid}",
        label_visibility="collapsed",
    )
    meta = by_id[chosen]
    st.caption(
        f"${meta['in_per_1m']:.2f} in / ${meta['out_per_1m']:.2f} out per 1M tokens "
        f"· ~{meta['speed_tps']} tok/s"
    )
    st.caption("Accepts images — required for photo receipts." if meta["vision"]
               else "Text only. PDFs only, not images.")

    if models.key_configured():
        st.markdown(
            f'<div style="font-size:0.78rem;color:{ACCENT};margin-top:0.4rem">'
            f"● key configured</div>",
            unsafe_allow_html=True,
        )
    else:
        st.error("No GROQ_API_KEY — copy .env.example to .env and add your key.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("#### Policy index")
    try:
        stats = index_stats()
        st.markdown(
            f'<div class="stat"><div class="v">{stats["chunks"]} chunks</div>'
            f'<div class="l">{stats["store"]}</div></div>'
            f'<div class="stat"><div class="v" style="font-size:0.8rem">'
            f'{stats["embedding_model"]}</div><div class="l">embeddings</div></div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption(f"Not built yet ({type(e).__name__}). Builds on first question.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("#### Recent")
    try:
        rows = get_recent_logs(6)
        if rows:
            for r in rows:
                try:
                    cited = ", ".join(json.loads(r.get("sources") or "[]")[:1])
                except (json.JSONDecodeError, TypeError):
                    cited = ""
                st.caption(f"**{r['filename']}** — {r['question'][:42]}")
                st.caption(f"`{r.get('model') or '—'}` · {r['latency_ms']:.0f}ms · {cited}")
        else:
            st.caption("Nothing logged yet.")
    except Exception:
        st.caption("Log unavailable.")

# --- main --------------------------------------------------------------------
uploaded = st.file_uploader(
    "Drop a PDF, JPG or PNG", type=["pdf", "jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded is None:
    st.markdown(
        '<div class="sub" style="margin-top:1.8rem">PDF text is parsed locally '
        "with PyMuPDF; images are sent to the vision model as base64. "
        "Nothing is stored beyond the local SQLite log.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if st.session_state.get("doc_name") != uploaded.name:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    try:
        st.session_state["doc"] = load_document(tmp.name)
        st.session_state["doc_name"] = uploaded.name
        st.session_state.pop("fields", None)
        st.session_state.pop("result", None)
    except ValueError as e:
        st.error(str(e))
        st.stop()

doc = st.session_state["doc"]
has_key = models.key_configured()

left, right = st.columns([1, 1], gap="large")

with left:
    if doc["type"] == "image":
        st.image(doc["path"], width=360)
    else:
        st.text_area(
            "Extracted PDF text", doc["text"][:2000], height=220,
            disabled=True, label_visibility="collapsed",
        )

with right:
    if st.button("Extract fields", type="primary", use_container_width=True, disabled=not has_key):
        try:
            with st.spinner(f"Extracting with {chosen}…"):
                st.session_state["fields"] = extract_fields(doc, model=chosen)
        except ExtractionError as e:
            st.error(f"Extraction failed — {e}")

    fields = st.session_state.get("fields")
    if fields:
        if "error" in fields:
            st.error(fields.get("detail") or fields["error"])
            with st.expander("Raw response"):
                st.code(json.dumps(fields, indent=2, default=str), language="json")
        else:
            st.markdown(f"**{fields.get('vendor') or 'Unknown vendor'}**")
            st.markdown(
                f'<div class="sub" style="margin:0.1rem 0 0.6rem">'
                f'{fields.get("date") or "no date"} · total '
                f"<strong>{fields.get('total')}</strong></div>",
                unsafe_allow_html=True,
            )
            with st.expander("All extracted fields"):
                st.json(fields)

    question = st.text_input(
        "Ask about this charge",
        value="Why was this charge deducted?",
        label_visibility="collapsed",
    )
    if st.button("Ask", use_container_width=True, disabled=not has_key):
        try:
            with st.spinner("Retrieving policy context and answering…"):
                st.session_state["result"] = answer_question(doc, question, model=chosen)
        except AnswerError as e:
            st.error(f"Answer failed — {e}")

result = st.session_state.get("result")
if result:
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("#### Answer")
    st.write(result["answer"])
    if result.get("policy_files"):
        st.markdown("**Cited from**")
        for fname in result["policy_files"]:
            st.markdown(f'<div class="cite">{fname}</div>', unsafe_allow_html=True)
    st.caption(f"`{result['model']}` · {result.get('latency_ms', 0):.0f} ms")
