import json
import logging
import os
from pathlib import Path
from threading import RLock

import httpx
from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from .answer import GroundedAnswerer
from .extract import MAX_PDF_BYTES, extract_pdf_bytes
from .llm import GroqClient
from .models import AskRequest, AskResponse, ImageRequest, ImageResponse, IngestResponse, DocumentChunk, SourceType
from .observability import ProcessMetrics
from .retrieval import ChromaRetriever, SentenceTransformerEmbedder
from .store import InteractionStore

logger = logging.getLogger(__name__)
app = FastAPI(title="Multimodal Financial Assistant", version="0.2.0")

_retriever = None
_answerer = None
_store = None
_vision_client = None
_dependencies_lock = RLock()
_metrics = ProcessMetrics()
_METRICS_EXCLUDED = {"/health", "/ready", "/metrics"}


def configure(retriever, answerer, store=None, vision_client=None):
    global _retriever, _answerer, _store, _vision_client
    with _dependencies_lock:
        _retriever, _answerer, _store, _vision_client = retriever, answerer, store, vision_client


def _ensure_dependencies():
    global _retriever, _answerer, _store, _vision_client
    with _dependencies_lock:
        if _retriever is None:
            _retriever = ChromaRetriever(
                os.getenv("CHROMA_DIR", "./.chroma"),
                SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")),
            )
            if _retriever.collection.count() == 0:
                sample_path = Path(__file__).resolve().parents[2] / "data" / "sample_policy.txt"
                if sample_path.exists():
                    _retriever.upsert([DocumentChunk(
                        source=sample_path.name,
                        source_type=SourceType.text,
                        text=sample_path.read_text(encoding="utf-8").strip(),
                    )])
        if _answerer is None:
            _answerer = GroundedAnswerer(GroqClient())
        if _store is None:
            _store = InteractionStore(os.getenv("SQLITE_PATH", "./data/interactions.db"))
        if _vision_client is None:
            _vision_client = GroqClient()


@app.middleware("http")
async def collect_request_metrics(request, call_next):
    if request.url.path in _METRICS_EXCLUDED:
        return await call_next(request)
    started = _metrics.start()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        _metrics.finish(f"{request.method} {request.url.path}", status_code, started)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response):
    try:
        _ensure_dependencies()
        return {"status": "ready", "llm_configured": bool(os.getenv("GROQ_API_KEY"))}
    except Exception:
        logger.exception("readiness check failed")
        response.status_code = 503
        return {"status": "not_ready"}


@app.get("/metrics")
def metrics():
    return _metrics.snapshot()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="upload a PDF file")
    raw = await file.read(MAX_PDF_BYTES + 1)
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 10 MiB limit")
    try:
        chunks = extract_pdf_bytes(raw, source_name=Path(file.filename).name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not chunks:
        raise HTTPException(status_code=422, detail="PDF contains no extractable text; scanned PDFs are not OCR-processed")
    try:
        _ensure_dependencies()
        _retriever.upsert(chunks)
    except Exception as exc:
        logger.exception("document indexing failed")
        raise HTTPException(status_code=503, detail="document indexing is temporarily unavailable") from exc
    finally:
        await file.close()
    return IngestResponse(source=Path(file.filename).name, chunks_indexed=len(chunks), pages_extracted=len({chunk.page for chunk in chunks}))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, response: Response):
    try:
        _ensure_dependencies()
        evidence = _retriever.search(request.question)
        result = _answerer.answer(request.question, evidence)
        _store.record(request.question, result.supported, result.answer)
        return result
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError, json.JSONDecodeError):
        logger.exception("answer generation failed")
        response.status_code = 503
        return AskResponse(supported=False, answer="The answer service is temporarily unavailable. No answer was generated.", sources=[])


@app.post("/describe-image", response_model=ImageResponse)
def describe_image(request: ImageRequest, response: Response):
    try:
        _ensure_dependencies()
        answer = _vision_client.describe_image(request.question, request.image_data_url)
        return ImageResponse(answer=answer)
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError):
        logger.exception("image description failed")
        response.status_code = 503
        return ImageResponse(answer="Image analysis is temporarily unavailable.")
