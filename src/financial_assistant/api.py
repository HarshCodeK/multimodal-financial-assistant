import hmac
import json
import logging
import os
from collections import deque
from pathlib import Path
from threading import Lock, RLock
from time import monotonic, perf_counter

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from .answer import GroundedAnswerer
from .extract import MAX_PDF_BYTES, extract_pdf_bytes
from .llm import GroqClient
from .models import AskRequest, AskResponse, ImageRequest, ImageResponse, IngestResponse, DocumentChunk, SourceType
from .observability import ProcessMetrics
from .retrieval import ChromaRetriever, SentenceTransformerEmbedder
from .store import InteractionStore

logger = logging.getLogger(__name__)
app = FastAPI(title="Multimodal Financial Assistant", version="0.3.0")
_retriever = None
_answerer = None
_store = None
_vision_client = None
_dependencies_lock = RLock()
_metrics = ProcessMetrics()
_METRICS_EXCLUDED = {"/health", "/ready", "/metrics"}
_provider_hits: deque[float] = deque()
_provider_hits_lock = Lock()


def configure(retriever, answerer, store=None, vision_client=None):
    global _retriever, _answerer, _store, _vision_client
    with _dependencies_lock:
        _retriever, _answerer, _store, _vision_client = retriever, answerer, store, vision_client


def _ensure_dependencies():
    global _retriever, _answerer, _store, _vision_client
    with _dependencies_lock:
        if _retriever is None:
            _retriever = ChromaRetriever(os.getenv("CHROMA_DIR", "./.chroma"), SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")))
            if _retriever.collection.count() == 0:
                sample_path = Path(__file__).resolve().parents[2] / "data" / "sample_policy.txt"
                if sample_path.exists():
                    _retriever.upsert([DocumentChunk(source=sample_path.name, source_type=SourceType.text, text=sample_path.read_text(encoding="utf-8").strip())])
        if _answerer is None:
            _answerer = GroundedAnswerer(GroqClient())
        if _store is None:
            _store = InteractionStore(os.getenv("SQLITE_PATH", "./data/interactions.db"))
        if _vision_client is None:
            _vision_client = GroqClient()


def _provider_request_allowed() -> bool:
    try:
        limit = max(1, int(os.getenv("API_REQUESTS_PER_MINUTE", "60")))
    except ValueError:
        limit = 60
    if os.getenv("GROQ_API_KEY"):
        try:
            llm_limit = max(1, int(os.getenv("LLM_REQUESTS_PER_MINUTE", "30")))
        except ValueError:
            llm_limit = 30
        limit = min(limit, llm_limit)
    now = monotonic()
    with _provider_hits_lock:
        while _provider_hits and now - _provider_hits[0] >= 60:
            _provider_hits.popleft()
        if len(_provider_hits) >= limit:
            return False
        _provider_hits.append(now)
        return True


@app.middleware("http")
async def collect_request_metrics(request, call_next):
    if request.url.path in _METRICS_EXCLUDED:
        return await call_next(request)
    started = _metrics.start()
    status_code = 500
    try:
        if request.url.path in {"/ask", "/describe-image"} and not _provider_request_allowed():
            status_code = 429
            return JSONResponse(status_code=429, content={"detail": "the demo's shared request limit was reached; retry later"})
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        route_name = getattr(route, "path", "<unmatched>")
        # Early middleware responses happen before routing populates scope["route"].
        if route is None and request.url.path in {"/ask", "/describe-image"}:
            route_name = request.url.path
        _metrics.finish(f"{request.method} {route_name}", status_code, started)


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
async def ingest(file: UploadFile = File(...), x_ingest_token: str = Header(default="")):
    configured_token = os.getenv("INGEST_API_TOKEN", "")
    if not configured_token:
        await file.close()
        raise HTTPException(status_code=503, detail="document ingestion is disabled; configure INGEST_API_TOKEN")
    if not x_ingest_token or not hmac.compare_digest(x_ingest_token, configured_token):
        await file.close()
        raise HTTPException(status_code=401, detail="invalid ingestion token")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(status_code=415, detail="upload a PDF file")
    raw = await file.read(MAX_PDF_BYTES + 1)
    if len(raw) > MAX_PDF_BYTES:
        await file.close()
        raise HTTPException(status_code=413, detail="PDF exceeds the 10 MiB limit")
    try:
        chunks = extract_pdf_bytes(raw, source_name=Path(file.filename).name)
    except ValueError as exc:
        await file.close()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await file.close()
    if not chunks:
        raise HTTPException(status_code=422, detail="PDF contains no extractable text; scanned PDFs are not OCR-processed")
    try:
        _ensure_dependencies()
        _retriever.upsert(chunks)
    except Exception as exc:
        logger.exception("document indexing failed")
        raise HTTPException(status_code=503, detail="document indexing is temporarily unavailable") from exc
    pages_extracted = len({chunk.page for chunk in chunks})
    _store.record(
        question=f"PDF extraction: {Path(file.filename).name}",
        supported=True,
        answer=f"Indexed {len(chunks)} text chunks across {pages_extracted} pages.",
        latency_ms=round((perf_counter() - started) * 1000, 2),
        extraction_result={
            "source": Path(file.filename).name,
            "chunks_indexed": len(chunks),
            "pages_extracted": pages_extracted,
            "source_type": "pdf",
        },
    )
    return IngestResponse(source=Path(file.filename).name, chunks_indexed=len(chunks), pages_extracted=pages_extracted)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, response: Response):
    try:
        _ensure_dependencies()
        evidence = _retriever.search(request.question)
        result = _answerer.answer(request.question, evidence)
        _store.record(request.question, result.supported, result.answer, latency_ms=round((perf_counter() - started) * 1000, 2))
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
