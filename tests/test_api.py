import pytest
import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from financial_assistant.api import app, configure
from financial_assistant.models import RetrievedChunk, SourceType, Answer, DocumentChunk


class FakeRetriever:
    def __init__(self):
        self.upserted = []
    def search(self, question):
        return [RetrievedChunk(source="policy.pdf", source_type=SourceType.pdf, text="Meals are reimbursed up to 500 INR.", page=3, distance=.1)]
    def upsert(self, chunks):
        self.upserted.extend(chunks)


class FakeAnswerer:
    def answer(self, question, evidence):
        return Answer(supported=True, answer="Meals are reimbursed up to 500 INR.", sources=evidence)


class FakeStore:
    def __init__(self): self.rows = []
    def record(self, question, supported, answer, latency_ms=None, extraction_result=None):\n        self.rows.append((question, supported, answer, latency_ms, extraction_result))


class FakeVisionClient:
    def __init__(self): self.calls = []
    def describe_image(self, question, image_data_url):
        self.calls.append((question, image_data_url))
        return "The image shows a 500 INR meal limit."


def configure_fakes(retriever=None):
    retriever = retriever or FakeRetriever()
    store = FakeStore()
    vision = FakeVisionClient()
    configure(retriever, FakeAnswerer(), store, vision)
    return retriever, store, vision


def valid_jpeg_data_url():
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(220, 30, 30)).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_health_metrics_and_ask():
    retriever, store, _ = configure_fakes()
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/ask", json={"question": "What is the meal limit?"})
    assert response.status_code == 200
    assert response.json()["supported"] is True
    assert len(store.rows) == 1
    metrics = client.get("/metrics").json()
    assert metrics["routes"]["POST /ask"]["requests_since_start"] >= 1


def test_describe_image():
    _, _, vision = configure_fakes()
    response = TestClient(app).post("/describe-image", json={"question": "What is shown?", "image_data_url": valid_jpeg_data_url()})
    assert response.status_code == 200
    assert "500 INR" in response.json()["answer"]
    assert len(vision.calls) == 1


def test_image_rejects_non_image_data_url():
    configure_fakes()
    response = TestClient(app).post("/describe-image", json={"question": "What is shown?", "image_data_url": "data:text/plain;base64,AAAA"})
    assert response.status_code == 422


def test_image_rejects_invalid_base64():
    response = TestClient(app).post("/describe-image", json={"question": "What is shown?", "image_data_url": "data:image/jpeg;base64,not!base64"})
    assert response.status_code == 422


def test_ingestion_is_disabled_without_owner_token(monkeypatch):
    monkeypatch.delenv("INGEST_API_TOKEN", raising=False)
    configure_fakes()
    response = TestClient(app).post("/ingest", files={"file": ("policy.pdf", b"%PDF-demo", "application/pdf")})
    assert response.status_code == 503


def test_ingestion_requires_correct_token(monkeypatch):
    monkeypatch.setenv("INGEST_API_TOKEN", "owner-secret")
    configure_fakes()
    response = TestClient(app).post("/ingest", files={"file": ("policy.pdf", b"%PDF-demo", "application/pdf")}, headers={"X-Ingest-Token": "wrong"})
    assert response.status_code == 401


def test_ingestion_indexes_bounded_extracted_chunks(monkeypatch):
    monkeypatch.setenv("INGEST_API_TOKEN", "owner-secret")
    retriever, _, _ = configure_fakes()
    chunk = DocumentChunk(source="policy.pdf", source_type=SourceType.pdf, text="Meals are reimbursed up to 500 INR.", page=1)
    from unittest.mock import patch
    with patch("financial_assistant.api.extract_pdf_bytes", return_value=[chunk]):
        response = TestClient(app).post("/ingest", headers={"X-Ingest-Token": "owner-secret"}, files={"file": ("policy.pdf", b"%PDF-demo", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 1
    assert len(retriever.upserted) == 1



def test_image_rejects_excessive_pixel_count(monkeypatch):
    from pydantic import ValidationError
    from financial_assistant.models import ImageRequest

    class OversizedImage:
        format = "JPEG"
        size = (10_000, 10_000)

        def verify(self):
            return None

    monkeypatch.setattr("financial_assistant.models.Image.open", lambda _: OversizedImage())
    with pytest.raises(ValidationError, match="pixel limit"):
        ImageRequest(question="Read the statement", image_data_url=valid_jpeg_data_url())


def test_unmatched_paths_do_not_create_unbounded_metric_labels():
    configure_fakes()
    client = TestClient(app)
    client.get("/unmatched-a")
    client.get("/unmatched-b")
    routes = client.get("/metrics").json()["routes"]
    assert "GET <unmatched>" in routes
    assert "GET /unmatched-a" not in routes
    assert "GET /unmatched-b" not in routes


def test_provider_limit_counts_early_rejections_on_the_correct_route(monkeypatch):
    import financial_assistant.api as api_module

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "1")
    with api_module._provider_hits_lock:
        api_module._provider_hits.clear()
    configure_fakes()
    client = TestClient(app)

    first = client.post("/ask", json={"question": "What is the meal limit?"})
    second = client.post("/ask", json={"question": "What is the hotel limit?"})

    assert first.status_code == 200
    assert second.status_code == 429
    statuses = client.get("/metrics").json()["routes"]["POST /ask"]["status_codes_since_start"]
    assert statuses.get("429", 0) >= 1
