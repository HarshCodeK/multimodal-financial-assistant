# Multimodal Financial Assistant
#
# Two stages so the runtime image does not carry the build toolchain.
# Note: this image is ~1.5GB because sentence-transformers pulls in torch.
# A text-only deployment could swap the embedding model and drop most of that.

# --- builder: install into a virtualenv we can copy wholesale ----------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed for any wheel without a prebuilt binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Install into /opt/venv so the runtime stage copies one directory rather than
# resolving every dependency path again.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# --- runtime: copy the venv, drop the toolchain ------------------------------
FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MFA_CHROMA=/data/chroma_db \
    MFA_DB=/data/logs.db

# PyMuPDF and torch both want these at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app.py ./
COPY src/ ./src/
COPY data/ ./data/

# The vector store and the log are written at runtime, so they need a writable
# location that is not the source tree.
RUN mkdir -p /data && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
