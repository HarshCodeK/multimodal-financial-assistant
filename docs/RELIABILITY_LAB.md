# Reliability lab

The client-side load probe and the application's /metrics endpoint answer different questions: the probe measures caller-observed HTTP behavior; /metrics measures traffic seen by one application process.

## Local workflow

~~~bash
pip install -e .
INGEST_API_TOKEN=local-owner-token GROQ_API_KEY=your-key uvicorn financial_assistant.api:app --host 127.0.0.1 --port 8000
~~~

In another terminal, run a staged ramp:

~~~bash
python scripts/load_test.py --base-url http://127.0.0.1:8000 --path /ask --payload '{"question":"What is the meal reimbursement limit?"}' --requests 100 --concurrency 10 --output artifacts/ask-10.json
python scripts/load_test.py --base-url http://127.0.0.1:8000 --path /ask --payload '{"question":"What is the meal reimbursement limit?"}' --requests 250 --concurrency 20 --output artifacts/ask-20.json
curl http://127.0.0.1:8000/metrics
~~~

Test ingestion separately from Q&A. They stress different paths. Ingestion requires the owner-issued X-Ingest-Token matching INGEST_API_TOKEN.

## Read the report correctly

- Record commit SHA, host/plan, region, request mix, cold/warm state, request count and concurrency. Repeat each level at least three times.
- The report contains p50/p95/p99/max latency, throughput, error rate, status codes and transport failures. It is client-observed, not a server CPU/memory report.
- The generator uses a bounded worker pool, caps a run at 100,000 requests and 10,000 configured in-flight requests, and treats only 2xx responses as successful. These are client-side safety bounds, not target-capacity claims.
- /metrics is process-local, resets on restart, and stores percentiles for a bounded recent sample window.
- Concurrency means in-flight requests from this generator, not real users or persistent sessions.
- Do not publish capacity or accuracy figures until measured on a defined version, dataset and host.

## Failure diagnosis

1. Correlate p95/p99 jumps or non-2xx responses with application logs and host CPU/RAM/restarts.
2. Separate request validation, PDF extraction, embedding/Chroma, LLM provider, SQLite contention, client timeout and host resource exhaustion.
3. Change one variable and repeat the identical ramp. Keep both the failed and retest reports.
4. Local embedding initialization can dominate cold-start time and memory. Verify startup, the first question and sustained load separately.
5. Chroma and SQLite use local disk; data can disappear after a free host restart or spin-down.

## Safety and deployment notes

/ask and /describe-image share a process-local ceiling of 60 calls/minute by default (`API_REQUESTS_PER_MINUTE`). When `GROQ_API_KEY` is configured, the effective ceiling is also limited by `LLM_REQUESTS_PER_MINUTE` (default 30/minute). It is a shared ceiling, not per-user or distributed rate limiting. The embedding stack may exceed a small free instance's memory, so inspect host logs before publishing results. Free Render services sleep after inactivity and do not provide production uptime guarantees.
