# Reliability lab

The client-side load probe and the application's /metrics endpoint answer different questions: the probe measures the HTTP experience at the caller; /metrics measures routes observed by one application process.

## Local workflow

Install dependencies and start the API:

~~~bash
pip install -e .
uvicorn financial_assistant.api:app --host 127.0.0.1 --port 8000
~~~

In another terminal, use a staged ramp:

~~~bash
python scripts/load_test.py --base-url http://127.0.0.1:8000 --path /ask --payload '{"question":"What is the meal reimbursement limit?"}' --requests 100 --concurrency 10 --output artifacts/ask-10.json
python scripts/load_test.py --base-url http://127.0.0.1:8000 --path /ask --payload '{"question":"What is the meal reimbursement limit?"}' --requests 250 --concurrency 20 --output artifacts/ask-20.json
curl http://127.0.0.1:8000/metrics
~~~

Test document ingestion separately from answer generation. They stress different paths and should not be combined into one capacity claim.

## Read the report correctly

- Record commit SHA, host/plan, region, request mix, cold/warm state, request count and concurrency. Repeat each level at least three times.
- The report contains p50/p95/p99/max latency, throughput, error rate, status codes and transport failures. It is client-observed, not a server CPU/memory report.
- /metrics is process-local, resets on restart, and stores percentiles for a bounded recent sample window.
- Concurrency means requests simultaneously in flight from this generator, not real users or persistent sessions.
- Do not publish a capacity or accuracy figure until it has been measured on a defined version, dataset and host.

## Failure diagnosis

1. Correlate the first p95/p99 jump or non-2xx response with application logs and host CPU/RAM/restarts.
2. Separate failures into request validation, PDF extraction, embedding/Chroma, LLM provider, SQLite contention, client timeout, or host resource exhaustion.
3. Change one variable and repeat the identical ramp. Keep both the failed and retest reports.
4. The local embedding model can dominate cold-start time and memory on a free instance. Verify startup, first request, and sustained load separately.
5. Chroma and SQLite are stored on local disk by default; on free hosts with ephemeral storage, uploaded indexes and logs can disappear on restart or spin-down. This is not a durable production data store.

## Deployment caution

The Render Blueprint uses a free web instance as a demo starting point, not a production capacity claim. Free Render services sleep after inactivity, have ephemeral storage and cannot scale to multiple instances. The embedding stack may exceed a small free instance's memory; inspect deployment logs and host metrics before presenting a live benchmark.
