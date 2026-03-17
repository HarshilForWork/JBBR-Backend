# final_backend.py
"""
Final consolidated backend — ThreadPoolExecutor edition.
  • Shared ThreadPoolExecutor (worker count from config.yaml executor.max_workers)
  • Entire blocking pipeline runs as ONE sync task per request
  • FastAPI event loop never blocked — only truly async download stays async
  • Groq LLM inference via FAISS retrieval + BGE reranking
  • Optional session_id / user_id tracking (echoed in response + logs)
  • All params configurable via config.yaml
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import re
import uuid
import datetime
import json
import time
import threading
import asyncio

import yaml
from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline import process_all_documents_pipeline, query_documents_sync
from src.embedding.embedder import generate_query_embedding_pinecone
from src.data_ingestion.downloader import download_pdf as _download_pdf

# ── LLMOps / MLOps layer ─────────────────────────────────────────────────────
from src.ops.metrics import (
    ACTIVE_PIPELINES, REQUEST_COUNTER, ERROR_COUNTER,
    EMPTY_RETRIEVAL_COUNTER, LOW_SIMILARITY_COUNTER,
    TOKEN_USAGE_COUNTER, LLM_CONFIDENCE,
    record_stage, metrics_app,
    inc_tokens,
)
from src.ops.experiment_tracker import ExperimentTracker
from src.ops.evaluator import RagasEvaluator
from src.ops.alerts import AlertManager
from src.ops.cost_tracker import CostTracker

# ---------------------------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"⚠️ config.yaml not found or invalid ({exc}), using defaults")
        return {}

_CONFIG = _load_config()
_LLM_CFG  = _CONFIG.get("llm",              {})
_EMB_CFG  = _CONFIG.get("embedding",        {})
_RET_CFG  = _CONFIG.get("retrieval",        {})
_LOG_CFG  = _CONFIG.get("logging",          {})
_SRV_CFG  = _CONFIG.get("server",           {})
_CON_CFG  = _CONFIG.get("concurrency",      {})
_STR_CFG  = _CONFIG.get("storage",          {})
_EXE_CFG  = _CONFIG.get("executor",         {})
_OPS_ET   = _CONFIG.get("experiment_tracking", {})
_OPS_EVAL = _CONFIG.get("evaluation",       {})

# Testing-mode toggle from config.yaml (testing: true/false)
# true  → MLflow only, skip Prometheus/Grafana/AlertManager
# false → Prometheus/Grafana/AlertManager, skip MLflow
_TESTING_MODE: bool = bool(_CONFIG.get("testing", False))
_mode_label = "TESTING" if _TESTING_MODE else "PRODUCTION"
print(f"🔀 Observability mode: {_mode_label} ({'MLflow only' if _TESTING_MODE else 'Prometheus/Grafana only'})")

# Resolved runtime values
_INDEX_NAME      = _RET_CFG.get("index_name",          "policy-index")
_LOG_DIR         = _LOG_CFG.get("log_dir",             "request_logs")
_LOG_PREFIX      = _LOG_CFG.get("log_prefix",          "async_log")
_PDF_STORAGE_DIR = _STR_CFG.get("pdf_storage_dir",     "stored_pdfs")
_EMB_MODEL       = _EMB_CFG.get("model",               "multilingual-e5-large")
_EMB_INPUT_TYPE  = _EMB_CFG.get("input_type",          "query")
_EMB_TRUNCATE    = _EMB_CFG.get("truncate",            "END")
_MAX_WORKERS     = int(_EXE_CFG.get("max_workers",     10))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Shared ThreadPoolExecutor — initialised ONCE at startup
# Each request gets one worker for its entire blocking pipeline duration.
# Multiple concurrent requests are handled by the pool in parallel.
# ---------------------------------------------------------------------------
EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
print(f"🧵 ThreadPoolExecutor ready — max_workers={_MAX_WORKERS}")

# ---------------------------------------------------------------------------
# Per-request FAISS isolation helpers
# ---------------------------------------------------------------------------
_FAISS_STORAGE_DIR = "faiss_storage"
_INDEX_TTL_SECONDS = 86400   # 1 day


def _make_index_name(session_id: Optional[str], user_id: Optional[str]) -> str:
    """
    Build a short, filesystem-safe FAISS index name for this request.
    Falls back to a random hex suffix when either id is absent.
    """
    _safe = lambda s: re.sub(r"[^a-zA-Z0-9_-]", "-", s or "")[:20].strip("-")
    sid = _safe(session_id) or uuid.uuid4().hex[:8]
    uid = _safe(user_id)   or uuid.uuid4().hex[:8]
    return f"req-{sid}-{uid}"


def _cleanup_old_request_indexes():
    """
    Delete `req-*` FAISS index triplets (*.faiss + *_metadata.json + *_id_map.pkl)
    whose .faiss file is older than _INDEX_TTL_SECONDS.
    Safe to call from any thread.
    """
    if not os.path.isdir(_FAISS_STORAGE_DIR):
        return
    now = time.time()
    removed = 0
    for fname in os.listdir(_FAISS_STORAGE_DIR):
        if not (fname.startswith("req-") and fname.endswith(".faiss")):
            continue
        fpath = os.path.join(_FAISS_STORAGE_DIR, fname)
        try:
            age = now - os.path.getmtime(fpath)
            if age > _INDEX_TTL_SECONDS:
                base = fname[:-6]   # strip ".faiss"
                for ext in (".faiss", "_metadata.json", "_id_map.pkl"):
                    p = os.path.join(_FAISS_STORAGE_DIR, base + ext)
                    if os.path.exists(p):
                        os.remove(p)
                removed += 1
        except Exception as exc:
            print(f"⚠️ Cleanup error for {fname}: {exc}")
    if removed:
        print(f"🗑️ TTL cleanup: removed {removed} stale request index(es) (>{_INDEX_TTL_SECONDS}s old)")


def _cleanup_loop():
    """Background daemon thread: initial cleanup + hourly sweeps."""
    _cleanup_old_request_indexes()
    while True:
        time.sleep(3600)
        _cleanup_old_request_indexes()


@asynccontextmanager
async def lifespan(app):
    """Manage startup and shutdown lifecycle."""
    # ── startup ──
    print(f"🚀 HackRx API starting up — executor workers: {_MAX_WORKERS}")
    # Start background TTL cleanup thread (daemon — exits with process)
    t = threading.Thread(target=_cleanup_loop, daemon=True, name="faiss-ttl-cleanup")
    t.start()
    print(f"🧹 FAISS TTL cleanup thread started (TTL={_INDEX_TTL_SECONDS}s, sweep every 3600s)")

    # ── LLMOps singletons (created once per process) ──
    if _OPS_ET.get("enabled", True):
        app.state.tracker = ExperimentTracker(
            experiment_name=_OPS_ET.get("experiment_name", "hackrx-rag")
        )
    else:
        app.state.tracker = None
    app.state.evaluator = RagasEvaluator()
    app.state.alerts    = AlertManager()

    yield
    # ── shutdown ──
    EXECUTOR.shutdown(wait=False)
    print("🛑 ThreadPoolExecutor shut down")


app = FastAPI(
    title="HackRx Insurance API",
    description="High-performance API for querying insurance PDFs",
    version="4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus /metrics endpoint ──────────────────────────────────────────────
app.mount("/metrics", metrics_app)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class QueryPDFRequest(BaseModel):
    documents:  str            # URL to the PDF
    questions:  List[str]      # List of questions to answer
    session_id: Optional[str] = None   # Optional — for request tracing
    user_id:    Optional[str] = None   # Optional — for per-user analytics


# ---------------------------------------------------------------------------
# Sync helpers — safe to run inside executor threads
# ---------------------------------------------------------------------------

def _batch_embed_sync(
    queries: List[str],
    pinecone_key: str,
) -> tuple[list, list]:
    """
    Batch-embed all queries with a single Pinecone inference call (sync).
    Falls back to per-query embedding if batch fails.

    Returns
    -------
    (embeddings: list[list[float]], per_query_times: list[float])
    """
    pc = Pinecone(api_key=pinecone_key)
    try:
        t0   = time.time()
        resp = pc.inference.embed(
            model=_EMB_MODEL,
            inputs=queries,
            parameters={"input_type": _EMB_INPUT_TYPE, "truncate": _EMB_TRUNCATE},
        )
        # Handle polymorphic Pinecone response
        if isinstance(resp, dict) and "data" in resp:
            embeddings = [item["values"] for item in resp["data"]]
        elif isinstance(resp, list):
            embeddings = [item["values"] for item in resp]
        elif hasattr(resp, "data"):
            embeddings = [item["values"] for item in resp.data]
        else:
            raise ValueError(f"Unexpected embedding response type: {type(resp)}")

        elapsed = time.time() - t0
        print(f"✅ Batch-embedded {len(embeddings)} queries in {elapsed:.2f}s")
        return embeddings, [elapsed / len(queries)] * len(queries)

    except Exception as exc:
        print(f"⚠️ Batch embed failed ({exc}), falling back to individual")
        t0    = time.time()
        embs  = []
        times = []
        for q in queries:
            qt0 = time.time()
            embs.append(generate_query_embedding_pinecone(q, pinecone_key))
            times.append(time.time() - qt0)
        print(f"✅ Individual-embedded {len(embs)} queries in {time.time()-t0:.2f}s")
        return embs, times


def _run_pipeline_sync(
    tmpdir: str,
    queries: List[str],
    pinecone_key: str,
    groq_key: str,
    index_name: str,
) -> dict:
    """
    Single synchronous worker that runs the complete pipeline for one request.
    Called via loop.run_in_executor — runs in a ThreadPoolExecutor thread.

    Pipeline (parallelised):
        A (PDF parse + chunk + embed + FAISS index)  ─┐
                                                       ├─ parallel ─→ join → C (query)
        B (embed user queries via Pinecone)           ─┘

    A and B are independent — no reason to wait for the PDF index before
    embedding the queries.  C needs both results so it waits for the join.

    Each request uses its own FAISS index (index_name) for full isolation
    when multiple requests run concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor as _SubPool

    total_start = time.time()
    ACTIVE_PIPELINES.inc()
    try:
        def _run_pdf():
            t0 = time.time()
            result = asyncio.run(
                process_all_documents_pipeline(
                    docs_dir=tmpdir,
                    pinecone_api_key=pinecone_key,
                    force_reprocess=True,
                    index_name=index_name,
                )
            )
            elapsed = time.time() - t0
            print(f"📄 PDF pipeline done in {elapsed:.2f}s  [index={index_name}]")
            return result, elapsed

        # ── Worker B: embed all queries (zero dependency on FAISS) ───────────
        def _run_embed():
            t0 = time.time()
            embs, times = _batch_embed_sync(queries, pinecone_key)
            elapsed = time.time() - t0
            return embs, times, elapsed

        # ── Run A and B in parallel, then join before C ───────────────────────
        print("⚡ A (PDF ingest) + B (query embed) running in parallel...")
        parallel_start = time.time()
        with _SubPool(max_workers=2) as sub_pool:
            fut_pdf   = sub_pool.submit(_run_pdf)
            fut_embed = sub_pool.submit(_run_embed)
            pdf_result, pdf_time             = fut_pdf.result()
            embeddings, emb_times, emb_time  = fut_embed.result()
        print(f"✅ A+B done in {time.time()-parallel_start:.2f}s  "
              f"(PDF={pdf_time:.2f}s | embed={emb_time:.2f}s, "
              f"saved ~{emb_time:.2f}s vs sequential)")

        # Guard against partial embedding failures
        if len(embeddings) < len(queries):
            pad = len(queries) - len(embeddings)
            print(f"⚠️ {pad} embeddings missing — padding with zero vectors")
            embeddings.extend([[0.0] * 1024] * pad)
            emb_times.extend([0.0] * pad)

        # ── C: Process each query sequentially ───────────────────────────────
        q_start      = time.time()
        query_results = []
        query_times   = []
        for i, (q, emb) in enumerate(zip(queries, embeddings)):
            qt0 = time.time()
            try:
                r = query_documents_sync(
                    query=q,
                    pinecone_api_key=pinecone_key,
                    gemini_api_key=groq_key,
                    index_name=index_name,
                    query_embedding=emb,
                )
            except Exception as exc:
                print(f"  ❌ Query {i+1} failed: {exc}")
                r = {
                    "evaluation": {"answer": f"Query processing error: {exc}", "confidence": 0.0},
                    "search_results": [],
                    "success": False,
                }
            elapsed = time.time() - qt0
            query_times.append(elapsed)
            query_results.append(r)
            print(f"  ✅ Query {i+1}/{len(queries)} done in {elapsed:.2f}s")
        q_time = time.time() - q_start

        return {
            "pdf_result":    pdf_result,
            "embeddings":    embeddings,
            "emb_times":     emb_times,
            "query_results": query_results,
            "query_times":   query_times,
            "timings": {
                "pdf":       pdf_time,
                "embedding": emb_time,
                "queries":   q_time,
                "pipeline":  time.time() - total_start,
            },
        }
    finally:
        ACTIVE_PIPELINES.dec()


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/hackrx/run")
async def query_pdf(input: QueryPDFRequest):
    total_start = time.time()

    pdf_url    = input.documents
    queries    = input.questions
    session_id = input.session_id
    user_id    = input.user_id

    if not pdf_url or not queries or not isinstance(queries, list):
        return JSONResponse(
            {"error": "documents URL and questions (list) are required",
             "answers": ["Error: Invalid input"] * (len(queries) if queries else 0),
             "success": False},
            status_code=400,
        )

    print(f"📥 Request — session={session_id} user={user_id} queries={len(queries)}")

    # Per-request FAISS index name for concurrent isolation
    index_name = _make_index_name(session_id, user_id)
    print(f"🗂️  Using FAISS index: {index_name}")

    # --- Directories ---
    tmpdir = tempfile.mkdtemp()
    os.makedirs(_PDF_STORAGE_DIR, exist_ok=True)

    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"input_{timestamp}.pdf"
    pdf_path     = os.path.join(_PDF_STORAGE_DIR, pdf_filename)

    # ── Step 1: Async download (truly async — stays in event loop) ────────
    try:
        t0 = time.time()
        await _download_pdf(pdf_url, pdf_path, tmpdir, pdf_filename)
        download_time = time.time() - t0
        print(f"📄 Downloaded PDF in {download_time:.2f}s → {pdf_path}")
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to download PDF: {exc}",
             "answers": ["Error: PDF download failed"] * len(queries),
             "success": False},
            status_code=400,
        )

    # ── Step 2: Keys ──────────────────────────────────────────────────────
    pinecone_key = os.getenv("PINECONE_API_KEY")
    groq_key     = os.getenv("GROQ_API_KEY", "")

    if not pinecone_key:
        return JSONResponse(
            {"error": "PINECONE_API_KEY not set", "success": False}, status_code=500
        )
    if not groq_key:
        print("⚠️ GROQ_API_KEY not set — LLM answers will be unavailable")

    # ── Step 3: Offload ENTIRE blocking pipeline to shared executor ────────
    loop = asyncio.get_event_loop()
    try:
        pipeline_result = await loop.run_in_executor(
            EXECUTOR,
            _run_pipeline_sync,
            tmpdir,
            queries,
            pinecone_key,
            groq_key,
            index_name,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Pipeline failed: {exc}",
             "answers": ["Error: Pipeline error"] * len(queries),
             "session_id": session_id,
             "user_id": user_id,
             "success": False},
            status_code=500,
        )

    pdf_result   = pipeline_result["pdf_result"]
    all_embs     = pipeline_result["embeddings"]
    emb_times    = pipeline_result["emb_times"]
    query_results = pipeline_result["query_results"]
    query_times  = pipeline_result["query_times"]
    timings      = pipeline_result["timings"]

    if not pdf_result.get("success"):
        return JSONResponse(
            {"error": f"PDF processing failed: {pdf_result.get('error', 'Unknown')}",
             "answers": ["Error: PDF processing failed"] * len(queries),
             "session_id": session_id,
             "user_id": user_id,
             "success": False},
            status_code=500,
        )

    # ── Step 4: Build response (fast dict ops — safe in event loop) ────────
    answers            = []
    confidences        = []
    similarity_vectors = []
    llm_contexts       = []
    source_vectors_info = []

    for result in query_results:
        evaluation = result.get("evaluation", {})
        answers.append(evaluation.get("answer", "No answer found"))
        confidences.append(evaluation.get("confidence", 0.0))
        llm_contexts.append(evaluation.get("llm_context", ""))
        source_vectors_info.append(evaluation.get("source_vectors", []))

        sim_data = []
        for sr in result.get("search_results", []):
            sim_data.append({
                "id":               sr.get("id", ""),
                "similarity_score": sr.get("score", sr.get("similarity_score", 0.0)),
                "hybrid_score":     sr.get("hybrid_score", sr.get("score", 0.0)),
                "text":             sr.get("text", ""),
                "document_name":    sr.get("document_name", ""),
                "page_number":      sr.get("page_number", 1),
            })
        similarity_vectors.append(sim_data)

    total_time = time.time() - total_start
    max_qt     = max(query_times) if query_times else 0
    min_qt     = min(query_times) if query_times else 0

    performance_stats = {
        "pdf_download":         download_time,
        "pdf_processing":       timings["pdf"],
        "embedding_time":       timings["embedding"],
        "query_processing_time": timings["queries"],
        "pipeline_total":       timings["pipeline"],
        "total_time":           total_time,
    }

    executor_stats = {
        "model":         "ThreadPoolExecutor",
        "max_workers":   _MAX_WORKERS,
        "num_queries":   len(queries),
        "individual_embedding_times": emb_times,
        "individual_query_times":    query_times,
        "slowest_query": max_qt,
        "fastest_query": min_qt,
    }

    response_data = {
        # Session / user tracking
        "session_id":  session_id,
        "user_id":     user_id,
        # Core results
        "answers":           answers,
        "confidences":       confidences,
        "similarity_vectors": similarity_vectors,
        "llm_contexts":      llm_contexts,
        "source_vectors":    source_vectors_info,
        # Timing
        "total_time_taken":  total_time,
        "timing_breakdown":  performance_stats,
        "executor_stats":    executor_stats,
        # Meta
        "vector_info": {
            "total_queries":            len(queries),
            "query_embedding_dimension": len(all_embs[0]) if all_embs else 0,
            "embedding_model":          _EMB_MODEL,
            "llm_model":                _LLM_CFG.get("gemini_model", "gemini-2.5-flash")
                                        if _LLM_CFG.get("provider", "groq") == "gemini"
                                        else _LLM_CFG.get("model", "llama-3.3-70b-versatile"),
            "llm_provider":             _LLM_CFG.get("provider", "groq"),
            "total_similarity_results": sum(len(sv) for sv in similarity_vectors),
            "context_info": {
                "total_contexts_provided": len(llm_contexts),
                "average_context_length": (
                    sum(len(c) for c in llm_contexts) / len(llm_contexts)
                    if llm_contexts else 0
                ),
                "total_source_vectors": sum(len(sv) for sv in source_vectors_info),
            },
        },
    }
    # ── LLMOps instrumentation ────────────────────────────────────────────
    _tracker  = getattr(app.state, "tracker",  None)
    _alerts   = getattr(app.state, "alerts",   None)
    _evaluator = getattr(app.state, "evaluator", None)

    model_label = (
        _LLM_CFG.get("gemini_model", "gemini-2.0-flash")
        if _LLM_CFG.get("provider", "groq") == "gemini"
        else _LLM_CFG.get("model", "llama-3.3-70b-versatile")
    )

    # ── 3. RAGAS evaluation ──────────────────────────────────────────────
    all_evals = []
    if _evaluator and _OPS_EVAL.get("enabled", True):
        for i, (q, qr) in enumerate(zip(queries, query_results)):
            ev    = qr.get("evaluation", {})
            svecs = qr.get("search_results", [])
            emb   = all_embs[i] if i < len(all_embs) else None
            
            if _TESTING_MODE:
                # Sync eval for MLflow in testing mode
                metrics = _evaluator.evaluate(
                    query=q, query_embedding=emb, answer=ev.get("answer", ""),
                    context=ev.get("llm_context", ""), source_vectors=svecs,
                    confidence=float(ev.get("confidence", 0.0)),
                    session_id=session_id or "test", pdf_filename=pdf_filename
                )
                all_evals.append(metrics)
            else:
                # Async eval for production
                def _on_eval_done(m: dict):
                    if _alerts: _alerts.check_eval_metrics(m, q, pdf_filename, session_id or "")

                _evaluator.evaluate_async(
                    query=q, query_embedding=emb, answer=ev.get("answer", ""),
                    context=ev.get("llm_context", ""), source_vectors=svecs,
                    confidence=float(ev.get("confidence", 0.0)),
                    session_id=session_id or "", pdf_filename=pdf_filename,
                    on_complete=_on_eval_done,
                )

    # ── 4. Telemetry logging ─────────────────────────────────────────────
    if _tracker and _TESTING_MODE and query_results:
        # Aggregate eval metrics if available
        avg_eval = {}
        if all_evals:
            for k in all_evals[0].keys():
                avg_eval[k] = sum(e[k] for e in all_evals) / len(all_evals)

        ct = CostTracker(model=model_label)
        ct.prompt_tokens     = sum(int(qr.get("evaluation", {}).get("prompt_tokens", 0)) for qr in query_results)
        ct.completion_tokens = sum(int(qr.get("evaluation", {}).get("completion_tokens", 0)) for qr in query_results)

        print(f"📊 [Telemetry] queries={len(queries)} query_results={len(query_results)} all_evals={len(all_evals)}")
        print(f"📊 [Telemetry] eval_metrics keys: {list(avg_eval.keys())}")

        _tracker.log_run(
            session_id=session_id or "unknown",
            user_id=user_id    or "unknown",
            endpoint="/hackrx/run",
            pipeline_params=_CONFIG,  # Now sends full config (flattened by tracker)
            stage_timings={"download": download_time, **timings},
            token_usage=ct.summary(),
            eval_metrics={
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                **avg_eval
            }
        )

    if not _TESTING_MODE:
        # Production Prometheus counters
        REQUEST_COUNTER.labels(endpoint="run", status="success").inc()
        for i, qr in enumerate(query_results):
            ev = qr.get("evaluation", {})
            LLM_CONFIDENCE.labels(model=model_label).observe(float(ev.get("confidence", 0.0)))
            inc_tokens(model_label, int(ev.get("prompt_tokens", 0)), int(ev.get("completion_tokens", 0)))
            if not qr.get("search_results"): EMPTY_RETRIEVAL_COUNTER.inc()

        for stage_key, prom_label in [("pdf", "pdf"), ("embedding", "embed"), ("queries", "llm")]:
            dur = timings.get(stage_key, 0.0)
            if dur > 0: PIPELINE_DURATION.labels(stage=prom_label).observe(dur)
        
        if _alerts:
            _alerts.check_latency(total_time, performance_stats, session_id or "")
            for i, (q, qr) in enumerate(zip(queries, query_results)):
                svecs = qr.get("search_results", [])
                _alerts.check_retrieval(svecs, q, pdf_filename, session_id or "")
                ev = qr.get("evaluation", {})
                _alerts.check_confidence(float(ev.get("confidence", 0.0)), q, pdf_filename, ev.get("answer", ""), session_id or "")

    if _LOG_CFG.get("enabled", True):
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            log_entry = {
                "timestamp":  datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "user_id":    user_id,
                "request": {
                    "pdf_url":         pdf_url,
                    "questions":       queries,
                    "pdf_filename":    pdf_filename,
                    "pdf_stored_path": pdf_path,
                },
                "response": response_data,
                "processing_summary": {
                    "success":                True,
                    "total_questions":         len(queries),
                    "total_time_seconds":      total_time,
                    "pdf_processing_time":     timings["pdf"],
                    "embedding_time":          timings["embedding"],
                    "query_processing_time":   timings["queries"],
                    "max_individual_query_time": max_qt,
                    "query_timing_details": [
                        {"query_index": i,
                         "query": q[:50] + "..." if len(q) > 50 else q,
                         "time_seconds": qt}
                        for i, (q, qt) in enumerate(zip(queries, query_times))
                    ],
                },
                "executor_stats": executor_stats,
            }
            log_path = os.path.join(_LOG_DIR, f"{_LOG_PREFIX}_{timestamp}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            print(f"📝 Logged to: {log_path}")
        except Exception as exc:
            print(f"⚠️ Failed to save request log: {exc}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"🏁 PERFORMANCE SUMMARY  session={session_id}  user={user_id}")
    print(f"   📥 PDF Download     : {download_time:.2f}s")
    print(f"   📄 PDF Processing   : {timings['pdf']:.2f}s")
    print(f"   🧠 Embedding        : {timings['embedding']:.2f}s")
    print(f"   🔍 Query Processing : {timings['queries']:.2f}s ({len(queries)} queries sequential)")
    print(f"   🧵 Executor workers : {_MAX_WORKERS} (max)")
    print(f"   ⏱️  Total           : {total_time:.2f}s")
    print(f"{'='*60}\n")

    return JSONResponse(content=response_data)


# ---------------------------------------------------------------------------
# Upload endpoint  —  multipart/form-data with direct PDF file
# ---------------------------------------------------------------------------

@app.post("/hackrx/run/upload")
async def query_pdf_upload(
    file:       UploadFile = File(..., description="PDF file to process"),
    questions:  List[str] = Form(
        ...,
        description=(
            'Questions to answer. You can send this field multiple times (one per question), '
            'OR send it once as a JSON array string like ["Q1","Q2"]. '
            'Repeated form fields: -F questions=Q1 -F questions=Q2'
        ),
    ),
    session_id: Optional[str] = Form(None, description="Optional session identifier"),
    user_id:    Optional[str] = Form(None, description="Optional user identifier"),
):
    """
    Same pipeline as /hackrx/run but accepts a direct PDF upload instead of a URL.
    Send as multipart/form-data — two ways to pass multiple questions:

    Option A — repeated form fields (recommended for curl / code):
      -F questions="What is the waiting period?" -F questions="What is covered under Plan A?"

    Option B — single JSON array string (Swagger UI / Postman):
      questions = '["What is the waiting period?","What is covered under Plan A?"]'

    Other fields:
      - file      : PDF binary
      - session_id: (optional)
      - user_id   : (optional)
    """
    total_start = time.time()

    # Debug: show exactly what was received
    print(f"📋 RAW questions field: {repr(questions)}")

    # Normalize: FastAPI gives us List[str] — each element may itself be a JSON array or plain text
    queries: List[str] = []
    for item in questions:
        item = (item or "").strip()
        if not item:
            continue
        # If the item looks like a JSON array, expand it
        if item.startswith("["):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, list):
                    queries.extend(str(q).strip() for q in parsed if str(q).strip())
                    continue
            except json.JSONDecodeError:
                pass
        # Newline-separated block
        if "\n" in item:
            queries.extend(ln.strip() for ln in item.splitlines() if ln.strip())
        else:
            queries.append(item)

    if not queries:
        return JSONResponse(
            {
                "error": (
                    "questions field is missing or empty. "
                    'Send repeated fields (-F questions=Q1 -F questions=Q2) '
                    'or a JSON array: ["Q1","Q2"]'
                ),
                "answers": [],
                "success": False,
            },
            status_code=400,
        )

    # Validate file type
    if file.content_type not in ("application/pdf", "application/octet-stream") and \
       not (file.filename or "").lower().endswith(".pdf"):
        return JSONResponse(
            {"error": "Uploaded file must be a PDF", "answers": [], "success": False},
            status_code=400,
        )

    print(f"📥 Upload request — file={file.filename} session={session_id} user={user_id} queries={len(queries)}")

    # Per-request FAISS index name for concurrent isolation
    index_name = _make_index_name(session_id, user_id)
    print(f"🗂️  Using FAISS index: {index_name}")

    # Save uploaded file to temp directory
    tmpdir = tempfile.mkdtemp()
    os.makedirs(_PDF_STORAGE_DIR, exist_ok=True)

    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = file.filename or f"upload_{timestamp}.pdf"
    pdf_path     = os.path.join(_PDF_STORAGE_DIR, pdf_filename)
    tmpdir_path  = os.path.join(tmpdir, pdf_filename)

    try:
        content = await file.read()
        if not content:
            return JSONResponse(
                {"error": "Uploaded file is empty", "answers": [], "success": False},
                status_code=400,
            )
        # Write to both permanent storage and tmpdir (pipeline reads from tmpdir)
        for dest in (pdf_path, tmpdir_path):
            with open(dest, "wb") as f_out:
                f_out.write(content)
        save_time = time.time() - total_start
        print(f"💾 Saved uploaded PDF ({len(content)/1024:.1f} KB) in {save_time:.2f}s → {pdf_path}")
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to save uploaded file: {exc}",
             "answers": ["Error: File save failed"] * len(queries),
             "success": False},
            status_code=500,
        )

    download_time = time.time() - total_start  # file save acts as "download" step

    # Keys
    pinecone_key = os.getenv("PINECONE_API_KEY")
    groq_key     = os.getenv("GROQ_API_KEY", "")

    if not pinecone_key:
        return JSONResponse(
            {"error": "PINECONE_API_KEY not set", "success": False}, status_code=500
        )
    if not groq_key:
        print("⚠️ GROQ_API_KEY not set — LLM answers will be unavailable")

    # Offload full pipeline to shared executor
    loop = asyncio.get_event_loop()
    try:
        pipeline_result = await loop.run_in_executor(
            EXECUTOR, _run_pipeline_sync, tmpdir, queries, pinecone_key, groq_key, index_name
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Pipeline failed: {exc}",
             "answers": ["Error: Pipeline error"] * len(queries),
             "session_id": session_id, "user_id": user_id, "success": False},
            status_code=500,
        )

    pdf_result    = pipeline_result["pdf_result"]
    all_embs      = pipeline_result["embeddings"]
    emb_times     = pipeline_result["emb_times"]
    query_results = pipeline_result["query_results"]
    query_times   = pipeline_result["query_times"]
    timings       = pipeline_result["timings"]

    if not pdf_result.get("success"):
        return JSONResponse(
            {"error": f"PDF processing failed: {pdf_result.get('error', 'Unknown')}",
             "answers": ["Error: PDF processing failed"] * len(queries),
             "session_id": session_id, "user_id": user_id, "success": False},
            status_code=500,
        )

    # Build response (same helper logic as URL endpoint)
    answers             = []
    confidences         = []
    similarity_vectors  = []
    llm_contexts        = []
    source_vectors_info = []

    for result in query_results:
        evaluation = result.get("evaluation", {})
        answers.append(evaluation.get("answer", "No answer found"))
        confidences.append(evaluation.get("confidence", 0.0))
        llm_contexts.append(evaluation.get("llm_context", ""))
        source_vectors_info.append(evaluation.get("source_vectors", []))
        sim_data = []
        for sr in result.get("search_results", []):
            sim_data.append({
                "id":               sr.get("id", ""),
                "similarity_score": sr.get("score", sr.get("similarity_score", 0.0)),
                "hybrid_score":     sr.get("hybrid_score", sr.get("score", 0.0)),
                "text":             sr.get("text", ""),
                "document_name":    sr.get("document_name", ""),
                "page_number":      sr.get("page_number", 1),
            })
        similarity_vectors.append(sim_data)

    total_time = time.time() - total_start
    max_qt     = max(query_times) if query_times else 0
    min_qt     = min(query_times) if query_times else 0

    performance_stats = {
        "file_save_time":        download_time,
        "pdf_processing":        timings["pdf"],
        "embedding_time":        timings["embedding"],
        "query_processing_time": timings["queries"],
        "pipeline_total":        timings["pipeline"],
        "total_time":            total_time,
    }

    executor_stats = {
        "model":       "ThreadPoolExecutor",
        "max_workers": _MAX_WORKERS,
        "num_queries": len(queries),
        "individual_embedding_times": emb_times,
        "individual_query_times":     query_times,
        "slowest_query": max_qt,
        "fastest_query": min_qt,
    }

    response_data = {
        "session_id":  session_id,
        "user_id":     user_id,
        "source":      "upload",
        "filename":    pdf_filename,
        "answers":            answers,
        "confidences":        confidences,
        "similarity_vectors": similarity_vectors,
        "llm_contexts":       llm_contexts,
        "source_vectors":     source_vectors_info,
        "total_time_taken":   total_time,
        "timing_breakdown":   performance_stats,
        "executor_stats":     executor_stats,
        "vector_info": {
            "total_queries":             len(queries),
            "query_embedding_dimension": len(all_embs[0]) if all_embs else 0,
            "embedding_model":           _EMB_MODEL,
            "llm_model":                 _LLM_CFG.get("gemini_model", "gemini-2.5-flash")
                                         if _LLM_CFG.get("provider", "groq") == "gemini"
                                         else _LLM_CFG.get("model", "llama-3.3-70b-versatile"),
            "llm_provider":              _LLM_CFG.get("provider", "groq"),
            "total_similarity_results":  sum(len(sv) for sv in similarity_vectors),
        },
    }

    # Logging
    if _LOG_CFG.get("enabled", True):
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            log_entry = {
                "timestamp":  datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "user_id":    user_id,
                "source":     "upload",
                "request": {
                    "filename":    pdf_filename,
                    "file_size_kb": len(content) / 1024,
                    "questions":   queries,
                    "pdf_stored_path": pdf_path,
                },
                "response": response_data,
            }
            log_path = os.path.join(_LOG_DIR, f"{_LOG_PREFIX}_upload_{timestamp}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            print(f"📝 Logged to: {log_path}")
        except Exception as exc:
            print(f"⚠️ Failed to save request log: {exc}")

    print(f"\n{'='*60}")
    print(f"🏁 UPLOAD SUMMARY  session={session_id}  user={user_id}")
    print(f"   💾 File Save        : {download_time:.2f}s")
    print(f"   📄 PDF Processing   : {timings['pdf']:.2f}s")
    print(f"   🧠 Embedding        : {timings['embedding']:.2f}s")
    print(f"   🔍 Query Processing : {timings['queries']:.2f}s")
    print(f"   ⏱️  Total           : {total_time:.2f}s")
    print(f"{'='*60}\n")

    # ── LLMOps instrumentation ────────────────────────────────────────────
    _tracker   = getattr(app.state, "tracker", None)
    _alerts    = getattr(app.state, "alerts", None)
    _evaluator = getattr(app.state, "evaluator", None)

    model_label = (
        _LLM_CFG.get("gemini_model", "gemini-2.0-flash")
        if _LLM_CFG.get("provider", "groq") == "gemini"
        else _LLM_CFG.get("model", "llama-3.3-70b-versatile")
    )

    # ── 3. RAGAS evaluation ──────────────────────────────────────────────
    all_evals = []
    if _evaluator and _OPS_EVAL.get("enabled", True):
        for i, (q, qr) in enumerate(zip(queries, query_results)):
            ev    = qr.get("evaluation", {})
            svecs = qr.get("search_results", [])
            emb   = all_embs[i] if i < len(all_embs) else None
            
            if _TESTING_MODE:
                # Sync eval for MLflow in testing mode
                metrics = _evaluator.evaluate(
                    query=q, query_embedding=emb, answer=ev.get("answer", ""),
                    context=ev.get("llm_context", ""), source_vectors=svecs,
                    confidence=float(ev.get("confidence", 0.0)),
                    session_id=session_id or "test", pdf_filename=pdf_filename
                )
                all_evals.append(metrics)
            else:
                # Async eval for production
                def _on_eval_done(m: dict):
                    if _alerts: _alerts.check_eval_metrics(m, q, pdf_filename, session_id or "")

                _evaluator.evaluate_async(
                    query=q, query_embedding=emb, answer=ev.get("answer", ""),
                    context=ev.get("llm_context", ""), source_vectors=svecs,
                    confidence=float(ev.get("confidence", 0.0)),
                    session_id=session_id or "", pdf_filename=pdf_filename,
                    on_complete=_on_eval_done,
                )

    # ── 4. Telemetry logging ─────────────────────────────────────────────
    if _tracker and _TESTING_MODE and query_results:
        avg_eval = {}
        if all_evals:
            for k in all_evals[0].keys():
                avg_eval[k] = sum(e[k] for e in all_evals) / len(all_evals)

        ct = CostTracker(model=model_label)
        ct.prompt_tokens     = sum(int(qr.get("evaluation", {}).get("prompt_tokens", 0)) for qr in query_results)
        ct.completion_tokens = sum(int(qr.get("evaluation", {}).get("completion_tokens", 0)) for qr in query_results)

        print(f"📊 [Telemetry] queries={len(queries)} query_results={len(query_results)} all_evals={len(all_evals)}")
        print(f"📊 [Telemetry] eval_metrics keys: {list(avg_eval.keys())}")

        _tracker.log_run(
            session_id=session_id or "unknown",
            user_id=user_id    or "unknown",
            endpoint="/hackrx/run/upload",
            pipeline_params=_CONFIG,
            stage_timings={"download": download_time, **timings},
            token_usage=ct.summary(),
            eval_metrics={
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                **avg_eval
            }
        )

    if not _TESTING_MODE:
        # Production Prometheus counters
        REQUEST_COUNTER.labels(endpoint="upload", status="success").inc()
        for i, qr in enumerate(query_results):
            ev = qr.get("evaluation", {})
            LLM_CONFIDENCE.labels(model=model_label).observe(float(ev.get("confidence", 0.0)))
            inc_tokens(model_label, int(ev.get("prompt_tokens", 0)), int(ev.get("completion_tokens", 0)))
            if not qr.get("search_results"): EMPTY_RETRIEVAL_COUNTER.inc()

        for stage_key, prom_label in [("pdf", "pdf"), ("embedding", "embed"), ("queries", "llm")]:
            dur = timings.get(stage_key, 0.0)
            if dur > 0: PIPELINE_DURATION.labels(stage=prom_label).observe(dur)
        
        if _alerts:
            _alerts.check_latency(total_time, performance_stats, session_id or "")
            for i, (q, qr) in enumerate(zip(queries, query_results)):
                svecs = qr.get("search_results", [])
                _alerts.check_retrieval(svecs, q, pdf_filename, session_id or "")
                ev = qr.get("evaluation", {})
                _alerts.check_confidence(float(ev.get("confidence", 0.0)), q, pdf_filename, ev.get("answer", ""), session_id or "")

    return JSONResponse(content=response_data)


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check — also reports executor state."""
    return {
        "status":      "healthy",
        "timestamp":   datetime.datetime.now().isoformat(),
        "executor": {
            "max_workers": _MAX_WORKERS,
            "active":      True,
        },
    }


@app.get("/stats")
async def get_stats():
    """Get FAISS index statistics."""
    try:
        from src.indexing.store import FAISSVectorStore
        vector_store = FAISSVectorStore("policy-index")
        stats = vector_store.get_stats()
        return JSONResponse({
            "success":    True,
            "faiss_stats": stats,
            "timestamp":  datetime.datetime.now().isoformat(),
        })
    except Exception as exc:
        return JSONResponse(
            {"success": False, "error": str(exc),
             "timestamp": datetime.datetime.now().isoformat()},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host   = _SRV_CFG.get("host",   "0.0.0.0")
    port   = int(_SRV_CFG.get("port",  8000))
    reload = bool(_SRV_CFG.get("reload", True))
    uvicorn.run("final_backend:app", host=host, port=port, reload=reload)

# To run: uvicorn final_backend:app --host 0.0.0.0 --port 8000 --reload
