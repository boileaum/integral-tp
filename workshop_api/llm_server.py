from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .llm import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    LLMClient,
    _is_transient_llm_error,
)
from .retrieval import DEFAULT_OPENROUTER_EMBEDDING_MODEL


class ChatRequest(BaseModel):
    system: str
    user: str
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = Field(
        default=DEFAULT_LLM_MAX_TOKENS,
        ge=1,
        le=DEFAULT_LLM_MAX_TOKENS,
    )
    reasoning_effort: str | None = None
    prompt_cache_key: str | None = None


class ChatResponse(BaseModel):
    text: str
    usage: dict[str, Any]
    raw_usage: Any | None = None
    job_id: str | None = None
    queue: dict[str, Any] | None = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = "float"
    input_type: str | None = "search_query"


class EnqueueResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int | None = None
    attempts: int
    retry_count: int
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    next_retry_at: float | None = None
    last_error: str | None = None
    text: str | None = None
    usage: dict[str, Any] | None = None
    raw_usage: Any | None = None
    queue: dict[str, Any]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_WORKERS = max(1, _env_int("WORKSHOP_LLM_SERVER_WORKERS", 16))
MAX_CONCURRENCY = max(1, _env_int("WORKSHOP_LLM_SERVER_CONCURRENCY", 4))
MIN_INTERVAL_SECONDS = _env_float("WORKSHOP_LLM_SERVER_MIN_INTERVAL_SECONDS", 0.25)
MAX_QUEUE_SIZE = max(1, _env_int("WORKSHOP_LLM_SERVER_QUEUE_SIZE", 500))
MAX_JOB_RETRIES = max(
    0,
    _env_int("WORKSHOP_LLM_SERVER_MAX_RETRIES", _env_int("MISTRAL_MAX_RETRIES", 5)),
)
BACKOFF_INITIAL_SECONDS = _env_float("WORKSHOP_LLM_SERVER_BACKOFF_INITIAL_SECONDS", 1.0)
RATE_LIMIT_BACKOFF_INITIAL_SECONDS = _env_float(
    "WORKSHOP_LLM_SERVER_RATE_LIMIT_BACKOFF_INITIAL_SECONDS",
    3.0,
)
BACKOFF_MAX_SECONDS = _env_float("WORKSHOP_LLM_SERVER_BACKOFF_MAX_SECONDS", 45.0)
JOB_TTL_SECONDS = _env_float("WORKSHOP_LLM_SERVER_JOB_TTL_SECONDS", 3600.0)
EMBEDDING_MAX_CONCURRENCY = max(
    1,
    _env_int("WORKSHOP_EMBEDDING_SERVER_CONCURRENCY", MAX_CONCURRENCY),
)
EMBEDDING_MIN_INTERVAL_SECONDS = _env_float(
    "WORKSHOP_EMBEDDING_SERVER_MIN_INTERVAL_SECONDS",
    0.0,
)
EMBEDDING_MAX_RETRIES = max(
    0,
    _env_int("WORKSHOP_EMBEDDING_SERVER_MAX_RETRIES", MAX_JOB_RETRIES),
)

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="llm-proxy")
_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
_jobs: dict[str, "QueuedJob"] = {}
_waiting_ids: deque[str] = deque()
_jobs_lock: asyncio.Lock | None = None
_worker_tasks: list[asyncio.Task[None]] = []
_limiter: "OutboundLimiter | None" = None
_embedding_limiter: "OutboundLimiter | None" = None
_embedding_semaphore: asyncio.Semaphore | None = None

app = FastAPI(title="Integral TP LLM Proxy", version="0.1.0")


@dataclass
class QueuedJob:
    id: str
    request: ChatRequest
    future: asyncio.Future[ChatResponse]
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    attempts: int = 0
    retry_count: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    next_retry_at: float | None = None
    last_error: str | None = None
    text: str | None = None
    usage: dict[str, Any] | None = None
    raw_usage: Any | None = None


class OutboundLimiter:
    def __init__(self, *, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(min_interval_seconds, 0.0)
        self._lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._backoff_until = 0.0

    async def wait_for_slot(self) -> float:
        async with self._lock:
            now = time.monotonic()
            request_at = max(now, self._next_request_at, self._backoff_until)
            wait_s = max(0.0, request_at - now)
            self._next_request_at = request_at + self.min_interval_seconds
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        return wait_s

    async def backoff(self, delay_s: float) -> None:
        if delay_s <= 0:
            return
        async with self._lock:
            self._backoff_until = max(self._backoff_until, time.monotonic() + delay_s)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.monotonic()
            return {
                "min_interval_seconds": self.min_interval_seconds,
                "backoff_remaining_s": max(0.0, self._backoff_until - now),
                "next_request_wait_s": max(0.0, self._next_request_at - now),
            }


@app.get("/health")
def health() -> dict[str, Any]:
    provider = (
        os.getenv("WORKSHOP_LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or ("openrouter" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("MISTRAL_API_KEY") else "mistral")
    ).strip().lower()
    model = (
        (os.getenv("OPENROUTER_MODEL") if provider == "openrouter" else os.getenv("MISTRAL_MODEL"))
        or os.getenv("LLM_MODEL")
        or (DEFAULT_OPENROUTER_MODEL if provider == "openrouter" else DEFAULT_MISTRAL_MODEL)
    )
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "max_workers": MAX_WORKERS,
        "max_concurrency": MAX_CONCURRENCY,
        "min_interval_seconds": MIN_INTERVAL_SECONDS,
        "max_queue_size": MAX_QUEUE_SIZE,
        "max_job_retries": MAX_JOB_RETRIES,
        "embedding_model": (
            os.getenv("OPENROUTER_EMBEDDING_MODEL")
            or DEFAULT_OPENROUTER_EMBEDDING_MODEL
        ),
        "embedding_max_concurrency": EMBEDDING_MAX_CONCURRENCY,
        "embedding_min_interval_seconds": EMBEDDING_MIN_INTERVAL_SECONDS,
        "embedding_max_retries": EMBEDDING_MAX_RETRIES,
    }


def _check_auth(authorization: str | None) -> None:
    expected = os.getenv("WORKSHOP_LLM_SERVER_TOKEN") or os.getenv("LLM_SERVER_TOKEN")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid LLM proxy token.")


def _complete_once(request: ChatRequest) -> ChatResponse:
    client = LLMClient.direct_from_env(model=request.model)
    client.max_retries = 0
    if request.reasoning_effort is not None:
        client.reasoning_effort = request.reasoning_effort or None
    if request.prompt_cache_key is not None:
        client.prompt_cache_key = request.prompt_cache_key or None
    result = client.chat_with_usage(
        system=request.system,
        user=request.user,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    return ChatResponse(
        text=result.text,
        usage=result.usage.to_dict(),
        raw_usage=result.raw_usage,
    )


def _configured_embedding_model() -> str:
    return (
        os.getenv("OPENROUTER_EMBEDDING_MODEL")
        or DEFAULT_OPENROUTER_EMBEDDING_MODEL
    ).strip()


def _embedding_inputs(request: EmbeddingRequest) -> list[str]:
    inputs = [request.input] if isinstance(request.input, str) else list(request.input)
    if not inputs or len(inputs) > 64:
        raise HTTPException(
            status_code=400,
            detail="Embedding requests must contain between 1 and 64 inputs.",
        )
    if any(not isinstance(item, str) or not item.strip() for item in inputs):
        raise HTTPException(status_code=400, detail="Embedding inputs must be non-empty text.")
    return inputs


def _embed_once(request: EmbeddingRequest) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for proxy-side embeddings.")

    model = _configured_embedding_model()
    if request.model and request.model.strip().lower() != model.lower():
        raise ValueError(
            f"Embedding model {request.model!r} is not enabled by this proxy; "
            f"use {model!r}."
        )
    payload: dict[str, Any] = {
        "input": _embedding_inputs(request),
        "model": model,
        "encoding_format": "float",
    }
    if request.input_type:
        payload["input_type"] = request.input_type
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "embeddings"),
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=_env_float("WORKSHOP_EMBEDDING_UPSTREAM_TIMEOUT_SECONDS", 120.0),
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise ValueError("OpenRouter returned an invalid embeddings response.")
    return body


def _short_error(exc: Exception, *, limit: int = 1000) -> str:
    text = " ".join(str(exc).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _is_rate_limit_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limited" in text


def _retry_delay(exc: Exception, *, retry_count: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = None
    try:
        retry_after = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        retry_after = None
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, BACKOFF_MAX_SECONDS)

    base = RATE_LIMIT_BACKOFF_INITIAL_SECONDS if _is_rate_limit_error(exc) else BACKOFF_INITIAL_SECONDS
    delay = base * (2 ** max(retry_count - 1, 0))
    delay += random.uniform(0.0, min(1.0, base))
    return min(delay, BACKOFF_MAX_SECONDS)


def _now() -> float:
    return time.time()


def _job_queue_stats(job: QueuedJob) -> dict[str, Any]:
    finished_at = job.finished_at or _now()
    started_at = job.started_at or finished_at
    return {
        "job_id": job.id,
        "status": job.status,
        "attempts": job.attempts,
        "retry_count": job.retry_count,
        "queued_wait_s": max(0.0, started_at - job.created_at),
        "total_elapsed_s": max(0.0, finished_at - job.created_at),
        "last_error": job.last_error,
    }


async def _lock() -> asyncio.Lock:
    global _jobs_lock
    if _jobs_lock is None:
        _jobs_lock = asyncio.Lock()
    return _jobs_lock


async def _enqueue(request: ChatRequest) -> QueuedJob:
    if _queue.full():
        raise HTTPException(status_code=503, detail="LLM proxy queue is full.")
    loop = asyncio.get_running_loop()
    job = QueuedJob(id=uuid.uuid4().hex, request=request, future=loop.create_future())
    async with await _lock():
        _jobs[job.id] = job
        _waiting_ids.append(job.id)
    try:
        _queue.put_nowait(job.id)
    except asyncio.QueueFull as exc:
        async with await _lock():
            _jobs.pop(job.id, None)
            try:
                _waiting_ids.remove(job.id)
            except ValueError:
                pass
        raise HTTPException(status_code=503, detail="LLM proxy queue is full.") from exc
    return job


async def _queue_position(job_id: str) -> int | None:
    async with await _lock():
        try:
            return list(_waiting_ids).index(job_id) + 1
        except ValueError:
            return None


async def _queue_snapshot() -> dict[str, Any]:
    async with await _lock():
        queued = len(_waiting_ids)
        running = sum(1 for job in _jobs.values() if job.status in {"running", "retrying"})
        completed = sum(1 for job in _jobs.values() if job.status in {"succeeded", "failed"})
        known = len(_jobs)
    limiter = _limiter
    limiter_stats = await limiter.snapshot() if limiter is not None else {}
    return {
        "queued": queued,
        "running": running,
        "completed": completed,
        "known_jobs": known,
        "max_queue_size": MAX_QUEUE_SIZE,
        "max_concurrency": MAX_CONCURRENCY,
        "max_job_retries": MAX_JOB_RETRIES,
        **limiter_stats,
    }


async def _status_response(job: QueuedJob) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        queue_position=await _queue_position(job.id),
        attempts=job.attempts,
        retry_count=job.retry_count,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        next_retry_at=job.next_retry_at,
        last_error=job.last_error,
        text=job.text if job.status == "succeeded" else None,
        usage=job.usage if job.status == "succeeded" else None,
        raw_usage=job.raw_usage if job.status == "succeeded" else None,
        queue=await _queue_snapshot(),
    )


async def _mark_job(job: QueuedJob, **updates: Any) -> None:
    async with await _lock():
        for key, value in updates.items():
            setattr(job, key, value)


async def _cleanup_old_jobs() -> None:
    cutoff = _now() - JOB_TTL_SECONDS
    async with await _lock():
        for job_id, job in list(_jobs.items()):
            if job.status in {"succeeded", "failed"} and (job.finished_at or 0.0) < cutoff:
                _jobs.pop(job_id, None)


async def _run_job(job: QueuedJob, *, worker_id: int) -> None:
    del worker_id
    loop = asyncio.get_running_loop()
    assert _limiter is not None
    await _mark_job(
        job,
        status="running",
        started_at=job.started_at or _now(),
        next_retry_at=None,
    )
    while True:
        await _limiter.wait_for_slot()
        await _mark_job(job, status="running", attempts=job.attempts + 1, next_retry_at=None)
        try:
            response = await loop.run_in_executor(_executor, _complete_once, job.request)
        except Exception as exc:
            retryable = _is_transient_llm_error(exc)
            can_retry = retryable and job.retry_count < MAX_JOB_RETRIES
            error = _short_error(exc)
            if not can_retry:
                await _mark_job(
                    job,
                    status="failed",
                    finished_at=_now(),
                    last_error=error,
                    next_retry_at=None,
                )
                if not job.future.done():
                    job.future.set_exception(HTTPException(status_code=502, detail=error))
                return

            retry_count = job.retry_count + 1
            delay_s = _retry_delay(exc, retry_count=retry_count)
            if _is_rate_limit_error(exc):
                await _limiter.backoff(delay_s)
            await _mark_job(
                job,
                status="retrying",
                retry_count=retry_count,
                last_error=error,
                next_retry_at=_now() + delay_s,
            )
            await asyncio.sleep(delay_s)
            continue

        await _mark_job(
            job,
            status="succeeded",
            finished_at=_now(),
            next_retry_at=None,
            last_error=None,
            text=response.text,
            usage=response.usage,
            raw_usage=response.raw_usage,
        )
        response.job_id = job.id
        response.queue = _job_queue_stats(job)
        if not job.future.done():
            job.future.set_result(response)
        return


async def _worker(worker_id: int) -> None:
    while True:
        job_id = await _queue.get()
        try:
            async with await _lock():
                job = _jobs.get(job_id)
                try:
                    _waiting_ids.remove(job_id)
                except ValueError:
                    pass
            if job is not None:
                await _run_job(job, worker_id=worker_id)
                await _cleanup_old_jobs()
        finally:
            _queue.task_done()


@app.on_event("startup")
async def _startup() -> None:
    global _embedding_limiter, _embedding_semaphore, _limiter
    _limiter = OutboundLimiter(min_interval_seconds=MIN_INTERVAL_SECONDS)
    _embedding_limiter = OutboundLimiter(
        min_interval_seconds=EMBEDDING_MIN_INTERVAL_SECONDS
    )
    _embedding_semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)
    _worker_tasks[:] = [
        asyncio.create_task(_worker(worker_id), name=f"llm-proxy-worker-{worker_id}")
        for worker_id in range(MAX_CONCURRENCY)
    ]


@app.on_event("shutdown")
async def _shutdown() -> None:
    for task in _worker_tasks:
        task.cancel()
    if _worker_tasks:
        await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _executor.shutdown(wait=False, cancel_futures=True)


@app.get("/queue")
async def queue_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_auth(authorization)
    return await _queue_snapshot()


@app.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Proxy query embeddings without exposing the OpenRouter key to notebooks."""

    _check_auth(authorization)
    _embedding_inputs(request)
    limiter = _embedding_limiter
    semaphore = _embedding_semaphore
    if limiter is None or semaphore is None:
        raise HTTPException(status_code=503, detail="Embedding proxy is not ready.")

    loop = asyncio.get_running_loop()
    retry_count = 0
    while True:
        await limiter.wait_for_slot()
        try:
            async with semaphore:
                return await loop.run_in_executor(_executor, _embed_once, request)
        except HTTPException:
            raise
        except Exception as exc:
            can_retry = (
                _is_transient_llm_error(exc)
                and retry_count < EMBEDDING_MAX_RETRIES
            )
            if not can_retry:
                status_code = 400 if isinstance(exc, ValueError) else 502
                raise HTTPException(
                    status_code=status_code,
                    detail=_short_error(exc),
                ) from exc
            retry_count += 1
            delay_s = _retry_delay(exc, retry_count=retry_count)
            if _is_rate_limit_error(exc):
                await limiter.backoff(delay_s)
            await asyncio.sleep(delay_s)


@app.post("/jobs")
async def enqueue_chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> EnqueueResponse:
    _check_auth(authorization)
    job = await _enqueue(request)
    return EnqueueResponse(
        job_id=job.id,
        status=job.status,
        queue_position=await _queue_position(job.id) or 0,
        status_url=f"/jobs/{job.id}",
    )


@app.get("/jobs/{job_id}")
async def job_status(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> JobStatusResponse:
    _check_auth(authorization)
    async with await _lock():
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown LLM job.")
    return await _status_response(job)


@app.post("/chat")
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    _check_auth(authorization)
    job = await _enqueue(request)
    try:
        return await asyncio.shield(job.future)
    except HTTPException:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Integral TP LLM proxy server.")
    parser.add_argument("--host", default=os.getenv("WORKSHOP_LLM_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("WORKSHOP_LLM_SERVER_PORT", 8010))
    parser.add_argument("--log-level", default=os.getenv("WORKSHOP_LLM_SERVER_LOG_LEVEL", "info"))
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - import error path.
        raise RuntimeError("Install `uvicorn` to run the LLM proxy server.") from exc

    uvicorn.run(
        "workshop_api.llm_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
