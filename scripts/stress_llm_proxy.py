from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workshop_api import LLMClient
from workshop_api.llm import DEFAULT_OPENROUTER_MODEL
from scripts.test_workshop_api import build_analytic_doc, prove_by


EXP_PLUS_HIT = {
    "uid": "stdlib:exp_plus",
    "name": "exp_plus",
    "kind": "theorem",
    "library": "Stdlib",
    "source": "Stdlib/Reals/Rtrigo_def.v",
    "statement": "Lemma exp_plus : forall x y : R, exp (x + y) = exp x * exp y.",
}


@dataclass
class StressResult:
    task_id: int
    ok: bool
    latency_s: float
    llm_requests: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    error_kind: str = ""
    error: str = ""


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def classify_error(error: str) -> str:
    lowered = error.lower()
    network_markers = [
        "llm call failed",
        "timeout",
        "connecterror",
        "readerror",
        "remoteprotocolerror",
        "httpstatuserror",
        "connection",
        "network",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    ]
    if any(marker in lowered for marker in network_markers):
        return "network"
    if error:
        return "proof"
    return ""


def prepare_f2_case(task_id: int, *, host: str, port: int, timeout: float) -> tuple[Any, Any, list[dict[str, Any]]]:
    doc = build_analytic_doc(host, port, timeout)
    f2 = doc.add_theorem(
        """Lemma F2_derivative (x : R) :
  is_derive F2 x ((sech (10 * x - 2)) ^ 2)."""
    )
    assert f2.run_tac("unfold F2, A2, sech, tanh_exp.")["ok"]
    assert f2.run_tac("auto_derive.")["ok"]
    f2.checkpoint("after_auto_derive")

    denominator = doc.add_theorem(
        """Lemma sech_denominator_nonzero (u : R) :
  exp u + 1 <> 0."""
    )
    prove_by(
        denominator,
        """
        apply Rgt_not_eq with (r1 := ((exp u) + 1)) (r2 := 0).
        apply Rplus_lt_0_compat with (r1 := exp u) (r2 := 1).
        apply exp_pos.
        lra.
        """,
    )
    f2.reverse("after_auto_derive")
    return doc, f2, [denominator.as_retrieval_hit(), EXP_PLUS_HIT]


def run_f2_tools_task(
    task_id: int,
    *,
    theorem: Any,
    selected_hits: list[dict[str, Any]],
    model: str,
    max_tool_calls: int,
    max_tokens: int,
    start_barrier: threading.Barrier,
) -> StressResult:
    llm = LLMClient.from_env(model=model)
    start_barrier.wait()
    started = time.perf_counter()
    try:
        result = llm.prove(
            theorem,
            selected_hits=selected_hits,
            extra_context=(
                "Mathematically, one goal is the nonzero denominator "
                "condition. The remaining equality is a rational identity "
                "after relating the exponential of twice an expression to a "
                "product of exponentials: view the doubled argument as the "
                "sum of two equal arguments, then use the selected "
                "exponential-addition theorem before the final algebra. Once "
                "the goal displays the exponential of that sum, the next "
                "mathematical step is the selected exponential-addition theorem."
            ),
            tools={
                "run_tac": theorem.run_tac,
                "reverse": theorem.reverse,
            },
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            verbose=False,
            close=False,
        )
        latency = time.perf_counter() - started
        return StressResult(
            task_id=task_id,
            ok=result.ok,
            latency_s=latency,
            llm_requests=len(result.usage_events) or len(result.attempts),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_cost_usd=result.usage.total_cost_usd,
            error_kind=classify_error(result.error),
            error=result.error[:500],
        )
    except Exception as exc:
        latency = time.perf_counter() - started
        error = repr(exc)
        return StressResult(
            task_id=task_id,
            ok=False,
            latency_s=latency,
            llm_requests=0,
            input_tokens=0,
            output_tokens=0,
            total_cost_usd=0.0,
            error_kind=classify_error(error),
            error=error[:500],
        )


def summarize(results: list[StressResult], *, setup_latency_s: float) -> dict[str, Any]:
    latencies = [result.latency_s for result in results]
    ok_count = sum(1 for result in results if result.ok)
    network_failures = sum(1 for result in results if result.error_kind == "network")
    proof_failures = sum(1 for result in results if result.error_kind == "proof")
    return {
        "tasks": len(results),
        "ok": ok_count,
        "failed": len(results) - ok_count,
        "network_failures": network_failures,
        "proof_failures": proof_failures,
        "setup_latency_s": setup_latency_s,
        "latency_s": {
            "min": min(latencies) if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "llm_requests": sum(result.llm_requests for result in results),
        "input_tokens": sum(result.input_tokens for result in results),
        "output_tokens": sum(result.output_tokens for result in results),
        "total_cost_usd": sum(result.total_cost_usd for result in results),
        "sample_errors": [
            {"task_id": result.task_id, "kind": result.error_kind, "error": result.error}
            for result in results
            if result.error
        ][:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the workshop LLM proxy.")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--setup-concurrency", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=20_000)
    args = parser.parse_args()

    if not LLMClient.from_env(model=args.model).configured:
        raise RuntimeError("Set WORKSHOP_LLM_SERVER_URL or MISTRAL_API_KEY before stress testing.")

    setup_started = time.perf_counter()
    cases: list[tuple[Any, Any, list[dict[str, Any]]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.setup_concurrency) as pool:
        futures = [
            pool.submit(prepare_f2_case, task_id, host=args.host, port=args.port, timeout=args.timeout)
            for task_id in range(args.tasks)
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            case = future.result()
            cases.append(case)
            print(f"setup {idx}/{args.tasks}", flush=True)
    setup_latency_s = time.perf_counter() - setup_started

    barrier = threading.Barrier(args.tasks)
    results: list[StressResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_f2_tools_task,
                task_id,
                theorem=theorem,
                selected_hits=selected_hits,
                model=args.model,
                max_tool_calls=args.max_tool_calls,
                max_tokens=args.max_tokens,
                start_barrier=barrier,
            )
            for task_id, (_doc, theorem, selected_hits) in enumerate(cases)
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(json.dumps(asdict(result)), flush=True)
            print(f"completed {idx}/{args.tasks}", flush=True)

    for doc, _theorem, _hits in cases:
        try:
            doc.close()
        except Exception:
            pass

    print("SUMMARY " + json.dumps(summarize(results, setup_latency_s=setup_latency_s), sort_keys=True))


if __name__ == "__main__":
    main()
