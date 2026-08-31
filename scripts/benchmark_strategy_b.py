from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workshop_api import LLMClient
from workshop_api.llm import DEFAULT_OPENROUTER_MODEL
from scripts.benchmark_strategy_a import SETUPS
from scripts.test_workshop_api import build_analytic_doc


@dataclass
class FeedbackTrialResult:
    proof: str
    trial: int
    ok: bool
    attempts_used: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    error: str = ""


def _goals_text(result: Any) -> str:
    goals = getattr(result, "goals", None) or []
    return "\n\n".join(str(goal) for goal in goals)


def run_feedback_trial(
    *,
    proof_name: str,
    setup: Callable[[Any], tuple[Any, list[dict[str, Any]], str]],
    trial: int,
    llm: LLMClient,
    host: str,
    port: int,
    timeout: float,
    max_tokens: int,
    attempts: int,
) -> FeedbackTrialResult:
    doc = build_analytic_doc(host, port, timeout)
    try:
        theorem, selected_hits, base_context = setup(doc)
        checkpoint = theorem.checkpoint("strategy_b_start")
        feedback_context = base_context.strip()
        input_tokens = 0
        output_tokens = 0
        total_cost_usd = 0.0
        last_result = None

        for attempt in range(1, attempts + 1):
            if attempt > 1:
                theorem.reverse(checkpoint)
            result = llm.prove(
                theorem,
                selected_hits=selected_hits,
                extra_context=feedback_context,
                max_tokens=max_tokens,
                close=False,
            )
            last_result = result
            input_tokens += result.usage.input_tokens
            output_tokens += result.usage.output_tokens
            total_cost_usd += result.usage.total_cost_usd
            if result.ok:
                return FeedbackTrialResult(
                    proof=proof_name,
                    trial=trial,
                    ok=True,
                    attempts_used=attempt,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_cost_usd=total_cost_usd,
                )

            feedback_context = (
                f"{base_context.strip()}\n\n"
                f"Previous failed attempt #{attempt}:\n"
                f"```coq\n{result.script}\n```\n\n"
                f"Rocq error:\n{result.error}\n\n"
                f"Remaining goals after the failed attempt:\n"
                f"{_goals_text(result) or '(none)'}\n"
            ).strip()

        assert last_result is not None
        return FeedbackTrialResult(
            proof=proof_name,
            trial=trial,
            ok=False,
            attempts_used=attempts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost_usd,
            error=str(last_result.error)[:500],
        )
    finally:
        doc.close()


def summarize(results: list[FeedbackTrialResult]) -> dict[str, Any]:
    successes = [result for result in results if result.ok]
    return {
        "proof": results[0].proof if results else "",
        "ok": len(successes),
        "trials": len(results),
        "avg_attempts_used": (
            sum(result.attempts_used for result in results) / len(results)
            if results
            else 0.0
        ),
        "avg_success_attempt": (
            sum(result.attempts_used for result in successes) / len(successes)
            if successes
            else None
        ),
        "input_tokens": sum(result.input_tokens for result in results),
        "output_tokens": sum(result.output_tokens for result in results),
        "total_cost_usd": sum(result.total_cost_usd for result in results),
        "sample_errors": [result.error for result in results if result.error][:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark strategy B on all notebook LLM proofs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=20_000)
    args = parser.parse_args()

    llm = LLMClient.direct_from_env(model=args.model)
    if not llm.configured:
        raise RuntimeError("Set MISTRAL_API_KEY before running the strategy B benchmark.")

    all_results: list[FeedbackTrialResult] = []
    summaries: list[dict[str, Any]] = []
    for proof_name, setup in SETUPS:
        proof_results: list[FeedbackTrialResult] = []
        print(f"=== {proof_name} ===", flush=True)
        for trial in range(1, args.trials + 1):
            result = run_feedback_trial(
                proof_name=proof_name,
                setup=setup,
                trial=trial,
                llm=llm,
                host=args.host,
                port=args.port,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                attempts=args.attempts,
            )
            proof_results.append(result)
            all_results.append(result)
            print(json.dumps(asdict(result)), flush=True)
        summary = summarize(proof_results)
        summaries.append(summary)
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    print("=== FINAL SUMMARY ===")
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))
    print(
        "TOTAL "
        + json.dumps(
            {
                "proofs": len(summaries),
                "trials": len(all_results),
                "ok": sum(1 for result in all_results if result.ok),
                "input_tokens": sum(result.input_tokens for result in all_results),
                "output_tokens": sum(result.output_tokens for result in all_results),
                "total_cost_usd": sum(result.total_cost_usd for result in all_results),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
