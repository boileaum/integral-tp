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
from scripts.test_full_tp_llm import (
    AUTO_DERIVE,
    EX_DERIVE_CONTINUOUS,
    EXP_PLUS,
    EXP_POS,
    IS_DERIVE_PLUS,
    IS_RINT_DERIVE,
    IS_RINT_UNIQUE,
    RGT_NOT_EQ,
    RPLUS_LT_0_COMPAT,
)
from scripts.test_workshop_api import build_analytic_doc, prove_by


@dataclass
class TrialResult:
    proof: str
    trial: int
    ok: bool
    requests: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    error: str = ""


DENOMINATOR_CONTEXT = (
    "Mathematically, the denominator is nonzero because it is strictly "
    "positive: the exponential is positive, and adding 1 preserves strict "
    "positivity. Use the selected facts that express these ideas."
)

F2_CONTEXT = (
    "Mathematically, one goal is the nonzero denominator condition. The "
    "remaining equality is a rational identity after relating the exponential "
    "of twice an expression to a product of exponentials."
)

F_DERIVATIVE_CONTEXT = (
    "Mathematically, F is the sum of F2, F4, and F6, and f is the "
    "corresponding sum of their derivatives. Use the derivative rule for sums "
    "together with the derivative lemmas already proved."
)

F_EX_CONTEXT = (
    "Mathematically, f is built from sums, powers, exponentials, and quotients "
    "of differentiable functions. The only obstruction to differentiability is "
    "division by zero, handled by the local denominator lemma."
)

CLOSED_FORM_CONTEXT = (
    "Mathematically, this is the fundamental theorem of calculus: since F is "
    "an antiderivative of f on the interval and f is continuous, the integral "
    "of f from 0 to 1 is F(1) - F(0). Use the selected formal facts "
    "corresponding to this theorem and the two lemmas already proved."
)


def add_denominator(doc: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma sech_denominator_nonzero (u : R) :
  exp u + 1 <> 0."""
    )
    prove_by(
        theorem,
        """
        apply Rgt_not_eq with (r1 := ((exp u) + 1)) (r2 := 0).
        apply Rplus_lt_0_compat with (r1 := exp u) (r2 := 1).
        apply exp_pos.
        lra.
        """,
    )
    return theorem


def add_f2(doc: Any, denominator: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma F2_derivative (x : R) :
  is_derive F2 x ((sech (10 * x - 2)) ^ 2)."""
    )
    assert theorem.run_tac("unfold F2, A2, sech, tanh_exp.")["ok"]
    assert theorem.run_tac("auto_derive.")["ok"]
    prove_by(
        theorem,
        """
        apply sech_denominator_nonzero.
        simpl.
        replace (10 * x + - (2)) with (10 * x - 2) by ring.
        replace (2 * (10 * x - 2)) with ((10 * x - 2) + (10 * x - 2)) by ring.
        rewrite exp_plus.
        field; nra.
        """,
    )
    return theorem


def add_f4(doc: Any, denominator: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4)."""
    )
    assert theorem.run_tac("unfold F4, A4, sech, tanh_exp.")["ok"]
    assert theorem.run_tac("auto_derive.")["ok"]
    prove_by(
        theorem,
        """
        repeat split.
        apply sech_denominator_nonzero.
        apply sech_denominator_nonzero.
        simpl.
        replace (100 * x + - (40)) with (100 * x - 40) by ring.
        replace (2 * (100 * x - 40)) with ((100 * x - 40) + (100 * x - 40)) by ring.
        rewrite exp_plus.
        field.
        nra.
        """,
    )
    return theorem


def add_f6(doc: Any, denominator: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6)."""
    )
    assert theorem.run_tac("unfold F6, A6, sech, tanh_exp.")["ok"]
    assert theorem.run_tac("auto_derive.")["ok"]
    prove_by(
        theorem,
        """
        repeat split.
        apply sech_denominator_nonzero.
        apply sech_denominator_nonzero.
        apply sech_denominator_nonzero.
        trivial.
        simpl.
        replace (1000 * x + - (600)) with (1000 * x - 600) by ring.
        replace (2 * (1000 * x - 600)) with ((1000 * x - 600) + (1000 * x - 600)) by ring.
        rewrite exp_plus.
        field.
        nra.
        """,
    )
    return theorem


def add_f_derivative(doc: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma F_derivative (x : R) :
  is_derive F x (f x)."""
    )
    prove_by(
        theorem,
        """
        unfold F, f.
        apply is_derive_plus with (f := fun x0 => ((F2 x0) + (F4 x0))) (g := F6).
        - apply is_derive_plus with (f := F2) (g := F4).
          + apply F2_derivative.
          + apply F4_derivative.
        - apply F6_derivative.
        """,
    )
    return theorem


def add_f_ex_derive(doc: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma f_ex_derive (x : R) :
  ex_derive f x."""
    )
    prove_by(
        theorem,
        """
        unfold f, sech.
        auto_derive.
        repeat split.
        all: apply sech_denominator_nonzero.
        """,
    )
    return theorem


def add_f_continuous(doc: Any) -> Any:
    theorem = doc.add_theorem(
        """Lemma f_continuous (x : R) :
  continuous f x."""
    )
    prove_by(
        theorem,
        """
        apply (ex_derive_continuous f x).
        apply f_ex_derive.
        """,
    )
    return theorem


def setup_denominator(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    theorem = doc.add_theorem(
        """Lemma sech_denominator_nonzero (u : R) :
  exp u + 1 <> 0."""
    )
    return theorem, [EXP_POS, RPLUS_LT_0_COMPAT, RGT_NOT_EQ], DENOMINATOR_CONTEXT


def setup_f2(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    theorem = doc.add_theorem(
        """Lemma F2_derivative (x : R) :
  is_derive F2 x ((sech (10 * x - 2)) ^ 2)."""
    )
    assert theorem.run_tac("unfold F2, A2, sech, tanh_exp.")["ok"]
    assert theorem.run_tac("auto_derive.")["ok"]
    theorem.checkpoint("after_auto_derive")
    denominator = add_denominator(doc)
    theorem.reverse("after_auto_derive")
    return theorem, [denominator.as_retrieval_hit(), EXP_PLUS], F2_CONTEXT


def setup_f4(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    denominator = add_denominator(doc)
    f2 = add_f2(doc, denominator)
    theorem = doc.add_theorem(
        """Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4)."""
    )
    assert theorem.run_tac("unfold F4, A4, sech, tanh_exp.")["ok"]
    assert theorem.run_tac("auto_derive.")["ok"]
    context = (
        "Mathematically this is the same calculation as the power-2 "
        "derivative: denominators must be nonzero, and the final equality is a "
        "rational identity after using the exponential addition identity."
    )
    return theorem, [denominator.as_retrieval_hit(), EXP_PLUS, AUTO_DERIVE], context


def setup_f6(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    denominator = add_denominator(doc)
    f2 = add_f2(doc, denominator)
    f4 = add_f4(doc, denominator)
    theorem = doc.add_theorem(
        """Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6)."""
    )
    assert theorem.run_tac("unfold F6, A6, sech, tanh_exp.")["ok"]
    assert theorem.run_tac("auto_derive.")["ok"]
    context = (
        "Mathematically this is the same calculation as the previous even "
        "powers: denominators must be nonzero, and the final equality is a "
        "rational identity after using the exponential addition identity."
    )
    return theorem, [denominator.as_retrieval_hit(), EXP_PLUS, AUTO_DERIVE], context


def setup_f_derivative(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    denominator = add_denominator(doc)
    f2 = add_f2(doc, denominator)
    f4 = add_f4(doc, denominator)
    f6 = add_f6(doc, denominator)
    theorem = doc.add_theorem(
        """Lemma F_derivative (x : R) :
  is_derive F x (f x)."""
    )
    return (
        theorem,
        [f2.as_retrieval_hit(), f4.as_retrieval_hit(), f6.as_retrieval_hit(), IS_DERIVE_PLUS],
        F_DERIVATIVE_CONTEXT,
    )


def setup_f_ex_derive(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    denominator = add_denominator(doc)
    theorem = doc.add_theorem(
        """Lemma f_ex_derive (x : R) :
  ex_derive f x."""
    )
    return theorem, [denominator.as_retrieval_hit(), AUTO_DERIVE], F_EX_CONTEXT


def setup_f_continuous(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    denominator = add_denominator(doc)
    f_ex = add_f_ex_derive(doc)
    theorem = doc.add_theorem(
        """Lemma f_continuous (x : R) :
  continuous f x."""
    )
    return theorem, [f_ex.as_retrieval_hit(), EX_DERIVE_CONTINUOUS], ""


def setup_closed_form(doc: Any) -> tuple[Any, list[dict[str, Any]], str]:
    denominator = add_denominator(doc)
    add_f2(doc, denominator)
    add_f4(doc, denominator)
    add_f6(doc, denominator)
    f_derivative = add_f_derivative(doc)
    add_f_ex_derive(doc)
    f_continuous = add_f_continuous(doc)
    theorem = doc.add_theorem(
        """Theorem I_closed_form_correct :
  I = I_closed_form."""
    )
    return (
        theorem,
        [
            f_derivative.as_retrieval_hit(),
            f_continuous.as_retrieval_hit(),
            IS_RINT_UNIQUE,
            IS_RINT_DERIVE,
        ],
        CLOSED_FORM_CONTEXT,
    )


SETUPS: list[tuple[str, Callable[[Any], tuple[Any, list[dict[str, Any]], str]]]] = [
    ("sech_denominator_nonzero", setup_denominator),
    ("F2_derivative", setup_f2),
    ("F4_derivative", setup_f4),
    ("F6_derivative", setup_f6),
    ("F_derivative", setup_f_derivative),
    ("f_ex_derive", setup_f_ex_derive),
    ("f_continuous", setup_f_continuous),
    ("I_closed_form_correct", setup_closed_form),
]


def run_trial(
    *,
    proof_name: str,
    setup: Callable[[Any], tuple[Any, list[dict[str, Any]], str]],
    trial: int,
    llm: LLMClient,
    host: str,
    port: int,
    timeout: float,
    max_tokens: int,
) -> TrialResult:
    doc = build_analytic_doc(host, port, timeout)
    try:
        theorem, selected_hits, extra_context = setup(doc)
        result = llm.prove(
            theorem,
            selected_hits=selected_hits,
            extra_context=extra_context,
            max_tokens=max_tokens,
            close=False,
        )
        return TrialResult(
            proof=proof_name,
            trial=trial,
            ok=result.ok,
            requests=1,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_cost_usd=result.usage.total_cost_usd,
            error=result.error[:500],
        )
    finally:
        doc.close()


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    return {
        "proof": results[0].proof if results else "",
        "ok": sum(1 for result in results if result.ok),
        "trials": len(results),
        "input_tokens": sum(result.input_tokens for result in results),
        "output_tokens": sum(result.output_tokens for result in results),
        "total_cost_usd": sum(result.total_cost_usd for result in results),
        "sample_errors": [result.error for result in results if result.error][:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark strategy A on all notebook LLM proofs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=20_000)
    args = parser.parse_args()

    llm = LLMClient.direct_from_env(model=args.model)
    if not llm.configured:
        raise RuntimeError("Set MISTRAL_API_KEY before running the strategy A benchmark.")

    all_results: list[TrialResult] = []
    summaries: list[dict[str, Any]] = []
    for proof_name, setup in SETUPS:
        proof_results: list[TrialResult] = []
        print(f"=== {proof_name} ===", flush=True)
        for trial in range(1, args.trials + 1):
            result = run_trial(
                proof_name=proof_name,
                setup=setup,
                trial=trial,
                llm=llm,
                host=args.host,
                port=args.port,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
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
