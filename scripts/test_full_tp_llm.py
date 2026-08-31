from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workshop_api import LLMClient
from workshop_api.llm import DEFAULT_OPENROUTER_MODEL
from scripts.test_workshop_api import build_analytic_doc


def hit(
    name: str,
    statement: str,
    *,
    library: str = "Coquelicot",
    kind: str = "theorem",
    docstring: str = "",
) -> dict[str, Any]:
    return {
        "uid": f"{library.lower()}:{name}",
        "name": name,
        "kind": kind,
        "library": library,
        "source": "test_full_tp_llm.py",
        "statement": statement,
        "docstring": docstring,
    }


EXP_POS = hit(
    "exp_pos",
    "Theorem exp_pos : forall x : R, 0 < exp x.",
    library="Stdlib",
)
RPLUS_LT_0_COMPAT = hit(
    "Rplus_lt_0_compat",
    "Theorem Rplus_lt_0_compat : forall r1 r2 : R, 0 < r1 -> 0 < r2 -> 0 < r1 + r2.",
    library="Stdlib",
)
RGT_NOT_EQ = hit(
    "Rgt_not_eq",
    "Theorem Rgt_not_eq : forall r1 r2 : R, r1 > r2 -> r1 <> r2.",
    library="Stdlib",
)
EXP_PLUS = hit(
    "exp_plus",
    "Lemma exp_plus : forall x y : R, exp (x + y) = exp x * exp y.",
    library="Stdlib",
)
AUTO_DERIVE = hit(
    "auto_derive",
    "Ltac auto_derive := ...",
    kind="Ltac",
    docstring="Coquelicot tactic for derivative and differentiability goals.",
)
IS_DERIVE_PLUS = hit(
    "is_derive_plus",
    """is_derive_plus :
  forall (f g : R -> R) (x df dg : R),
    is_derive f x df ->
    is_derive g x dg ->
    is_derive (fun y => f y + g y) x (df + dg)""",
)
EX_DERIVE_CONTINUOUS = hit(
    "ex_derive_continuous",
    "ex_derive_continuous : forall (f : R -> R) (x : R), ex_derive f x -> continuous f x",
)
IS_RINT_UNIQUE = hit(
    "is_RInt_unique",
    "is_RInt_unique : forall f a b If, is_RInt f a b If -> RInt f a b = If",
)
IS_RINT_DERIVE = hit(
    "is_RInt_derive",
    """is_RInt_derive :
  forall (f g : R -> R) (a b : R),
    (forall x, Rmin a b <= x <= Rmax a b -> is_derive f x (g x)) ->
    (forall x, Rmin a b <= x <= Rmax a b -> continuous g x) ->
    is_RInt g a b (f b - f a)""",
)


@dataclass
class RehearsalStep:
    name: str
    ok: bool
    requests: int
    script: str = ""


def result_goals(result: Any) -> str:
    goals = getattr(result, "goals", None) or []
    return "\n\n".join(str(goal) for goal in goals)


def prove_with_feedback(
    llm: LLMClient,
    theorem: Any,
    *,
    selected_hits: list[dict[str, Any]],
    attempts: int,
    extra_context: str = "",
    checkpoint: str = "llm_start",
    verbose: bool = False,
    max_tokens: int = 20_000,
    temperature: float | None = None,
) -> RehearsalStep:
    theorem.checkpoint(checkpoint)
    feedback_context = extra_context.strip()
    last_result = None
    for request_id in range(1, attempts + 1):
        result = llm.prove(
            theorem,
            selected_hits=selected_hits,
            extra_context=feedback_context,
            verbose=verbose,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        last_result = result
        if result.ok:
            return RehearsalStep(
                name=theorem.name,
                ok=True,
                requests=request_id,
                script=result.script,
            )
        feedback_context = (
            f"{extra_context.strip()}\n\n"
            f"Previous failed attempt #{request_id}:\n"
            f"```coq\n{result.script}\n```\n\n"
            f"Rocq error:\n{result.error}\n\n"
            f"Remaining goals:\n{result_goals(result) or '(none)'}\n"
        ).strip()
        theorem.reverse(checkpoint)

    assert last_result is not None
    raise AssertionError(
        f"{theorem.name} failed after {attempts} request(s).\n"
        f"Last script:\n{last_result.script}\n\n"
        f"Last error:\n{last_result.error}\n\n"
        f"Remaining goals:\n{result_goals(last_result)}"
    )


def prove_with_tools(
    llm: LLMClient,
    theorem: Any,
    *,
    selected_hits: list[dict[str, Any]],
    max_tool_calls: int,
    extra_context: str = "",
    checkpoint: str = "start",
    verbose: bool = False,
    max_tokens: int = 20_000,
    temperature: float | None = None,
) -> RehearsalStep:
    theorem.checkpoint(checkpoint)

    result = llm.prove(
        theorem,
        selected_hits=selected_hits,
        extra_context=extra_context.strip(),
        tools={
            "run_tac": theorem.run_tac,
            "reverse": theorem.reverse,
        },
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if result.ok:
        return RehearsalStep(
            name=theorem.name,
            ok=True,
            requests=len(result.attempts),
            script=result.script,
        )
    raise AssertionError(
        f"{theorem.name} failed after {len(result.attempts)} tool request(s).\n"
        f"Partial script:\n{result.script}\n\n"
        f"Last error:\n{result.error}\n\n"
        f"Remaining goals:\n{result_goals(result)}"
    )


def print_step(step: RehearsalStep) -> None:
    print(f"{step.name}: ok={step.ok} requests={step.requests}")
    for line in step.script.splitlines():
        print(f"  {line}")


def run_full_tp_llm(
    *,
    host: str,
    port: int,
    timeout: float,
    model: str,
    attempts: int,
    verbose: bool,
) -> list[RehearsalStep]:
    if not LLMClient.from_env(model=model).configured:
        raise RuntimeError(
            "Set WORKSHOP_LLM_SERVER_URL or MISTRAL_API_KEY before running the LLM rehearsal."
        )

    doc = build_analytic_doc(host, port, timeout)
    llm = LLMClient.from_env(model=model)
    steps: list[RehearsalStep] = []

    try:
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
        steps.append(
            prove_with_tools(
                llm,
                denominator,
                selected_hits=[EXP_POS, RPLUS_LT_0_COMPAT, RGT_NOT_EQ],
                max_tool_calls=30,
                extra_context=(
                    "Mathematically, the denominator is nonzero because it is "
                    "strictly positive: the exponential is positive, and adding "
                    "1 preserves strict positivity. Use the selected facts that "
                    "express these ideas."
                ),
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        technical_derivative_hits = [denominator.as_retrieval_hit(), EXP_PLUS]
        f2.reverse("after_auto_derive")
        steps.append(
            prove_with_tools(
                llm,
                f2,
                selected_hits=technical_derivative_hits,
                checkpoint="after_auto_derive",
                max_tool_calls=48,
                extra_context=(
                    "Mathematically, one goal is the nonzero denominator "
                    "condition. The remaining equality is a rational identity "
                    "after relating the exponential of twice an expression to a "
                    "product of exponentials: view the doubled argument as the "
                    "sum of two equal arguments, then use the selected "
                    "exponential-addition theorem before the final algebra. "
                    "Once the goal displays the exponential of that sum, the "
                    "next mathematical step is the selected exponential-addition "
                    "theorem."
                ),
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        f4 = doc.add_theorem(
            """Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4)."""
        )
        assert f4.run_tac("unfold F4, A4, sech, tanh_exp.")["ok"]
        assert f4.run_tac("auto_derive.")["ok"]
        steps.append(
            prove_with_tools(
                llm,
                f4,
                selected_hits=technical_derivative_hits + [AUTO_DERIVE, f2.as_retrieval_hit()],
                max_tool_calls=48,
                extra_context=(
                    "Mathematically this is the same calculation as the "
                    "power-2 derivative: denominators must be nonzero, and the "
                    "final equality is a rational identity after using the "
                    "exponential addition identity on a doubled argument "
                    "viewed as a sum of two equal arguments. The selected "
                    "F2_derivative proof is a successful pattern for this "
                    "calculation; once the sum form is visible, use the "
                    "selected exponential-addition theorem before the final "
                    "algebra."
                ),
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        f6 = doc.add_theorem(
            """Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6)."""
        )
        assert f6.run_tac("unfold F6, A6, sech, tanh_exp.")["ok"]
        assert f6.run_tac("auto_derive.")["ok"]
        steps.append(
            prove_with_tools(
                llm,
                f6,
                selected_hits=technical_derivative_hits
                + [AUTO_DERIVE, f2.as_retrieval_hit(), f4.as_retrieval_hit()],
                max_tool_calls=52,
                extra_context=(
                    "Mathematically this is the same calculation as the "
                    "previous even powers: denominators must be nonzero, and "
                    "the final equality is a rational identity after using the "
                    "exponential addition identity on a doubled argument "
                    "viewed as a sum of two equal arguments. The selected "
                    "F4_derivative proof is a successful pattern for this "
                    "calculation; once the sum form is visible, use the "
                    "selected exponential-addition theorem before the final "
                    "algebra."
                ),
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        f_derivative = doc.add_theorem(
            """Lemma F_derivative (x : R) :
  is_derive F x (f x)."""
        )
        steps.append(
            prove_with_tools(
                llm,
                f_derivative,
                selected_hits=[
                    f2.as_retrieval_hit(),
                    f4.as_retrieval_hit(),
                    f6.as_retrieval_hit(),
                    IS_DERIVE_PLUS,
                ],
                extra_context=(
                    "Mathematically, F is the sum of F2, F4, and F6, and f is "
                    "the corresponding sum of their derivatives. Use the "
                    "derivative rule for sums together with the derivative "
                    "lemmas already proved."
                ),
                max_tool_calls=40,
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        f_ex_derive = doc.add_theorem(
            """Lemma f_ex_derive (x : R) :
  ex_derive f x."""
        )
        steps.append(
            prove_with_tools(
                llm,
                f_ex_derive,
                selected_hits=[denominator.as_retrieval_hit(), AUTO_DERIVE],
                max_tool_calls=36,
                extra_context=(
                    "Mathematically, f is built from sums, powers, "
                    "exponentials, and quotients of differentiable functions. "
                    "The local helper sech is itself a quotient, so expose its "
                    "definition when derivative obligations mention sech. The "
                    "only obstruction to differentiability is division by zero, "
                    "handled by the local denominator lemma. Keep ex_derive as "
                    "the differentiability predicate; do not unfold its "
                    "internal definition."
                ),
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        f_continuous = doc.add_theorem(
            """Lemma f_continuous (x : R) :
  continuous f x."""
        )
        steps.append(
            prove_with_tools(
                llm,
                f_continuous,
                selected_hits=[f_ex_derive.as_retrieval_hit(), EX_DERIVE_CONTINUOUS],
                max_tool_calls=16,
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        closed_form = doc.add_theorem(
            """Theorem I_closed_form_correct :
  I = I_closed_form."""
        )
        steps.append(
            prove_with_tools(
                llm,
                closed_form,
                selected_hits=[
                    f_derivative.as_retrieval_hit(),
                    f_continuous.as_retrieval_hit(),
                    IS_RINT_UNIQUE,
                    IS_RINT_DERIVE,
                ],
                max_tool_calls=24,
                extra_context=(
                    "Mathematically, this is the fundamental theorem of "
                    "calculus: since F is an antiderivative of f on the "
                    "interval and f is continuous, the integral of f from 0 to "
                    "1 is F(1) - F(0). Use the selected formal facts "
                    "corresponding to this theorem and the two lemmas already "
                    "proved."
                ),
                verbose=verbose,
            )
        )
        print_step(steps[-1])

        return steps
    finally:
        doc.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    steps = run_full_tp_llm(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        model=args.model,
        attempts=args.attempts,
        verbose=args.verbose,
    )
    total_requests = sum(step.requests for step in steps)
    print(f"full_tp_llm_ok steps={len(steps)} requests={total_requests}")


if __name__ == "__main__":
    main()
