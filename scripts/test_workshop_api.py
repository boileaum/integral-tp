from __future__ import annotations

import argparse

from workshop_api import LLMClient, new_document
from workshop_api.llm import DEFAULT_OPENROUTER_MODEL


EXP_PLUS_HIT = {
    "uid": "stdlib:exp_plus",
    "name": "exp_plus",
    "kind": "theorem",
    "library": "Stdlib",
    "source": "Stdlib/Reals/Rtrigo_def.v",
    "statement": "Lemma exp_plus : forall x y : R, exp (x + y) = exp x * exp y.",
}


def prove_by(theorem, script: str) -> None:
    outputs = theorem.run_script(script)
    for out in outputs:
        if not out.get("ok"):
            raise AssertionError(f"Rocq rejected `{out.get('tactic')}`: {out.get('error')}")
    goals = theorem.goals()
    if goals:
        raise AssertionError(f"Remaining goals for {theorem.name}: {goals}")
    qed = theorem.qed()
    if not qed.get("ok"):
        raise AssertionError(f"Qed failed for {theorem.name}: {qed}")


def build_analytic_doc(host: str, port: int, timeout: float):
    doc = new_document(host=host, port=port, timeout=timeout)
    doc.add_import("Coq", "Reals Lra Psatz")
    doc.add_import("Coquelicot", "Coquelicot")

    doc.add_definition(
        """Definition sech (u : R) : R :=
  2 * exp (u) / (exp (2 * u) + 1)."""
    )
    doc.add_definition(
        """Definition f (x : R) : R :=
    (sech (10 * x - 2))^2
  + (sech (100 * x - 40))^4
  + (sech (1000 * x - 600))^6."""
    )
    doc.add_definition("Definition I : R := RInt f 0 1.")
    doc.add_definition(
        """Definition tanh_exp (u : R) : R :=
  (exp (2 * u) - 1) / (exp (2 * u) + 1)."""
    )
    doc.add_definition(
        """Definition A2 (u : R) : R :=
  tanh_exp u."""
    )
    doc.add_definition(
        """Definition A4 (u : R) : R :=
  tanh_exp u - (/ 3) * (tanh_exp u)^3."""
    )
    doc.add_definition(
        """Definition A6 (u : R) : R :=
  tanh_exp u - (2 / 3) * (tanh_exp u)^3 + (/ 5) * (tanh_exp u)^5."""
    )
    doc.add_definition(
        """Definition F2 (x : R) : R :=
  A2 (10 * x - 2) / 10."""
    )
    doc.add_definition(
        """Definition F4 (x : R) : R :=
  A4 (100 * x - 40) / 100."""
    )
    doc.add_definition(
        """Definition F6 (x : R) : R :=
  A6 (1000 * x - 600) / 1000."""
    )
    doc.add_definition(
        """Definition F (x : R) : R :=
  F2 x + F4 x + F6 x."""
    )
    doc.add_definition(
        """Definition I_closed_form : R :=
  F 1 - F 0."""
    )
    return doc


def run_analytic_api(host: str, port: int, timeout: float) -> None:
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

    # This is the key workshop scenario: F2 was opened before the denominator
    # lemma existed. Closing the denominator lemma must refresh the open F2
    # proof state so that the new lemma is available.
    assert f2.run_tac("simpl.")["ok"]
    assert f2.run_tac("apply sech_denominator_nonzero.")["ok"]
    prove_by(
        f2,
        """
        simpl.
        replace (10 * x + - (2)) with (10 * x - 2) by ring.
        replace (2 * (10 * x - 2)) with ((10 * x - 2) + (10 * x - 2)) by ring.
        repeat rewrite exp_plus.
        field; nra.
        """,
    )

    f4 = doc.add_theorem(
        """Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4)."""
    )
    assert f4.run_tac("unfold F4, A4, sech, tanh_exp.")["ok"]
    assert f4.run_tac("auto_derive.")["ok"]
    prove_by(
        f4,
        """
        - repeat split.
          - apply sech_denominator_nonzero.
          - apply sech_denominator_nonzero.
        simpl.
        replace (100 * x + - (40)) with (100 * x - 40) by ring.
        replace (2 * (100 * x - 40)) with ((100 * x - 40) + (100 * x - 40)) by ring.
        rewrite exp_plus.
        field.
        nra.
        """,
    )

    f6 = doc.add_theorem(
        """Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6)."""
    )
    assert f6.run_tac("unfold F6, A6, sech, tanh_exp.")["ok"]
    assert f6.run_tac("auto_derive.")["ok"]
    prove_by(
        f6,
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

    f_derivative = doc.add_theorem(
        """Lemma F_derivative (x : R) :
  is_derive F x (f x)."""
    )
    prove_by(
        f_derivative,
        """
        unfold F, f.
        apply is_derive_plus with (f := fun x0 => ((F2 x0) + (F4 x0))) (g := F6).
        - apply is_derive_plus with (f := F2) (g := F4).
          + apply F2_derivative.
          + apply F4_derivative.
        - apply F6_derivative.
        """,
    )

    f_ex_derive = doc.add_theorem(
        """Lemma f_ex_derive (x : R) :
  ex_derive f x."""
    )
    prove_by(
        f_ex_derive,
        """
        unfold f, sech.
        auto_derive.
        repeat split.
        all: apply sech_denominator_nonzero.
        """,
    )

    f_continuous = doc.add_theorem(
        """Lemma f_continuous (x : R) :
  continuous f x."""
    )
    prove_by(
        f_continuous,
        """
        apply (ex_derive_continuous f x).
        apply f_ex_derive.
        """,
    )

    closed_form = doc.add_theorem(
        """Theorem I_closed_form_correct :
  I = I_closed_form."""
    )
    prove_by(
        closed_form,
        """
        unfold I, I_closed_form.
        apply is_RInt_unique.
        apply (is_RInt_derive F f 0 1).
        - intros x _. apply F_derivative.
        - intros x _. apply f_continuous.
        """,
    )

    print("analytic_api_ok")
    doc.close()


def run_llm_smoke(host: str, port: int, timeout: float, model: str) -> None:
    llm = LLMClient.from_env(model=model)
    if not llm.configured:
        print("llm_smoke_skipped_no_llm")
        return
    doc = new_document(host=host, port=port, timeout=timeout)
    theorem = doc.add_theorem("Lemma llm_smoke : 1 = 1.")
    result = llm.prove(theorem, selected_hits=[])
    if not result.ok:
        raise AssertionError(result)
    print("llm_smoke_ok")
    doc.close()


def run_section5_f2_llm(
    host: str,
    port: int,
    timeout: float,
    model: str,
    attempts: int,
    strategy: str,
) -> None:
    if not LLMClient.from_env(model=model).configured:
        print("section5_f2_llm_skipped_no_llm")
        return

    if strategy == "all":
        for name in ("direct", "feedback", "tools"):
            run_section5_f2_llm(host, port, timeout, model, attempts, name)
        return

    for attempt in range(1, attempts + 1):
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
        llm = LLMClient.from_env(model=model)
        selected_hits = [denominator.as_retrieval_hit(), EXP_PLUS_HIT]
        if strategy == "direct":
            result = llm.prove(
                f2,
                selected_hits=selected_hits,
            )
            request_count = 1
        elif strategy == "feedback":
            feedback_context = ""
            result = None
            request_count = 0
            for request_count in range(1, attempts + 1):
                result = llm.prove(
                    f2,
                    selected_hits=selected_hits,
                    extra_context=feedback_context,
                )
                if result.ok:
                    break
                feedback_context += f"""
Previous attempt #{request_count}:
{result.script}

Rocq feedback:
{result.error}
"""
                f2.reverse("after_auto_derive")
            assert result is not None
        elif strategy == "tools":
            result = llm.prove(
                f2,
                selected_hits=selected_hits,
                extra_context=(
                    "Mathematically, one goal is the nonzero denominator "
                    "condition. The remaining equality is a rational identity "
                    "after relating the exponential of twice an expression to "
                    "a product of exponentials: view the doubled argument as "
                    "the sum of two equal arguments, then use the selected "
                    "exponential-addition theorem before the final algebra. "
                    "Once the goal displays the exponential of that sum, the "
                    "next mathematical step is the selected exponential-addition "
                    "theorem."
                ),
                tools={
                    "run_tac": f2.run_tac,
                    "reverse": f2.reverse,
                },
                max_tool_calls=48,
            )
            request_count = len(result.attempts)
        else:
            raise ValueError(f"Unknown section 5 strategy: {strategy}")
        doc.close()
        print(
            f"section5_f2_{strategy}_attempt_{attempt}_ok={result.ok} "
            f"requests={request_count}"
        )
        if result.ok:
            print(f"section5_f2_{strategy}_ok")
            return
        print(result)
    raise AssertionError(
        f"Section 5 F2 LLM strategy `{strategy}` failed after {attempts} attempt(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--llm-smoke", action="store_true")
    parser.add_argument("--section5-f2-llm", action="store_true")
    parser.add_argument("--section5-attempts", type=int, default=1)
    parser.add_argument(
        "--section5-strategy",
        choices=["direct", "feedback", "tools", "all"],
        default="direct",
    )
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    args = parser.parse_args()

    run_analytic_api(args.host, args.port, args.timeout)
    if args.llm_smoke:
        run_llm_smoke(args.host, args.port, args.timeout, args.model)
    if args.section5_f2_llm:
        run_section5_f2_llm(
            args.host,
            args.port,
            args.timeout,
            args.model,
            args.section5_attempts,
            args.section5_strategy,
        )


if __name__ == "__main__":
    main()
