from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workshop_api import new_document


@dataclass
class StepTiming:
    name: str
    latency_s: float
    ok: bool = True
    error: str = ""


@dataclass
class UserResult:
    user_id: int
    ok: bool
    latency_s: float
    steps: list[StepTiming] = field(default_factory=list)
    error: str = ""
    failed_step: str = ""


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def summarize(results: list[UserResult]) -> dict[str, Any]:
    step_names: list[str] = []
    by_step: dict[str, list[float]] = {}
    for result in results:
        for step in result.steps:
            if step.name not in by_step:
                by_step[step.name] = []
                step_names.append(step.name)
            if step.ok:
                by_step[step.name].append(step.latency_s)
    return {
        "users": len(results),
        "ok": sum(result.ok for result in results),
        "failed": sum(not result.ok for result in results),
        "total_latency_s": summarize_values([result.latency_s for result in results]),
        "steps": {name: summarize_values(by_step[name]) for name in step_names},
        "sample_errors": [
            {
                "user_id": result.user_id,
                "failed_step": result.failed_step,
                "error": result.error[:800],
            }
            for result in results
            if not result.ok
        ][:10],
    }


def require_ok(label: str, result: dict[str, Any]) -> None:
    if not result.get("ok", False):
        raise RuntimeError(f"{label} failed: {result}")


def prove_by(theorem: Any, script: str) -> None:
    for out in theorem.run_script(script):
        require_ok(out.get("tactic", "tactic"), out)
    goals = theorem.goals()
    if goals:
        raise RuntimeError(f"Remaining goals for {theorem.name}: {goals}")
    require_ok(f"{theorem.name}.qed", theorem.qed())


class UserRunner:
    def __init__(self, user_id: int, *, host: str, port: int, timeout: float):
        self.user_id = user_id
        self.host = host
        self.port = port
        self.timeout = timeout
        self.steps: list[StepTiming] = []
        self.numerical_doc = None
        self.analytic_doc = None

    def timed(self, name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:
            latency = time.perf_counter() - started
            self.steps.append(StepTiming(name=name, latency_s=latency, ok=False, error=repr(exc)))
            raise
        latency = time.perf_counter() - started
        self.steps.append(StepTiming(name=name, latency_s=latency))
        return value

    def close(self) -> None:
        for doc in (self.numerical_doc, self.analytic_doc):
            if doc is None:
                continue
            try:
                doc.close()
            except Exception:
                pass

    def run(self) -> None:
        self.timed("01_numerical_doc_imports", self.numerical_doc_imports)
        self.timed("02_numerical_definitions", self.numerical_definitions)
        self.timed("03_interval_decimal_enclosure", self.interval_decimal_enclosure)
        self.timed("04_I_first_4_decimal_digits", self.I_first_4_decimal_digits)
        self.timed("05_analytic_doc_imports", self.analytic_doc_imports)
        self.timed("06_analytic_base_definitions", self.analytic_base_definitions)
        self.timed("07_antiderivative_definitions", self.antiderivative_definitions)
        self.timed("08_closed_form_definitions", self.closed_form_definitions)
        self.timed("09_F2_open_auto_derive", self.F2_open_auto_derive)
        self.timed("10_sech_denominator_nonzero", self.sech_denominator_nonzero)
        self.timed("11_F2_derivative_finish", self.F2_derivative_finish)
        self.timed("12_F4_derivative", self.F4_derivative)
        self.timed("13_F6_derivative", self.F6_derivative)
        self.timed("14_F_derivative", self.F_derivative)
        self.timed("15_f_ex_derive", self.f_ex_derive)
        self.timed("16_f_continuous", self.f_continuous)
        self.timed("17_I_closed_form_correct", self.I_closed_form_correct)

    def numerical_doc_imports(self) -> None:
        self.numerical_doc = new_document(host=self.host, port=self.port, timeout=self.timeout)
        self.numerical_doc.add_import("Coq", "Reals Lra Psatz")
        self.numerical_doc.add_import("Coquelicot", "Coquelicot")
        self.numerical_doc.add_import("Interval", "Tactic Plot")

    def numerical_definitions(self) -> None:
        doc = self.numerical_doc
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

    def interval_decimal_enclosure(self) -> None:
        self.numerical_doc.execute(
            """Do integral
  ltac:(let J := eval cbv [I f sech] in I in exact J)
  with (i_prec 25, i_degree 3, i_fuel 300,
        i_width (-15), i_decimal)."""
        )

    def I_first_4_decimal_digits(self) -> None:
        theorem = self.numerical_doc.add_theorem(
            """Theorem I_first_4_decimal_digits :
  Rabs (I - 0.2108) <= 1e-4."""
        )
        require_ok("I_digits unfold", theorem.run_tac("unfold I, f, sech."))
        require_ok(
            "I_digits interval",
            theorem.run_tac("integral with (i_prec 25, i_degree 3, i_fuel 300)."),
        )
        require_ok("I_digits qed", theorem.qed())

    def analytic_doc_imports(self) -> None:
        self.analytic_doc = new_document(host=self.host, port=self.port, timeout=self.timeout)
        self.analytic_doc.add_import("Coq", "Reals Lra Psatz")
        self.analytic_doc.add_import("Coquelicot", "Coquelicot")
        self.analytic_doc.add_import("Interval", "Tactic Plot")

    def analytic_base_definitions(self) -> None:
        doc = self.analytic_doc
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

    def antiderivative_definitions(self) -> None:
        doc = self.analytic_doc
        doc.add_definition(
            """Definition tanh_exp (u : R) : R :=
  (exp (2 * u) - 1) / (exp (2 * u) + 1)."""
        )
        doc.add_definition("Definition A2 (u : R) : R :=\n  tanh_exp u.")
        doc.add_definition(
            """Definition A4 (u : R) : R :=
  tanh_exp u - (/ 3) * (tanh_exp u)^3."""
        )
        doc.add_definition(
            """Definition A6 (u : R) : R :=
  tanh_exp u - (2 / 3) * (tanh_exp u)^3 + (/ 5) * (tanh_exp u)^5."""
        )

    def closed_form_definitions(self) -> None:
        doc = self.analytic_doc
        doc.add_definition("Definition F2 (x : R) : R :=\n  A2 (10 * x - 2) / 10.")
        doc.add_definition("Definition F4 (x : R) : R :=\n  A4 (100 * x - 40) / 100.")
        doc.add_definition("Definition F6 (x : R) : R :=\n  A6 (1000 * x - 600) / 1000.")
        doc.add_definition("Definition F (x : R) : R :=\n  F2 x + F4 x + F6 x.")
        doc.add_definition("Definition I_closed_form : R :=\n  F 1 - F 0.")

    def F2_open_auto_derive(self) -> None:
        self.f2 = self.analytic_doc.add_theorem(
            """Lemma F2_derivative (x : R) :
  is_derive F2 x ((sech (10 * x - 2)) ^ 2)."""
        )
        require_ok("F2 unfold", self.f2.run_tac("unfold F2, A2, sech, tanh_exp."))
        require_ok("F2 auto_derive", self.f2.run_tac("auto_derive."))
        self.f2.checkpoint("after_auto_derive")

    def sech_denominator_nonzero(self) -> None:
        theorem = self.analytic_doc.add_theorem(
            """Lemma sech_denominator_nonzero (u : R) :
  exp u + 1 <> 0."""
        )
        prove_by(
            theorem,
            """
            apply Rgt_not_eq with (r1 := ((exp (u)) + 1)) (r2 := 0).
            apply Rplus_lt_0_compat with (r1 := (exp (u))) (r2 := 1).
            apply exp_pos.
            lra.
            """,
        )

    def F2_derivative_finish(self) -> None:
        require_ok("F2 simpl denominator", self.f2.run_tac("simpl."))
        require_ok("F2 denominator", self.f2.run_tac("apply sech_denominator_nonzero."))
        prove_by(
            self.f2,
            """
            simpl.
            replace (10 * x + - (2)) with (10 * x - 2) by ring.
            replace (2 * (10 * x - 2)) with ((10 * x - 2) + (10 * x - 2)) by ring.
            repeat rewrite exp_plus.
            field; nra.
            """,
        )

    def F4_derivative(self) -> None:
        theorem = self.analytic_doc.add_theorem(
            """Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4)."""
        )
        require_ok("F4 unfold", theorem.run_tac("unfold F4, A4, sech, tanh_exp."))
        require_ok("F4 auto_derive", theorem.run_tac("auto_derive."))
        prove_by(
            theorem,
            """
            - repeat split.
              - apply sech_denominator_nonzero.
              - apply sech_denominator_nonzero.
            simpl.
            replace (100*x + - (40)) with (100* x - 40) by ring.
            replace (2 * (100 * x - 40)) with ((100 * x - 40) + (100 * x - 40)) by ring.
            rewrite exp_plus.
            field.
            nra.
            """,
        )

    def F6_derivative(self) -> None:
        theorem = self.analytic_doc.add_theorem(
            """Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6)."""
        )
        require_ok("F6 unfold", theorem.run_tac("unfold F6, A6, sech, tanh_exp."))
        require_ok("F6 auto_derive", theorem.run_tac("auto_derive."))
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

    def F_derivative(self) -> None:
        theorem = self.analytic_doc.add_theorem(
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

    def f_ex_derive(self) -> None:
        theorem = self.analytic_doc.add_theorem("Lemma f_ex_derive (x : R) :\n  ex_derive f x.")
        prove_by(
            theorem,
            """
            unfold f, sech.
            auto_derive.
            repeat split.
            all: apply sech_denominator_nonzero.
            """,
        )

    def f_continuous(self) -> None:
        theorem = self.analytic_doc.add_theorem("Lemma f_continuous (x : R) :\n  continuous f x.")
        prove_by(
            theorem,
            """
            apply (ex_derive_continuous f x).
            apply f_ex_derive.
            """,
        )

    def I_closed_form_correct(self) -> None:
        theorem = self.analytic_doc.add_theorem("Theorem I_closed_form_correct :\n  I = I_closed_form.")
        prove_by(
            theorem,
            """
            unfold I, I_closed_form.
            apply is_RInt_unique.
            apply (is_RInt_derive F f 0 1).
            - intros x _. apply F_derivative.
            - intros x _. apply f_continuous.
            """,
        )


def run_user(
    user_id: int,
    *,
    host: str,
    port: int,
    timeout: float,
    start_barrier: threading.Barrier,
) -> UserResult:
    runner = UserRunner(user_id, host=host, port=port, timeout=timeout)
    start_barrier.wait()
    started = time.perf_counter()
    try:
        runner.run()
        return UserResult(
            user_id=user_id,
            ok=True,
            latency_s=time.perf_counter() - started,
            steps=runner.steps,
        )
    except Exception as exc:
        failed_step = runner.steps[-1].name if runner.steps else "startup"
        return UserResult(
            user_id=user_id,
            ok=False,
            latency_s=time.perf_counter() - started,
            steps=runner.steps,
            error=repr(exc),
            failed_step=failed_step,
        )
    finally:
        runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the Rocq side of the integral TP.")
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--jsonl", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    barrier = threading.Barrier(args.users)
    results: list[UserResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_user,
                user_id,
                host=args.host,
                port=args.port,
                timeout=args.timeout,
                start_barrier=barrier,
            )
            for user_id in range(args.users)
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "completed": idx,
                        "users": args.users,
                        "user_id": result.user_id,
                        "ok": result.ok,
                        "latency_s": result.latency_s,
                        "failed_step": result.failed_step,
                        "error": result.error[:300],
                    }
                ),
                flush=True,
            )

    summary = summarize(results)
    summary["wall_time_s"] = time.perf_counter() - started
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    if args.jsonl:
        path = Path(args.jsonl)
        with path.open("w", encoding="utf-8") as fh:
            for result in sorted(results, key=lambda item: item.user_id):
                fh.write(json.dumps(asdict(result)) + "\n")
            fh.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
