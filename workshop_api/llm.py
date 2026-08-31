from __future__ import annotations

import json
import os
import random
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin

import httpx

from .retrieval import format_retrieval_hits


try:
    try:
        from mistralai.client import Mistral
    except Exception:
        from mistralai import Mistral
except Exception as exc:  # pragma: no cover - depends on optional dependency version.
    Mistral = None  # type: ignore[assignment]
    _MISTRAL_IMPORT_ERROR = exc
else:
    _MISTRAL_IMPORT_ERROR = None

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - depends on optional dependency.
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

DEFAULT_MISTRAL_MODEL = "mistral-medium-latest"
DEFAULT_OPENROUTER_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_LLM_MAX_TOKENS = 20_000
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MISTRAL_MEDIUM_INPUT_USD_PER_MILLION = 1.50
MISTRAL_MEDIUM_CACHED_INPUT_USD_PER_MILLION = 0.15
MISTRAL_MEDIUM_OUTPUT_USD_PER_MILLION = 7.50
GLM_5_3_FLASH_INPUT_USD_PER_MILLION = 0.075
GLM_5_3_FLASH_CACHED_INPUT_USD_PER_MILLION = 0.015
GLM_5_3_FLASH_OUTPUT_USD_PER_MILLION = 0.25

MODEL_PRICES_USD_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "mistral-medium-latest": (
        MISTRAL_MEDIUM_INPUT_USD_PER_MILLION,
        MISTRAL_MEDIUM_CACHED_INPUT_USD_PER_MILLION,
        MISTRAL_MEDIUM_OUTPUT_USD_PER_MILLION,
    ),
    "mistralai/mistral-medium-3-5": (
        MISTRAL_MEDIUM_INPUT_USD_PER_MILLION,
        MISTRAL_MEDIUM_CACHED_INPUT_USD_PER_MILLION,
        MISTRAL_MEDIUM_OUTPUT_USD_PER_MILLION,
    ),
    "z-ai/glm-5.3-flash": (
        GLM_5_3_FLASH_INPUT_USD_PER_MILLION,
        GLM_5_3_FLASH_CACHED_INPUT_USD_PER_MILLION,
        GLM_5_3_FLASH_OUTPUT_USD_PER_MILLION,
    ),
}


ROCQ_SYSTEM_PROMPT = textwrap.dedent(
    """
    You write short Rocq/Coq proof scripts for a teaching notebook.

    Output rules:
    - Return only Rocq tactic commands unless explicitly asked for JSON.
    - Do not include `Proof.`, `Qed.`, `Admitted.`, or theorem statements.
    - Use Rocq/Coq tactics, not Lean tactics. For example, use
      `reflexivity.` rather than `rfl.`.
    - Do not use Lean/SSReflect-style proof terms such as `by ...`,
      `norm_num`, or `have h : P by ...`; those are not available in this
      environment. Use ordinary Rocq tactic commands instead.
    - Never use `norm_num`; it is not imported in this Rocq environment.
    - Do not use `have`. In this workshop environment, named intermediate
      facts should use ordinary Rocq tactics such as `assert`, and direct
      proof steps are usually preferable.
    - Prefer short, robust scripts over clever one-liners.
    - Use retrieved facts and local lemmas when they are relevant, but do not
      force them if the current goal has a different shape.
    - If the selected/local context contains a completed proof of a very
      similar theorem, prefer adapting its tactic sequence and normalization
      direction instead of inventing a different route.

    Rocq syntax and proof-state rules:
    - Tactics are executed on the focused goal. When the proof state lists
      several different goals, solve the first goal, then continue with the next
      one.
    - Do not use `all:` unless every remaining goal has the same syntactic shape
      and the tactic is appropriate for all of them.
    - Treat Rocq terms as syntax trees. Use parentheses when grouping matters.
      In particular, chained infix expressions such as `A + B + C` should be
      matched as the printed syntax tree, typically `(A + B) + C`, not as an
      algebraically equivalent regrouping. With a binary lemma, first split
      that shape into the left component `A + B` and the right component `C`.
    - When using `replace` or a targeted rewrite, copy the left-hand side from
      the current goal carefully, including parentheses and unary minus syntax.
      Put the intended normalized expression on the right.
    - In `replace A with B by proof`, the proof after `by` must be one tactic
      expression. If it needs several commands, wrap them in parentheses, for
      example `replace A with B by (rewrite some_lemma; ring).`
    - When a selected rewrite lemma has a recognizable left-hand syntactic
      pattern, transform the goal toward that pattern. Avoid replacing useful
      subexpressions by more expanded or divided arithmetic forms unless that
      directly matches a selected fact or a current algebraic side goal.
    - For binary lemmas, instantiate the split that matches the current
      top-level syntax rather than an algebraically equivalent regrouping.
    - Split only goals that are visibly conjunctions. Do not expose the
      internal definition of semantic predicates just because they may be
      implemented as records or conjunctions.
    - Do not unfold semantic predicates such as `ex_derive`, `is_derive`,
      `continuous`, or `is_RInt`; use suitable lemmas or automation for them.
    - If a lemma has ambiguous implicit arguments, instantiate the visible
      mathematical arguments explicitly with `with (...)` or by applying a
      partially instantiated lemma.
    - When applying a lemma to an explicit ordinary argument, do not write
      `apply lemma with (term).` That syntax is for named instantiations.
      Prefer either `apply (lemma term).` or
      `apply lemma with (name := term).` when the binder name is known.
      Do not write `apply lemma with (...) by tactic.`; if conversion is needed,
      first use a separate `replace ... by tactic.` command.
    - Never refer to a local hypothesis unless it is visibly present above the
      goal line. If Rocq feedback says a name was not found, continue from the
      displayed current goal rather than reusing that name.
    - To introduce several binders at once, use `intros`, for example
      `intros x _.`; do not write `intro x _.`. A single `intro` introduces only
      one binder.
    - If earlier setup tactics already appear in the partial proof, continue
      from the current proof state. Do not restart the proof unless the prompt
      explicitly asks you to.
    - If a multi-command script fails at an `apply` or unification step, test
      the corrected top-level command with one `run_tac` before proposing
      another full script.
    - After an automation tactic, inspect the generated goals. Some goals may be
      side conditions and others may be equalities; choose tactics according to
      the focused goal.
    - If automation leaves goals about a locally defined helper function as an
      opaque function, unfold that helper definition once before continuing.

    Arithmetic and algebra tactics:
    - Use `ring.` for polynomial identities over `+`, `-`, `*`, constants, and
      variables.
    - Use `field.` for equalities of rational expressions involving `/`; it may
      generate denominator side goals.
    - Use `lra.` for linear real arithmetic side conditions.
      This includes simple numeric real goals such as positivity of constants.
    - Use `nra.` for nonlinear polynomial arithmetic and side conditions
      generated by `field.`.
    - Algebra tactics treat opaque function applications as atoms. If an
      equality depends on a mathematical identity for such a function, use an
      appropriate rewrite lemma before expecting algebraic tactics to finish.
    - Small justified transformations such as `simpl.` and
      `replace ... by ring.` can help align a goal with a rewrite lemma or with
      the expected shape of an algebra tactic.
    - A rational equality tactic may generate side conditions; solve those
      follow-up goals with hypotheses, selected lemmas, or arithmetic tactics
      that match their shape.
    - After `field`, solve generated side conditions with the hypotheses,
      retrieved/local lemmas, or arithmetic tactics that match their shape.
    """
).strip()


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1).strip().lower(), match.group(2).strip())
        for match in re.finditer(r"```([A-Za-z0-9_-]*)?\s*(.*?)```", text, flags=re.S)
    ]


def _strip_code_fences(text: str, *, prefer_json: bool = False) -> str:
    raw = str(text).strip()
    fenced = _fenced_blocks(raw)
    if fenced:
        if prefer_json:
            for lang, block in reversed(fenced):
                if lang == "json" or block.lstrip().startswith("{"):
                    return block
        for lang, block in reversed(fenced):
            if lang in {"coq", "rocq", ""} and _looks_like_rocq_script(block):
                return block
        return fenced[-1][1].strip()
    return raw


_TACTIC_START_RE = re.compile(
    r"^\s*(?:[-+*{}]\s*)?(?:all:\s*)?"
    r"(?:apply|eapply|exact|reflexivity|assumption|trivial|simpl|cbn|cbv|"
    r"unfold|fold|change|rewrite|replace|field|ring|lra|nra|auto_derive|"
    r"repeat|split|intros|intro|destruct|constructor|exists|left|right|"
    r"now|try|solve|first|last|assert|pose|specialize|transitivity|"
    r"clear|subst|revert|generalize|set|remember|enough|cut|"
    r"tauto|intuition|auto|eauto|omega|lia|nia|easy|done)\b"
)


def _looks_like_rocq_script(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{"):
        return False
    return any(_TACTIC_START_RE.match(line) for line in stripped.splitlines())


def _after_last_marker(text: str) -> str:
    markers = [
        "final script:",
        "final rocq script:",
        "rocq script:",
        "proof script:",
        "script:",
    ]
    lowered = text.lower()
    best = -1
    marker_len = 0
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx > best:
            best = idx
            marker_len = len(marker)
    if best >= 0:
        return text[best + marker_len :].strip()
    return text


def _filter_probable_script_lines(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    in_comment = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept:
                kept.append(line)
            continue
        if stripped.startswith("```"):
            continue
        if stripped.startswith("(*"):
            in_comment = True
            kept.append(line)
            if "*)" in stripped:
                in_comment = False
            continue
        if in_comment:
            kept.append(line)
            if "*)" in stripped:
                in_comment = False
            continue
        if re.match(r"^(Lemma|Theorem|Fact|Proposition|Corollary|Proof|Qed|Defined|Admitted|Abort)\b", stripped):
            continue
        if _TACTIC_START_RE.match(line) or stripped in {"-", "+", "*", "{", "}"}:
            kept.append(line)
            continue
        if kept and not kept[-1].rstrip().endswith("."):
            kept.append(line)
    return textwrap.dedent("\n".join(kept)).strip()


def extract_rocq_script(text: str) -> str:
    """Extract tactic commands from an LLM response."""

    raw = _after_last_marker(str(text).strip())
    raw_had_fence = bool(_fenced_blocks(raw))
    script = _strip_code_fences(raw)
    if "Proof." in script:
        script = script.split("Proof.", 1)[1]
    for marker in ("Qed.", "Defined.", "Admitted.", "Abort."):
        if marker in script:
            script = script.split(marker, 1)[0]
    lines: list[str] = []
    skipping_statement = False
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if re.match(r"^(Lemma|Theorem|Fact|Proposition|Corollary)\b", stripped):
            skipping_statement = True
            continue
        if skipping_statement:
            if stripped.endswith("."):
                skipping_statement = False
            continue
        lines.append(line)
    script = textwrap.dedent("\n".join(lines)).strip()
    if not raw_had_fence or not _looks_like_rocq_script(script):
        script = _filter_probable_script_lines(script) or _filter_probable_script_lines(raw)
    return script


def split_rocq_commands(script: str) -> list[str]:
    """Split a Rocq proof script into vernacular/tactic sentences.

    The splitter is intentionally small, but it handles the common workshop
    scripts: comments, strings, bullets, and period-terminated commands.
    """

    script = extract_rocq_script(script)
    commands: list[str] = []
    current: list[str] = []
    comment_depth = 0
    in_string = False
    escaped = False
    i = 0
    while i < len(script):
        ch = script[i]
        nxt = script[i + 1] if i + 1 < len(script) else ""
        current.append(ch)

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if comment_depth:
            if ch == "(" and nxt == "*":
                current.append(nxt)
                comment_depth += 1
                i += 2
                continue
            if ch == "*" and nxt == ")":
                current.append(nxt)
                comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if ch == '"':
            in_string = True
        elif ch == "(" and nxt == "*":
            current.append(nxt)
            comment_depth += 1
            i += 2
            continue
        elif ch == ".":
            command = "".join(current).strip()
            if command:
                commands.append(command)
            current = []
        i += 1

    tail = "".join(current).strip()
    if tail:
        commands.append(tail)
    return commands


def _compact_goal_text(goals: list[str]) -> str:
    return "\n\n".join(goals) if goals else "(no focused goals)"


def _shorten(text: object, *, limit: int = 500) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(limit - 3, 0)].rstrip() + "..."


def _format_event(event: dict[str, Any]) -> str:
    kind = event.get("kind", "event")
    if kind == "llm_request":
        return f"[llm] requesting {event.get('mode', 'proof')}..."
    if kind == "llm_response":
        usage = event.get("usage")
        if isinstance(usage, LLMUsage):
            usage_text = " | " + usage.summary()
        elif isinstance(usage, dict):
            usage_text = " | " + LLMUsage.from_dict(usage).summary()
        else:
            usage_text = ""
        queue_text = ""
        queue = event.get("queue")
        if isinstance(queue, dict):
            queue_text = (
                f" | queue wait={float(queue.get('queued_wait_s') or 0.0):.1f}s"
                f", attempts={int(queue.get('attempts') or 0)}"
                f", retries={int(queue.get('retry_count') or 0)}"
            )
        return f"[llm] response received ({event.get('chars', 0)} chars){usage_text}{queue_text}"
    if kind == "llm_error":
        return f"[llm:error] {_shorten(event.get('error'), limit=900)}"
    if kind == "llm_queue":
        status = event.get("status", "queued")
        queue = event.get("queue") if isinstance(event.get("queue"), dict) else {}
        parts = [f"[llm:queue] {status}"]
        if event.get("queue_position"):
            parts.append(f"position={event.get('queue_position')}")
        if queue:
            parts.append(f"queued={queue.get('queued', 0)}")
            parts.append(f"running={queue.get('running', 0)}")
            backoff = float(queue.get("backoff_remaining_s") or 0.0)
            if backoff > 0:
                parts.append(f"backoff={backoff:.1f}s")
        if event.get("attempts"):
            parts.append(f"attempts={event.get('attempts')}")
        if event.get("retry_count"):
            parts.append(f"retries={event.get('retry_count')}")
        next_retry_at = event.get("next_retry_at")
        if next_retry_at:
            try:
                retry_in = max(0.0, float(next_retry_at) - time.time())
                parts.append(f"retry_in={retry_in:.1f}s")
            except (TypeError, ValueError):
                pass
        if event.get("last_error") and status in {"retrying", "failed"}:
            parts.append("last_error=" + _shorten(event.get("last_error"), limit=300))
        return " | ".join(parts)
    if kind == "script":
        return "[script]\n" + str(event.get("script", "")).strip()
    if kind == "tactic_start":
        return f"[rocq] {event.get('tactic')}"
    if kind == "tactic_result":
        status = "ok" if event.get("ok") else "failed"
        suffix = f", goals={event.get('goal_count')}" if event.get("goal_count") is not None else ""
        msg = f"[rocq] {status}{suffix}"
        if event.get("error"):
            msg += f"\n[rocq:error] {_shorten(event.get('error'), limit=700)}"
        return msg
    if kind == "qed":
        return "[rocq] Qed " + ("ok" if event.get("ok") else "failed")
    if kind == "tool_action":
        return f"[tool] {event.get('action')}: {_shorten(event.get('payload'), limit=700)}"
    if kind == "tool_result":
        return f"[tool] result: {_shorten(event.get('result'), limit=900)}"
    if kind == "finish_failed":
        return f"[agent] finish failed: {_shorten(event.get('error'), limit=900)}"
    return f"[{kind}] {_shorten(event, limit=900)}"


def _emit(
    on_event: Callable[[dict[str, Any]], None] | None,
    verbose: bool,
    **event: Any,
) -> None:
    if on_event is not None:
        on_event(event)
    if verbose:
        print(_format_event(event), flush=True)


def proof_prompt(
    *,
    lemma_header: str,
    goals: list[str],
    retrieval_hits: list[dict[str, Any]],
    extra_context: str = "",
) -> str:
    hits = format_retrieval_hits(
        retrieval_hits,
        statement_chars=1600,
        docstring_chars=1600,
    )
    return textwrap.dedent(
        f"""
        We are proving this Rocq statement:

        {lemma_header}

        Current goals:

        {_compact_goal_text(goals)}

        Retrieved/local context:

        {hits}

        Extra context:

        {extra_context or "(none)"}

        If a partial proof script is shown in the extra context, continue from
        the displayed current goals. Do not restart the proof.

        Return a compact Rocq tactic script. The response must contain actual
        Rocq commands, not only reasoning. Do not include prose, markdown,
        `Proof.`, or `Qed.`.
        """
    ).strip()


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text" and getattr(item, "text", None) is not None:
                parts.append(str(getattr(item, "text")))
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _with_partial_proof(theorem: Any, extra_context: str) -> str:
    try:
        proof = theorem.proof_script()
    except Exception:
        proof = ""
    if not proof.strip():
        return extra_context
    block = textwrap.dedent(
        f"""
        Current partial proof script already executed:

        ```coq
        {proof}
        ```
        """
    ).strip()
    return (extra_context.strip() + "\n\n" + block).strip() if extra_context else block


def _local_document_context(theorem: Any, *, limit: int = 16000) -> str:
    document = getattr(theorem, "document", None)
    if document is None:
        return "(unavailable)"
    try:
        source = document.source(include_open=False)
    except TypeError:
        try:
            source = document.source()
        except Exception:
            return "(unavailable)"
    except Exception:
        return "(unavailable)"
    source = source.strip()
    if not source:
        return "(empty)"
    if len(source) <= limit:
        return source
    return source[-limit:]


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _is_transient_llm_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "connection",
            "connecterror",
            "readerror",
            "remoteprotocolerror",
        )
    )


def _should_rollback_goal_increase(tactic: str) -> bool:
    """Return whether a goal-count increase is likely an accidental side goal."""
    cleaned = tactic.strip().lower()
    structural_prefixes = (
        "apply ",
        "eapply ",
        "refine ",
        "split",
        "repeat split",
        "constructor",
        "destruct ",
        "induction ",
        "exists ",
        "left",
        "right",
    )
    if cleaned.startswith(structural_prefixes):
        return False
    dangerous_prefixes = (
        "replace ",
        "assert ",
        "cut ",
        "field",
        "change ",
        "rewrite ",
    )
    if cleaned.startswith(dangerous_prefixes):
        return True
    return False


def _should_rollback_no_progress(tactic: str) -> bool:
    cleaned = tactic.strip().lower()
    no_progress_prefixes = (
        "unfold ",
        "fold ",
        "simpl",
        "cbn",
        "cbv",
        "rewrite ",
        "change ",
    )
    return cleaned.startswith(no_progress_prefixes)


def _should_rollback_semantic_unfold(tactic: str) -> bool:
    cleaned = tactic.strip().lower()
    if not cleaned.startswith("unfold "):
        return False
    semantic_names = (
        "ex_derive",
        "is_derive",
        "continuous",
        "is_rint",
        "rint",
    )
    return any(re.search(rf"\b{name}\b", cleaned) for name in semantic_names)


def _should_rollback_complexity_increase(
    tactic: str,
    before_goals: list[str],
    after_goals: list[str],
) -> bool:
    cleaned = tactic.strip().lower()
    if not cleaned.startswith("change "):
        return False
    before_size = sum(len(goal) for goal in before_goals)
    after_size = sum(len(goal) for goal in after_goals)
    return after_size > before_size + max(120, before_size // 4)


def _should_rollback_cycle(tactic: str) -> bool:
    cleaned = tactic.strip().lower()
    cycle_prefixes = (
        "replace ",
        "change ",
        "rewrite ",
        "unfold ",
        "fold ",
        "simpl",
        "cbn",
        "cbv",
    )
    return cleaned.startswith(cycle_prefixes)


def _normalized_replace_side(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def _replace_sides(tactic: str) -> tuple[str, str] | None:
    cleaned = tactic.strip().rstrip(".")
    match = re.match(r"replace\s+\((.*)\)\s+with\s+\((.*)\)\s+by\b", cleaned, flags=re.S)
    if not match:
        match = re.match(r"replace\s+\((.*)\)\s+with\s+(.*?)\s+by\b", cleaned, flags=re.S)
    if not match:
        return None
    lhs = _normalized_replace_side(match.group(1))
    rhs = _normalized_replace_side(match.group(2))
    if not lhs or not rhs:
        return None
    return lhs, rhs


def _tactic_failure_guidance(tactic: str, out: dict[str, Any]) -> str:
    cleaned = tactic.strip().lower()
    error = str(out.get("error", ""))
    parts = [f"run_tac({tactic!r}) failed and was rolled back -> {out}"]
    if cleaned.startswith("all:"):
        parts.append(
            "Guidance: `all:` sends the same tactic to every remaining goal. "
            "If even one remaining goal has a different shape, avoid `all:` "
            "and solve the focused goal first with the tactic that matches it."
        )
    if cleaned.startswith("replace "):
        parts.append(
            "Guidance: `replace A with B by proof` asks Rocq to prove the "
            "exact equality generated by that replacement. If the selected "
            "lemma rewrites a subterm of the main goal, try rewriting the main "
            "goal directly instead of proving a separate replacement equality."
        )
    if "not a valid field equation" in error.lower():
        parts.append(
            "Guidance: algebra tactics for rational expressions usually need "
            "the transcendental subterms to have matching syntactic forms first."
        )
    if "not a valid ring equation" in error.lower():
        parts.append(
            "Guidance: polynomial tactics prove polynomial equalities, not "
            "equalities whose outer symbols are opaque functions."
        )
    if cleaned.startswith(("apply ", "eapply ")) and "unable to unify" in error.lower():
        parts.append(
            "Guidance: an `apply` command only works when the conclusion of the "
            "selected fact matches the current goal after unification. If the "
            "goal is now an algebraic side condition, use an algebraic or "
            "arithmetic argument instead of forcing an analytic lemma."
        )
        parts.append(
            "Guidance: for a binary structural lemma on a printed chain such as "
            "`A + B + C`, the first split should follow the syntax tree: left "
            "component `A + B`, right component `C`."
        )
    return "\n".join(parts)


def _script_failure_guidance(script: str, result: ProofResult) -> str:
    error = str(result.error or "")
    lowered = error.lower()
    parts: list[str] = []
    if "unable to unify" in lowered:
        parts.append(
            "Guidance: the script was rolled back at a unification failure. "
            "Do not assume later commands in that script were executed. Match "
            "the conclusion of the next structural lemma to the focused goal's "
            "outer syntax exactly. For a chained binary expression, respect the "
            "printed grouping instead of using an algebraically equivalent "
            "regrouping. For a displayed `A + B + C`, the top-level split is "
            "left component `A + B` and right component `C`."
        )
        if "apply " in script.lower():
            parts.append(
                "Guidance: try the corrected `apply` as a single `run_tac` "
                "before sending another multi-command `finish` script."
            )
    return "\n".join(parts)


def _tactic_success_guidance(tactic: str, out: dict[str, Any]) -> str:
    cleaned = tactic.strip().lower()
    goal_count = out.get("goal_count")
    if goal_count in (None, 0):
        return ""
    if cleaned.startswith(("apply ", "eapply ")):
        return (
            "Guidance: this structural step created subgoals. Work on the "
            "focused subgoal first. If it exactly matches a selected or local "
            "lemma, apply that lemma. If it has the same binary structure as "
            "the previous goal, apply the same structural rule to that focused "
            "subgoal. If the focused subgoal is an inequality, use selected "
            "positivity/order facts whose conclusion matches its structure, "
            "then finish numeric arithmetic side goals."
        )
    if cleaned.startswith("auto_derive"):
        return (
            "Guidance: automation generated remaining obligations. Solve side "
            "conditions with matching selected/local lemmas. If a remaining "
            "goal is about an opaque local helper function, unfold that helper "
            "definition before rerunning automation."
        )
    if cleaned.startswith(("split", "repeat split", "constructor")):
        return (
            "Guidance: this split created several goals. Work on the focused "
            "goal first. Use `all:` only when every remaining goal visibly has "
            "the same shape; otherwise solve the repeated side conditions one "
            "by one and leave any different algebraic or semantic goal for a "
            "matching tactic."
        )
    if cleaned.startswith(("replace ", "change ")):
        return (
            "Guidance: if this transformation was meant to expose the left-hand "
            "pattern of a selected rewrite lemma, try that rewrite next before "
            "doing further arithmetic normalization."
        )
    if cleaned.startswith("field"):
        return (
            "Guidance: `field` transformed the main equality and left side "
            "conditions. If the remaining goals are polynomial or nonzero real "
            "arithmetic conditions over opaque subterms, use an arithmetic "
            "tactic rather than an analytic lemma whose conclusion does not "
            "match syntactically."
        )
    return ""


def _try_safe_cleanup(
    theorem: Any,
    *,
    on_event: Callable[[dict[str, Any]], None] | None,
    verbose: bool,
    step: int,
) -> list[dict[str, Any]]:
    """Try conservative arithmetic cleanups and keep only real progress."""
    outputs: list[dict[str, Any]] = []
    cleanup_tactics = ("lra.", "nra.", "field; nra.")
    made_progress = True
    rounds = 0
    while made_progress and rounds < 8:
        rounds += 1
        made_progress = False
        try:
            before_goals = theorem.goals()
        except Exception:
            return outputs
        before_count = len(before_goals)
        if before_count == 0:
            return outputs
        for tactic in cleanup_tactics:
            checkpoint_name = f"_agent_cleanup_{step}_{rounds}_{len(outputs)}"
            try:
                theorem.checkpoint(checkpoint_name)
            except Exception:
                checkpoint_name = ""
            out = theorem.run_tac(tactic)
            after_count = int(out.get("goal_count") if out.get("goal_count") is not None else before_count)
            if out.get("ok", False) and after_count < before_count:
                outputs.append(out)
                _emit(on_event, verbose, kind="tactic_start", tactic=f"[cleanup] {tactic}")
                _emit(
                    on_event,
                    verbose,
                    kind="tactic_result",
                    tactic=f"[cleanup] {tactic}",
                    ok=True,
                    goal_count=after_count,
                    error="",
                )
                made_progress = True
                break
            if checkpoint_name:
                try:
                    theorem.reverse(checkpoint_name)
                except Exception:
                    pass
    return outputs


def _maybe_unset(value: Any) -> Any:
    if value is None:
        return None
    if value.__class__.__name__.lower() in {"unset", "unsettype"}:
        return None
    return value


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    value = _maybe_unset(value)
    if value is None:
        return default
    if isinstance(value, dict):
        return _maybe_unset(value.get(name, default))
    return _maybe_unset(getattr(value, name, default))


def _as_int(value: Any) -> int:
    value = _maybe_unset(value)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _plain_data(value: Any) -> Any:
    value = _maybe_unset(value)
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_data(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain_data(model_dump(exclude_none=True))
    return str(value)


def _price_for_model(model: str) -> tuple[float, float, float]:
    normalized = (model or DEFAULT_MISTRAL_MODEL).strip()
    return MODEL_PRICES_USD_PER_MILLION.get(
        normalized,
        MODEL_PRICES_USD_PER_MILLION[DEFAULT_MISTRAL_MODEL],
    )


@dataclass
class LLMUsage:
    model: str = DEFAULT_MISTRAL_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0
    uncached_input_tokens: int = 0
    input_cost_usd: float = 0.0
    cached_input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    @classmethod
    def from_counts(
        cls,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cache_tokens: int = 0,
    ) -> "LLMUsage":
        input_tokens = max(int(input_tokens or 0), 0)
        output_tokens = max(int(output_tokens or 0), 0)
        total_tokens = max(int(total_tokens or input_tokens + output_tokens), 0)
        cache_tokens = min(max(int(cache_tokens or 0), 0), input_tokens)
        uncached_input_tokens = input_tokens - cache_tokens
        input_rate, cached_input_rate, output_rate = _price_for_model(model)
        input_cost_usd = uncached_input_tokens * input_rate / 1_000_000
        cached_input_cost_usd = cache_tokens * cached_input_rate / 1_000_000
        output_cost_usd = output_tokens * output_rate / 1_000_000
        return cls(
            model=model or DEFAULT_MISTRAL_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_tokens=cache_tokens,
            uncached_input_tokens=uncached_input_tokens,
            input_cost_usd=input_cost_usd,
            cached_input_cost_usd=cached_input_cost_usd,
            output_cost_usd=output_cost_usd,
            total_cost_usd=input_cost_usd + cached_input_cost_usd + output_cost_usd,
        )

    @classmethod
    def from_provider_usage(cls, usage: Any, *, model: str) -> "LLMUsage":
        prompt_tokens = _as_int(_get_field(usage, "prompt_tokens"))
        completion_tokens = _as_int(_get_field(usage, "completion_tokens"))
        total_tokens = _as_int(_get_field(usage, "total_tokens"))
        prompt_details = _get_field(usage, "prompt_tokens_details")
        cache_tokens = _as_int(_get_field(prompt_details, "cached_tokens"))
        if not cache_tokens:
            cache_tokens = _as_int(_get_field(usage, "num_cached_tokens"))
        return cls.from_counts(
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_tokens=cache_tokens,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMUsage":
        data = data or {}
        usage = cls.from_counts(
            model=str(data.get("model") or DEFAULT_MISTRAL_MODEL),
            input_tokens=_as_int(data.get("input_tokens", data.get("prompt_tokens"))),
            output_tokens=_as_int(data.get("output_tokens", data.get("completion_tokens"))),
            total_tokens=_as_int(data.get("total_tokens")),
            cache_tokens=_as_int(
                data.get(
                    "cache_tokens",
                    data.get("cached_input_tokens", data.get("cached_tokens")),
                )
            ),
        )
        if data.get("total_cost_usd") is not None:
            try:
                usage.total_cost_usd = float(data["total_cost_usd"])
            except (TypeError, ValueError):
                pass
        return usage

    @classmethod
    def aggregate(cls, usages: list["LLMUsage"]) -> "LLMUsage":
        if not usages:
            return cls()
        model = usages[-1].model
        return cls.from_counts(
            model=model,
            input_tokens=sum(usage.input_tokens for usage in usages),
            output_tokens=sum(usage.output_tokens for usage in usages),
            total_tokens=sum(usage.total_tokens for usage in usages),
            cache_tokens=sum(usage.cache_tokens for usage in usages),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_tokens": self.cache_tokens,
            "cached_input_tokens": self.cache_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def summary(self, *, show_cache: bool = False) -> str:
        parts = [f"input={self.input_tokens}", f"output={self.output_tokens}"]
        if show_cache or self.cache_tokens:
            parts.append(f"cached_input={self.cache_tokens}")
        parts.append(f"cost=${self.total_cost_usd:.6f}")
        return ", ".join(parts)


@dataclass
class ChatResult:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw_usage: Any = None
    queue: dict[str, Any] | None = None


@dataclass
class ProofResult:
    ok: bool
    script: str = ""
    error: str = ""
    goals: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    raw_response: str = ""
    attempts: list[Any] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    usage_events: list[LLMUsage] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        status = "ok" if self.ok else "failed"
        parts = [f"ProofResult({status})"]
        if self.usage.input_tokens or self.usage.output_tokens:
            parts.append("usage: " + self.usage.summary())
        if self.script:
            parts.append("script:\n" + self.script)
        if self.error:
            parts.append("error:\n" + self.error)
        if self.goals:
            parts.append("remaining goals:\n" + "\n\n".join(self.goals))
        return "\n\n".join(parts)

    __repr__ = __str__

    def usage_summary(self) -> str:
        return self.usage.summary()


@dataclass
class LLMClient:
    """LLM helper used by the workshop notebook and proxy server."""

    model: str = DEFAULT_MISTRAL_MODEL
    provider: str = "mistral"
    model_explicit: bool = False
    api_key: str | None = None
    server_url: str | None = None
    server_token: str | None = None
    timeout: float = 300.0
    reasoning_effort: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_retries: int = 2
    force_ipv4: bool = True
    prompt_cache_key: str | None = "integral-tp-workshop"
    openrouter_base_url: str = OPENROUTER_BASE_URL
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None

    @classmethod
    def from_env(cls, *, model: str | None = None) -> "LLMClient":
        server_url = os.getenv("WORKSHOP_LLM_SERVER_URL") or os.getenv("LLM_SERVER_URL")
        return cls._from_env(model=model, server_url=server_url)

    @classmethod
    def direct_from_env(cls, *, model: str | None = None) -> "LLMClient":
        return cls._from_env(model=model, server_url=None)

    @classmethod
    def _from_env(cls, *, model: str | None, server_url: str | None) -> "LLMClient":
        provider = (
            os.getenv("WORKSHOP_LLM_PROVIDER")
            or os.getenv("LLM_PROVIDER")
            or ("openrouter" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("MISTRAL_API_KEY") else "mistral")
        ).strip().lower()
        temperature = (
            os.getenv("MISTRAL_TEMPERATURE")
            or os.getenv("LLM_TEMPERATURE")
        )
        top_p = (
            os.getenv("MISTRAL_TOP_P")
            or os.getenv("LLM_TOP_P")
        )
        max_retries = os.getenv("MISTRAL_MAX_RETRIES") or os.getenv("LLM_MAX_RETRIES")
        force_ipv4 = os.getenv("MISTRAL_FORCE_IPV4") or os.getenv("LLM_FORCE_IPV4")
        timeout = os.getenv("MISTRAL_TIMEOUT") or os.getenv("LLM_TIMEOUT")
        prompt_cache_key = (
            (os.getenv("OPENROUTER_PROMPT_CACHE_KEY") or os.getenv("LLM_PROMPT_CACHE_KEY"))
            if provider == "openrouter"
            else (
                os.getenv("MISTRAL_PROMPT_CACHE_KEY")
                or os.getenv("LLM_PROMPT_CACHE_KEY")
                or "integral-tp-workshop"
            )
        )
        model_from_env = (
            model
            or (
                os.getenv("OPENROUTER_MODEL")
                if provider == "openrouter"
                else os.getenv("MISTRAL_MODEL")
            )
            or os.getenv("LLM_MODEL")
        )
        default_model = DEFAULT_OPENROUTER_MODEL if provider == "openrouter" else DEFAULT_MISTRAL_MODEL
        api_key = (
            os.getenv("OPENROUTER_API_KEY")
            if provider == "openrouter"
            else os.getenv("MISTRAL_API_KEY")
        ) or os.getenv("LLM_API_KEY")
        return cls(
            model=model_from_env or default_model,
            provider=provider,
            model_explicit=model_from_env is not None,
            api_key=api_key,
            server_url=server_url,
            server_token=os.getenv("WORKSHOP_LLM_SERVER_TOKEN") or os.getenv("LLM_SERVER_TOKEN"),
            reasoning_effort=(
                os.getenv("OPENROUTER_REASONING_EFFORT")
                if provider == "openrouter"
                else os.getenv("MISTRAL_REASONING_EFFORT")
            )
            or os.getenv("LLM_REASONING_EFFORT")
            or None,
            temperature=float(temperature) if temperature is not None else 0.7,
            top_p=float(top_p) if top_p is not None else 1.0,
            max_retries=int(max_retries) if max_retries is not None else 2,
            force_ipv4=_env_bool(force_ipv4, default=True),
            timeout=float(timeout) if timeout is not None else 300.0,
            prompt_cache_key=prompt_cache_key or None,
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL,
            openrouter_site_url=os.getenv("OPENROUTER_SITE_URL") or None,
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME") or "integral-tp",
        )

    @property
    def configured(self) -> bool:
        return bool(self.server_url or self.api_key)

    def _client(self) -> Any:
        if Mistral is None:
            raise RuntimeError(
                "Could not import the Mistral SDK. Install `mistralai`."
            ) from _MISTRAL_IMPORT_ERROR
        if not self.api_key:
            raise RuntimeError(
                "No Mistral API key configured. Set MISTRAL_API_KEY or LLM_API_KEY."
            )
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout_ms": int(self.timeout * 1000),
        }
        if self.force_ipv4:
            kwargs["client"] = httpx.Client(
                timeout=self.timeout,
                transport=httpx.HTTPTransport(
                    local_address="0.0.0.0",
                    retries=self.max_retries,
                ),
            )
        return Mistral(**kwargs)

    def _openrouter_client(self) -> Any:
        if OpenAI is None:
            raise RuntimeError(
                "Could not import the OpenAI SDK. Install `openai`."
            ) from _OPENAI_IMPORT_ERROR
        if not self.api_key:
            raise RuntimeError(
                "No OpenRouter API key configured. Set OPENROUTER_API_KEY or LLM_API_KEY."
            )
        headers: dict[str, str] = {}
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-OpenRouter-Title"] = self.openrouter_app_name
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.openrouter_base_url,
            "timeout": self.timeout,
        }
        if headers:
            kwargs["default_headers"] = headers
        if self.force_ipv4:
            kwargs["http_client"] = httpx.Client(
                timeout=self.timeout,
                transport=httpx.HTTPTransport(
                    local_address="0.0.0.0",
                    retries=self.max_retries,
                ),
            )
        return OpenAI(**kwargs)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> str:
        return self.chat_with_usage(
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ).text

    def chat_with_usage(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = False,
    ) -> ChatResult:
        if self.server_url:
            return self._chat_with_server(
                system=system,
                user=user,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                on_event=on_event,
                verbose=verbose,
            )
        if self.provider == "openrouter":
            return self._chat_with_openrouter(
                system=system,
                user=user,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        return self._chat_with_mistral(
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def _chat_kwargs(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        effective_temperature = self.temperature if temperature is None else temperature
        effective_top_p = self.top_p if top_p is None else top_p
        if effective_temperature == 0 and effective_top_p != 1:
            effective_top_p = 1
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": effective_temperature,
            "top_p": effective_top_p,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.prompt_cache_key:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
        kwargs["timeout_ms"] = int(self.timeout * 1000)
        return kwargs

    def _openai_chat_kwargs(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs = self._chat_kwargs(
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        kwargs.pop("timeout_ms", None)
        kwargs["extra_body"] = {}
        if self.prompt_cache_key:
            kwargs["extra_body"]["prompt_cache_key"] = self.prompt_cache_key
        if self.reasoning_effort:
            kwargs["extra_body"]["reasoning_effort"] = self.reasoning_effort
        if not kwargs["extra_body"]:
            kwargs.pop("extra_body")
        return kwargs

    def _chat_with_mistral(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
    ) -> ChatResult:
        kwargs = self._chat_kwargs(
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        last_error: Exception | None = None
        for attempt in range(max(self.max_retries, 0) + 1):
            client = self._client()
            supplied_http_client = getattr(client, "client", None)
            try:
                response = client.chat.complete(**kwargs)
                model = str(getattr(response, "model", None) or self.model)
                usage = LLMUsage.from_provider_usage(getattr(response, "usage", None), model=model)
                return ChatResult(
                    text=_content_to_text(response.choices[0].message.content),
                    usage=usage,
                    raw_usage=_plain_data(getattr(response, "usage", None)),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_transient_llm_error(exc):
                    raise
                time.sleep(min(8.0, 0.75 * (2**attempt)) + random.uniform(0.0, 0.25))
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                if supplied_http_client is not None:
                    supplied_http_client.close()
        assert last_error is not None
        raise last_error

    def _chat_with_openrouter(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
    ) -> ChatResult:
        kwargs = self._openai_chat_kwargs(
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        last_error: Exception | None = None
        for attempt in range(max(self.max_retries, 0) + 1):
            client = self._openrouter_client()
            try:
                response = client.chat.completions.create(**kwargs)
                model = str(getattr(response, "model", None) or self.model)
                usage = LLMUsage.from_provider_usage(getattr(response, "usage", None), model=model)
                return ChatResult(
                    text=_content_to_text(response.choices[0].message.content),
                    usage=usage,
                    raw_usage=_plain_data(getattr(response, "usage", None)),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_transient_llm_error(exc):
                    raise
                time.sleep(min(8.0, 0.75 * (2**attempt)) + random.uniform(0.0, 0.25))
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        assert last_error is not None
        raise last_error

    def _chat_with_server(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = False,
    ) -> ChatResult:
        assert self.server_url is not None
        payload = {
            "model": self.model if self.model_explicit else None,
            "system": system,
            "user": user,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "prompt_cache_key": self.prompt_cache_key,
        }
        headers: dict[str, str] = {}
        if self.server_token:
            headers["Authorization"] = f"Bearer {self.server_token}"
        if _env_bool(os.getenv("WORKSHOP_LLM_SERVER_USE_JOBS"), default=True):
            try:
                return self._chat_with_server_jobs(
                    payload,
                    headers=headers,
                    on_event=on_event,
                    verbose=verbose,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {404, 405}:
                    raise
        return self._chat_with_server_blocking(payload, headers=headers)

    def _chat_with_server_blocking(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
    ) -> ChatResult:
        assert self.server_url is not None
        url = urljoin(self.server_url.rstrip("/") + "/", "chat")
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code >= 400:
                message = (
                    f"Server error '{response.status_code}' for url '{url}': "
                    f"{_shorten(response.text, limit=1000)}"
                )
                raise httpx.HTTPStatusError(message, request=response.request, response=response)
            response.raise_for_status()
            data = response.json()
        return ChatResult(
            text=str(data.get("text", "")),
            usage=LLMUsage.from_dict(data.get("usage")),
            raw_usage=data.get("raw_usage"),
            queue=data.get("queue"),
        )

    def _chat_with_server_jobs(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        on_event: Callable[[dict[str, Any]], None] | None,
        verbose: bool,
    ) -> ChatResult:
        assert self.server_url is not None
        root = self.server_url.rstrip("/") + "/"
        jobs_url = urljoin(root, "jobs")
        poll_s = _env_float(os.getenv("WORKSHOP_LLM_SERVER_POLL_SECONDS"), default=2.0)
        poll_s = max(0.2, poll_s)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(jobs_url, json=payload, headers=headers)
            if response.status_code >= 400:
                message = (
                    f"Server error '{response.status_code}' for url '{jobs_url}': "
                    f"{_shorten(response.text, limit=1000)}"
                )
                raise httpx.HTTPStatusError(message, request=response.request, response=response)
            data = response.json()
            job_id = str(data.get("job_id") or "")
            if not job_id:
                raise RuntimeError("LLM proxy did not return a job id.")
            status_url = urljoin(root, f"jobs/{job_id}")
            last_reported: tuple[Any, ...] | None = None
            while True:
                status_response = client.get(status_url, headers=headers)
                if status_response.status_code >= 400:
                    message = (
                        f"Server error '{status_response.status_code}' for url '{status_url}': "
                        f"{_shorten(status_response.text, limit=1000)}"
                    )
                    raise httpx.HTTPStatusError(
                        message,
                        request=status_response.request,
                        response=status_response,
                    )
                status_data = status_response.json()
                report_key = (
                    status_data.get("status"),
                    status_data.get("queue_position"),
                    status_data.get("attempts"),
                    status_data.get("retry_count"),
                    status_data.get("next_retry_at"),
                )
                if report_key != last_reported:
                    _emit(
                        on_event,
                        verbose,
                        kind="llm_queue",
                        status=status_data.get("status"),
                        queue_position=status_data.get("queue_position"),
                        attempts=status_data.get("attempts"),
                        retry_count=status_data.get("retry_count"),
                        next_retry_at=status_data.get("next_retry_at"),
                        last_error=status_data.get("last_error"),
                        queue=status_data.get("queue"),
                    )
                    last_reported = report_key

                status = status_data.get("status")
                if status == "succeeded":
                    finished_at = float(status_data.get("finished_at") or time.time())
                    started_at = float(status_data.get("started_at") or finished_at)
                    created_at = float(status_data.get("created_at") or started_at)
                    queue_meta = {
                        "job_id": job_id,
                        "status": status,
                        "attempts": int(status_data.get("attempts") or 0),
                        "retry_count": int(status_data.get("retry_count") or 0),
                        "queued_wait_s": max(0.0, started_at - created_at),
                        "total_elapsed_s": max(0.0, finished_at - created_at),
                    }
                    if isinstance(status_data.get("queue"), dict):
                        queue_meta.update(status_data["queue"])
                    return ChatResult(
                        text=str(status_data.get("text") or ""),
                        usage=LLMUsage.from_dict(status_data.get("usage")),
                        raw_usage=status_data.get("raw_usage", status_data.get("usage")),
                        queue=queue_meta,
                    )
                if status == "failed":
                    raise RuntimeError(
                        "LLM proxy job failed: "
                        + _shorten(status_data.get("last_error"), limit=1000)
                    )
                time.sleep(poll_s)

    def formalized(
        self,
        message: str,
        *,
        doc: Any,
        selected_hits: list[dict[str, Any]] | None = None,
    ) -> str:
        selected_hits = selected_hits or []
        try:
            source = doc.source(include_open=False)
        except TypeError:
            source = doc.source()
        hits = format_retrieval_hits(selected_hits, statement_chars=1200, docstring_chars=800)
        prompt = textwrap.dedent(
            f"""
            Current Rocq document:

            {source}

            Retrieved context:

            {hits}

            Informal mathematical statement:

            {message}

            Return exactly one Rocq Lemma/Theorem statement ending with a period.
            Do not include `Proof.` or `Qed.`.
            """
        ).strip()
        return _strip_code_fences(
            self.chat(system=ROCQ_SYSTEM_PROMPT, user=prompt, max_tokens=800)
        )

    def prove(
        self,
        theorem: Any,
        *,
        selected_hits: list[dict[str, Any]] | None = None,
        extra_context: str = "",
        tools: dict[str, Callable[..., Any]] | None = None,
        max_tool_calls: int = 8,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        verbose: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        close: bool = True,
    ) -> ProofResult:
        selected_hits = selected_hits or []
        if tools:
            return self._prove_with_tools(
                theorem,
                selected_hits=selected_hits,
                extra_context=extra_context,
                tools=tools,
                max_tool_calls=max_tool_calls,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose,
                on_event=on_event,
                close=close,
            )
        prompt = proof_prompt(
            lemma_header=theorem.header,
            goals=theorem.goals(),
            retrieval_hits=selected_hits,
            extra_context=_with_partial_proof(theorem, extra_context),
        )
        _emit(on_event, verbose, kind="llm_request", mode="proof", theorem=getattr(theorem, "name", ""))
        try:
            chat_result = self.chat_with_usage(
                system=ROCQ_SYSTEM_PROMPT,
                user=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                on_event=on_event,
                verbose=verbose,
            )
            raw = chat_result.text
        except Exception as exc:
            error = str(exc)
            _emit(on_event, verbose, kind="llm_error", mode="proof", error=error)
            return ProofResult(ok=False, error=f"LLM call failed: {error}", goals=theorem.goals())
        _emit(
            on_event,
            verbose,
            kind="llm_response",
            mode="proof",
            chars=len(raw),
            usage=chat_result.usage,
            queue=chat_result.queue,
        )
        script = extract_rocq_script(raw)
        _emit(on_event, verbose, kind="script", script=script)
        result = self._apply_script(
            theorem,
            script,
            verbose=verbose,
            on_event=on_event,
            close=close,
        )
        result.raw_response = raw
        result.usage = chat_result.usage
        result.usage_events = [chat_result.usage]
        return result

    def _apply_script(
        self,
        theorem: Any,
        script: str,
        *,
        verbose: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        close: bool = True,
    ) -> ProofResult:
        commands = split_rocq_commands(script)
        if not commands:
            return ProofResult(ok=False, script=script, error="LLM returned no Rocq commands.")

        outputs: list[dict[str, Any]] = []
        for command in commands:
            _emit(on_event, verbose, kind="tactic_start", tactic=command)
            out = theorem.run_tac(command)
            outputs.append(out)
            _emit(
                on_event,
                verbose,
                kind="tactic_result",
                tactic=command,
                ok=out.get("ok", False),
                goal_count=out.get("goal_count"),
                error=out.get("error", ""),
            )
            if not out.get("ok", False):
                return ProofResult(
                    ok=False,
                    script=script,
                    error=out.get("error", "Rocq rejected a command."),
                    goals=out.get("goals", theorem.goals()),
                    commands=commands,
                    outputs=outputs,
                )
            cleanup_outputs = _try_safe_cleanup(
                theorem,
                on_event=on_event,
                verbose=verbose,
                step=len(outputs),
            )
            outputs.extend(cleanup_outputs)
            if out.get("proof_finished", False) or out.get("goal_count") == 0:
                break
            if not theorem.goals():
                break

        goals = theorem.goals()
        if goals:
            return ProofResult(
                ok=False,
                script=script,
                error="Proof script ended with remaining goals.",
                goals=goals,
                commands=commands,
                outputs=outputs,
            )

        if not close:
            return ProofResult(
                ok=True,
                script=script,
                goals=[],
                commands=commands,
                outputs=outputs,
                completed=False,
            )

        qed = theorem.qed()
        _emit(
            on_event,
            verbose,
            kind="qed",
            ok=qed.get("ok", False),
            error=qed.get("error", ""),
        )
        if not qed.get("ok", False):
            return ProofResult(
                ok=False,
                script=script,
                error=qed.get("error", "Rocq rejected Qed."),
                goals=qed.get("goals", theorem.goals()),
                commands=commands,
                outputs=outputs + [qed],
            )
        return ProofResult(
            ok=True,
            script=script,
            goals=[],
            commands=commands,
            outputs=outputs + [qed],
            completed=True,
        )

    def _prove_with_tools(
        self,
        theorem: Any,
        *,
        selected_hits: list[dict[str, Any]],
        extra_context: str,
        tools: dict[str, Callable[..., Any]],
        max_tool_calls: int,
        temperature: float | None,
        max_tokens: int,
        verbose: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        close: bool = True,
    ) -> ProofResult:
        observations: list[str] = []
        attempts: list[Any] = []
        usage_events: list[LLMUsage] = []
        context_hits = list(selected_hits)
        failed_tactics: set[tuple[str, str]] = set()
        seen_goal_states: set[str] = set()
        replace_edges: set[tuple[str, str]] = set()
        for step in range(max_tool_calls):
            tool_prompt = textwrap.dedent(
                f"""
                You may either call a tool or finish with a proof script.

                Available tool actions:
                - {{"action": "run_tac", "tactic": "<one Rocq tactic command>"}}
                - {{"action": "run_tac", "tactic": "<one Rocq tactic command>", "allow_new_goals": true, "reason": "<why the new goals are expected>"}}
                - {{"action": "reverse", "checkpoint": "<checkpoint name or integer>"}}
                - {{"action": "finish", "script": "<remaining Rocq proof script>"}}

                Return exactly one JSON object. No markdown.

                Statement:
                {theorem.header}

                Local Rocq context already available:
                {_local_document_context(theorem)}

                Current goals:
                {_compact_goal_text(theorem.goals())}

                Retrieved/local context:
                {format_retrieval_hits(context_hits, statement_chars=1200, docstring_chars=1200)}

                Extra context:
                {extra_context or "(none)"}

                Tool observations so far:
                {chr(10).join(observations) or "(none)"}

                If a previous tactic or finish script failed, do not repeat the
                same failing command. Use the Rocq feedback and current goals to
                continue with a smaller next step. Prefer `run_tac` for one
                command when unsure; use `finish` only for a short script you
                expect Rocq to accept.

                When a selected fact is a binary structural rule, match the
                focused goal's printed top-level syntax. For a displayed chain
                `A + B + C`, the top-level split is usually the left component
                `A + B` and the right component `C`.

                The selected/retrieved context above is the complete context
                for this agent step; there is no search tool. If a
                transformation exposes the pattern of a selected rewrite fact,
                use that selected fact next instead of continuing unrelated
                arithmetic normalization.

                The runner checkpoints before every `run_tac`. If a tactic
                succeeds but creates extra side goals in a way that is usually
                accidental, the runner rolls it back and reports the old and new
                goal counts. For transformations such as replacing one
                expression by another, prefer a command that also justifies the
                transformation. Set `allow_new_goals: true` only when the new
                goals are the intended mathematical subgoals.
                """
            ).strip()
            _emit(
                on_event,
                verbose,
                kind="llm_request",
                mode="tool",
                step=step + 1,
                theorem=getattr(theorem, "name", ""),
            )
            try:
                chat_result = self.chat_with_usage(
                    system=ROCQ_SYSTEM_PROMPT,
                    user=tool_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    on_event=on_event,
                    verbose=verbose,
                )
                raw = chat_result.text
                usage_events.append(chat_result.usage)
            except Exception as exc:
                error = str(exc)
                _emit(
                    on_event,
                    verbose,
                    kind="llm_error",
                    mode="tool",
                    step=step + 1,
                    error=error,
                )
                return ProofResult(
                    ok=False,
                    script=theorem.proof_script(),
                    error=f"LLM call failed: {error}",
                    goals=theorem.goals(),
                    attempts=attempts,
                    usage=LLMUsage.aggregate(usage_events),
                    usage_events=list(usage_events),
                )
            _emit(
                on_event,
                verbose,
                kind="llm_response",
                mode="tool",
                step=step + 1,
                chars=len(raw),
                usage=chat_result.usage,
                queue=chat_result.queue,
            )
            action = _parse_json_action(raw)
            attempts.append({"raw": raw, "action": action})
            if not action:
                script = extract_rocq_script(raw)
                _emit(on_event, verbose, kind="script", script=script)
                result = self._apply_script(
                    theorem,
                    script,
                    verbose=verbose,
                    on_event=on_event,
                    close=close,
                )
                result.attempts = attempts
                result.usage_events = list(usage_events)
                result.usage = LLMUsage.aggregate(usage_events)
                return result

            name = str(action.get("action", "")).strip()
            _emit(on_event, verbose, kind="tool_action", action=name, payload=action)
            if name == "finish":
                checkpoint_name = f"_agent_finish_{step}"
                try:
                    theorem.checkpoint(checkpoint_name)
                except Exception:
                    checkpoint_name = ""
                _emit(on_event, verbose, kind="script", script=str(action.get("script", "")))
                result = self._apply_script(
                    theorem,
                    str(action.get("script", "")),
                    verbose=verbose,
                    on_event=on_event,
                    close=close,
                )
                result.attempts = attempts
                result.usage_events = list(usage_events)
                result.usage = LLMUsage.aggregate(usage_events)
                if result.ok:
                    try:
                        result.script = theorem.proof_script()
                    except Exception:
                        pass
                    return result
                _emit(on_event, verbose, kind="finish_failed", error=result.error)
                finish_guidance = _script_failure_guidance(
                    str(action.get("script", "")),
                    result,
                )
                observations.append(
                    "finish failed with Rocq feedback:\n"
                    f"script:\n{result.script}\n"
                    f"error:\n{result.error}\n"
                    f"remaining goals:\n{_compact_goal_text(result.goals)}"
                    + (f"\n{finish_guidance}" if finish_guidance else "")
                )
                if checkpoint_name:
                    try:
                        theorem.reverse(checkpoint_name)
                    except Exception as exc:
                        observations.append(f"rollback after failed finish failed: {exc}")
                        result.usage_events = list(usage_events)
                        result.usage = LLMUsage.aggregate(usage_events)
                        return result
                continue

            if name == "run_tac":
                tactic = str(action.get("tactic", "")).strip()
                try:
                    before_goals = theorem.goals()
                except Exception:
                    before_goals = []
                before_state_key = _compact_goal_text(before_goals)[:8000]
                seen_goal_states.add(before_state_key)
                failed_key = (tactic, _compact_goal_text(before_goals)[:2000])
                if failed_key in failed_tactics:
                    observations.append(
                        f"Skipped repeated failed tactic {tactic!r} on the "
                        "same proof state. Choose a different command or "
                        "explain a different intended subgoal with "
                        "allow_new_goals=true."
                    )
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="skip_repeated_tactic",
                        result=f"Skipped repeated failed tactic on same proof state: {tactic}",
                    )
                    continue
                checkpoint_name = f"_agent_run_tac_{step}"
                try:
                    theorem.checkpoint(checkpoint_name)
                except Exception:
                    checkpoint_name = ""
                _emit(on_event, verbose, kind="tactic_start", tactic=tactic)
                out = tools["run_tac"](tactic)
                after_goals = out.get("goals")
                if not isinstance(after_goals, list):
                    try:
                        after_goals = theorem.goals()
                    except Exception:
                        after_goals = []
                after_count = int(out.get("goal_count") if out.get("goal_count") is not None else len(after_goals))
                _emit(
                    on_event,
                    verbose,
                    kind="tactic_result",
                    tactic=tactic,
                    ok=out.get("ok", False),
                    goal_count=after_count,
                    error=out.get("error", ""),
                )
                if not out.get("ok", False):
                    if checkpoint_name:
                        try:
                            theorem.reverse(checkpoint_name)
                        except Exception:
                            pass
                    failed_tactics.add(failed_key)
                    observations.append(_tactic_failure_guidance(tactic, out))
                    continue

                before_count = len(before_goals)
                allow_new_goals = bool(action.get("allow_new_goals", False))
                after_state_key = _compact_goal_text(after_goals)[:8000]
                replace_edge = _replace_sides(tactic)
                if _should_rollback_semantic_unfold(tactic):
                    rollback = None
                    if checkpoint_name:
                        try:
                            rollback = theorem.reverse(checkpoint_name)
                        except Exception as exc:
                            rollback = {"ok": False, "error": str(exc)}
                    failed_tactics.add(failed_key)
                    message = (
                        f"run_tac({tactic!r}) unfolded a semantic predicate; "
                        "the runner rolled it back. Use selected lemmas or "
                        "automation rather than exposing that internal "
                        "definition."
                    )
                    observations.append(message + f"\nrollback -> {rollback}")
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="rollback",
                        result=message,
                    )
                    continue
                if (
                    after_count == before_count
                    and after_goals == before_goals
                    and _should_rollback_no_progress(tactic)
                ):
                    rollback = None
                    if checkpoint_name:
                        try:
                            rollback = theorem.reverse(checkpoint_name)
                        except Exception as exc:
                            rollback = {"ok": False, "error": str(exc)}
                    failed_tactics.add(failed_key)
                    message = (
                        f"run_tac({tactic!r}) succeeded but did not change the "
                        "proof state; the runner rolled it back so the next "
                        "tool call can try a tactic that makes progress."
                    )
                    observations.append(message + f"\nrollback -> {rollback}")
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="rollback",
                        result=message,
                    )
                    continue
                if (
                    after_count == before_count
                    and replace_edge is not None
                    and (replace_edge[1], replace_edge[0]) in replace_edges
                ):
                    rollback = None
                    if checkpoint_name:
                        try:
                            rollback = theorem.reverse(checkpoint_name)
                        except Exception as exc:
                            rollback = {"ok": False, "error": str(exc)}
                    failed_tactics.add(failed_key)
                    message = (
                        f"run_tac({tactic!r}) succeeded but undid a previous "
                        "replacement; the runner rolled it back to avoid an "
                        "algebraic normalization cycle. Use the exposed shape "
                        "with a selected rewrite lemma or continue toward the "
                        "focused goal."
                    )
                    observations.append(message + f"\nrollback -> {rollback}")
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="rollback",
                        result=message,
                    )
                    continue
                if (
                    after_count == before_count
                    and after_goals != before_goals
                    and after_state_key in seen_goal_states
                    and _should_rollback_cycle(tactic)
                ):
                    rollback = None
                    if checkpoint_name:
                        try:
                            rollback = theorem.reverse(checkpoint_name)
                        except Exception as exc:
                            rollback = {"ok": False, "error": str(exc)}
                    failed_tactics.add(failed_key)
                    message = (
                        f"run_tac({tactic!r}) succeeded but returned to a "
                        "previously seen proof state; the runner rolled it back "
                        "to avoid a transformation cycle."
                    )
                    observations.append(message + f"\nrollback -> {rollback}")
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="rollback",
                        result=message,
                    )
                    continue
                if (
                    after_count == before_count
                    and after_goals != before_goals
                    and _should_rollback_complexity_increase(
                        tactic,
                        before_goals,
                        after_goals,
                    )
                ):
                    rollback = None
                    if checkpoint_name:
                        try:
                            rollback = theorem.reverse(checkpoint_name)
                        except Exception as exc:
                            rollback = {"ok": False, "error": str(exc)}
                    failed_tactics.add(failed_key)
                    message = (
                        f"run_tac({tactic!r}) succeeded but made the printed "
                        "goal substantially more complex; the runner rolled it "
                        "back. Prefer transformations that expose the pattern "
                        "needed by a selected lemma or by the focused algebraic "
                        "goal."
                    )
                    observations.append(message + f"\nrollback -> {rollback}")
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="rollback",
                        result=message,
                    )
                    continue
                if (
                    after_count > before_count
                    and not allow_new_goals
                    and _should_rollback_goal_increase(tactic)
                ):
                    rollback = None
                    if checkpoint_name:
                        try:
                            rollback = theorem.reverse(checkpoint_name)
                        except Exception as exc:
                            rollback = {"ok": False, "error": str(exc)}
                    message = (
                        f"run_tac({tactic!r}) succeeded but increased the number "
                        f"of goals from {before_count} to {after_count}; the "
                        "runner treated this as an accidental side-goal creation "
                        "and rolled it back. Use a tactic that discharges its "
                        "own side conditions, or set allow_new_goals=true only "
                        "when the new goals are intended."
                    )
                    observations.append(message + f"\nrollback -> {rollback}")
                    _emit(
                        on_event,
                        verbose,
                        kind="tool_result",
                        action="rollback",
                        result=message,
                    )
                    continue

                success_note = _tactic_success_guidance(tactic, out)
                observations.append(
                    f"run_tac({tactic!r}) -> {out}"
                    + (f"\n{success_note}" if success_note else "")
                )
                cleanup_outputs = _try_safe_cleanup(
                    theorem,
                    on_event=on_event,
                    verbose=verbose,
                    step=step,
                )
                if cleanup_outputs:
                    observations.append(
                        "Safe arithmetic cleanup reduced the remaining goals."
                    )
                    try:
                        after_goals = theorem.goals()
                    except Exception:
                        after_goals = []
                    after_count = len(after_goals)
                    after_state_key = _compact_goal_text(after_goals)[:8000]
                seen_goal_states.add(after_state_key)
                if replace_edge is not None:
                    replace_edges.add(replace_edge)
                if out.get("ok") and (out.get("proof_finished") or after_count == 0):
                    if not close:
                        return ProofResult(
                            ok=True,
                            script=theorem.proof_script(),
                            goals=[],
                            outputs=[out] + cleanup_outputs,
                            completed=False,
                            attempts=attempts,
                            usage=LLMUsage.aggregate(usage_events),
                            usage_events=list(usage_events),
                        )
                    qed = theorem.qed()
                    _emit(
                        on_event,
                        verbose,
                        kind="qed",
                        ok=qed.get("ok", False),
                        error=qed.get("error", ""),
                    )
                    return ProofResult(
                        ok=bool(qed.get("ok")),
                        script=theorem.proof_script(),
                        error="" if qed.get("ok") else qed.get("error", "Rocq rejected Qed."),
                        goals=[] if qed.get("ok") else qed.get("goals", theorem.goals()),
                        outputs=[out] + cleanup_outputs + [qed],
                        completed=bool(qed.get("ok")),
                        attempts=attempts,
                        usage=LLMUsage.aggregate(usage_events),
                        usage_events=list(usage_events),
                    )
                continue

            if name == "reverse":
                checkpoint = action.get("checkpoint", 0)
                out = tools["reverse"](checkpoint)
                _emit(on_event, verbose, kind="tool_result", action=name, result=out)
                observations.append(f"reverse({checkpoint!r}) -> {out}")
                continue

            observations.append(f"Unknown action from model: {action}")

        return ProofResult(
            ok=False,
            script=theorem.proof_script(),
            error=f"Agentic proof exceeded max_tool_calls={max_tool_calls}.",
            goals=theorem.goals(),
            attempts=attempts,
            usage=LLMUsage.aggregate(usage_events),
            usage_events=list(usage_events),
        )


def _parse_json_action(text: str) -> dict[str, Any] | None:
    raw = _strip_code_fences(text, prefer_json=True)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
