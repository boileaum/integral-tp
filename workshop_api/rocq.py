from __future__ import annotations

import math
import re
import textwrap
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

try:
    from rocq_ml_toolbox.inference.client import PytanqueExtended
except Exception as exc:  # pragma: no cover - exercised in notebooks.
    try:
        from pytanque import Pytanque, PytanqueMode
    except Exception as pytanque_exc:  # pragma: no cover - dependency error path.
        PytanqueExtended = None  # type: ignore[assignment]
        _IMPORT_ERROR = pytanque_exc
    else:

        class PytanqueExtended(Pytanque):  # type: ignore[no-redef]
            """Minimal local client for the Dockerized rocq-ml-server.

            The full `rocq_ml_toolbox.inference.client.PytanqueExtended` is
            still used when installed. Colab only needs a small subset of its
            file API, so this fallback avoids requiring a private toolbox clone.
            """

            def __init__(self, host: str, port: int):
                super().__init__(host=host, port=port, mode=PytanqueMode.HTTP)

            def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
                url = f"http://{self.host}:{self.port}/{endpoint.lstrip('/')}"
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response.json()

            def tmp_file(
                self,
                content: str | None = None,
                root: str | Path | None = None,
            ) -> str:
                result = self._post_json(
                    "tmp_file",
                    {"content": content, "root": None if root is None else str(root)},
                )
                if not isinstance(result, dict) or not isinstance(result.get("path"), str):
                    raise ValueError("Invalid response from /tmp_file: missing string `path`.")
                return result["path"]

        _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if PytanqueExtended is not None:

    class UrlPytanqueExtended(PytanqueExtended):  # type: ignore[misc, valid-type]
        """HTTP Rocq client variant for path-prefixed HTTPS proxies.

        The public `pytanque` HTTP client currently builds URLs from `(host, port)`
        as `http://host:port/...`. That is fine for local Docker, but not for a
        single ngrok endpoint routed as `https://host/rocq/...`.
        """

        def __init__(self, base_url: str, *, timeout_http: float | None = None):
            super().__init__("127.0.0.1", 1)
            self.base_url = base_url.rstrip("/") + "/"
            self.timeout_http = timeout_http

        def _headers(self) -> dict[str, str]:
            return {"ngrok-skip-browser-warning": "true"}

        def _url(self, endpoint: str) -> str:
            return urljoin(self.base_url, endpoint.lstrip("/"))

        def connect(self) -> None:
            response = requests.get(
                self._url("login"),
                headers=self._headers(),
                timeout=self.timeout_http,
            )
            response.raise_for_status()
            self.session_id = response.json()["session_id"]

        def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
            response = requests.post(
                self._url(endpoint),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_http,
            )
            response.raise_for_status()
            return response.json()

        def _send_request_message(
            self,
            route_name: Any,
            payload: Any,
            timeout: float | None = None,
        ) -> str:
            payload["timeout"] = _rocq_timeout(timeout)
            payload["route_name"] = route_name
            payload["session_id"] = self.session_id
            response = requests.post(
                self._url("rpc"),
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_http,
            )
            response.raise_for_status()
            return response.text

else:
    UrlPytanqueExtended = None  # type: ignore[assignment]


LEMMA_HEADER_RE = re.compile(
    r"^\s*(Lemma|Theorem|Fact|Proposition|Corollary)\s+([A-Za-z_][A-Za-z0-9_']*)\b",
    re.MULTILINE,
)


def _clean_command(command: str) -> str:
    out = textwrap.dedent(command).strip()
    if not out:
        raise ValueError("Empty Rocq command.")
    return out


def _rocq_timeout(timeout: float | int | None) -> int | None:
    if timeout is None:
        return None
    return max(1, math.ceil(float(timeout)))


def _strip_code_fence(text: str) -> str:
    out = textwrap.dedent(str(text)).strip()
    if out.startswith("```"):
        lines = out.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        out = "\n".join(lines).strip()
    return out


def _goal_text(goal: Any) -> str:
    return str(getattr(goal, "pp", None) or getattr(goal, "ty", None) or goal)


def _feedback(state: Any) -> list[str]:
    raw = getattr(state, "feedback", None)
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append(str(item[1]))
        elif isinstance(item, dict):
            out.append(str(item.get("message", item)))
        else:
            out.append(str(item))
    return out


@dataclass
class StateNode:
    index: int
    parent_index: int | None
    command: str | None
    state: Any


@dataclass
class LemmaSession:
    name: str
    header: str
    nodes: list[StateNode]
    latest_index: int = 0
    completed: bool = False
    qed_state: Any | None = None

    def node(self, state_index: int) -> StateNode:
        if state_index < 0 or state_index >= len(self.nodes):
            raise ValueError(
                f"Unknown state_index={state_index}; "
                f"available={[node.index for node in self.nodes]}"
            )
        return self.nodes[state_index]

    def active_commands(self, state_index: int | None = None) -> list[str]:
        idx = self.latest_index if state_index is None else state_index
        commands: list[str] = []
        while idx is not None and idx > 0:
            node = self.node(idx)
            if node.command:
                commands.append(node.command)
            idx = node.parent_index
        commands.reverse()
        return commands

    def source(self, *, state_index: int | None = None, close: bool | None = None) -> str:
        if close is None:
            close = self.completed
        commands = self.active_commands(state_index)
        proof = "\n".join(commands).strip()
        if proof:
            proof = textwrap.indent(proof, "  ")
        else:
            proof = "  (* no proof steps yet *)"
        end = "Qed." if close else "Abort."
        return f"{self.header}\nProof.\n{proof}\n{end}\n"


class RocqWorkshop:
    """Small teaching API over rocq-ml-server/Petanque.

    The object keeps one global Rocq state for imports/definitions/completed
    lemmas. Each open lemma has its own state history. Calling `run_tac` from an
    older state creates a branch and moves that lemma's active head to the new
    state.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        *,
        timeout: float = 180.0,
        connect: bool = True,
        server_url: str | None = None,
    ):
        rocq_timeout = _rocq_timeout(timeout)
        http_timeout = None if timeout is None else float(timeout)
        if PytanqueExtended is None:
            raise RuntimeError(
                "Could not import rocq_ml_toolbox or public pytanque. "
                "Install `integral-tp[colab]` or install pytanque manually."
            ) from _IMPORT_ERROR
        if server_url:
            if UrlPytanqueExtended is None:
                raise RuntimeError(
                    "Could not create a URL-based Rocq client because pytanque is unavailable."
                ) from _IMPORT_ERROR
            self.client = UrlPytanqueExtended(server_url, timeout_http=http_timeout)
        else:
            self.client = PytanqueExtended(host, port)
        if connect:
            self.client.connect()
        self.timeout = rocq_timeout
        self.host = host
        self.port = port
        self.server_url = server_url
        self.root_path = Path(self.client.tmp_file(content="")).resolve()
        self.root_state = self.client.get_root_state(str(self.root_path), timeout=self.timeout)
        self.global_state = self.root_state
        self.elements: list[str] = []
        self.lemmas: dict[str, LemmaSession] = {}
        self.completed_order: list[str] = []
        self.timeline: list[tuple[str, str]] = []

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def add_element(self, command: str, *, refresh_open_lemmas: bool = True) -> dict[str, Any]:
        command = _clean_command(command)
        new_state = self.client.run(self.global_state, command, timeout=self.timeout)
        self.global_state = new_state
        self.elements.append(command)
        self.timeline.append(("element", command))
        refreshed = self.refresh_open_lemmas() if refresh_open_lemmas else []
        return {
            "ok": True,
            "kind": "element",
            "command": command,
            "feedback": _feedback(new_state),
            "refreshed_open_lemmas": refreshed,
        }

    def add_import(self, libname: str, source: str) -> dict[str, Any]:
        return self.add_element(f"From {libname} Require Import {source}.")

    def open_scope(self, scope: str) -> dict[str, Any]:
        return self.add_element(f"Open Scope {scope}.")

    def add_definition(self, command: str) -> dict[str, Any]:
        if not re.search(r"^\s*(Definition|Fixpoint|CoFixpoint|Let)\b", command):
            raise ValueError("add_definition expects a Definition/Fixpoint/CoFixpoint/Let command.")
        return self.add_element(command)

    def add_lemma(self, header: str, *, name: str | None = None) -> dict[str, Any]:
        header = _clean_command(header)
        if re.search(r"\b(Proof|Qed|Admitted|Defined|Abort)\b", header):
            raise ValueError("Pass only the lemma/theorem statement, without Proof/Qed.")
        match = LEMMA_HEADER_RE.search(header)
        if not match and name is None:
            raise ValueError("Could not infer lemma name. Pass name=... explicitly.")
        inferred_name = match.group(2) if match else name
        assert inferred_name is not None
        if inferred_name in self.lemmas:
            raise ValueError(f"Lemma `{inferred_name}` already exists in this session.")

        proof_state = self.client.run(self.global_state, header, timeout=self.timeout)
        lemma = LemmaSession(
            name=inferred_name,
            header=header,
            nodes=[StateNode(index=0, parent_index=None, command=None, state=proof_state)],
        )
        self.lemmas[inferred_name] = lemma
        return {
            "ok": True,
            "lemma": inferred_name,
            "state_index": 0,
            "goals": self.goals(inferred_name)["goals"],
        }

    def ensure_lemma(self, header: str, *, name: str | None = None) -> dict[str, Any]:
        """Declare a lemma idempotently, replacing a changed open statement.

        Open lemmas live on branches rooted at ``global_state``, so replacing
        one is safe. A completed lemma is already part of the global Rocq state
        and therefore cannot be redefined in the same document.
        """

        header = _clean_command(header)
        if re.search(r"\b(Proof|Qed|Admitted|Defined|Abort)\b", header):
            raise ValueError("Pass only the lemma/theorem statement, without Proof/Qed.")
        match = LEMMA_HEADER_RE.search(header)
        if not match and name is None:
            raise ValueError("Could not infer lemma name. Pass name=... explicitly.")
        inferred_name = match.group(2) if match else name
        assert inferred_name is not None

        existing = self.lemmas.get(inferred_name)
        if existing is None:
            return self.add_lemma(header, name=name)
        if existing.header == header:
            return {
                "ok": True,
                "lemma": inferred_name,
                "state_index": existing.latest_index,
                "goals": self.goals(inferred_name)["goals"],
                "already_exists": True,
            }
        if existing.completed:
            raise ValueError(
                f"Lemma `{inferred_name}` is already completed and cannot be "
                "redefined in this Rocq document. Start a new document or use a new name."
            )

        # Run first so an invalid correction does not destroy the usable branch.
        proof_state = self.client.run(self.global_state, header, timeout=self.timeout)
        self.lemmas[inferred_name] = LemmaSession(
            name=inferred_name,
            header=header,
            nodes=[StateNode(index=0, parent_index=None, command=None, state=proof_state)],
        )
        return {
            "ok": True,
            "lemma": inferred_name,
            "state_index": 0,
            "goals": self.goals(inferred_name)["goals"],
            "replaced_open_lemma": True,
        }

    def list_lemmas(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, lemma in self.lemmas.items():
            out.append(
                {
                    "name": name,
                    "completed": lemma.completed,
                    "latest_state_index": lemma.latest_index,
                    "state_count": len(lemma.nodes),
                    "active_step_count": len(lemma.active_commands()),
                }
            )
        return out

    def goals(self, lemma_name: str, *, state_index: int | None = None) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        idx = lemma.latest_index if state_index is None else state_index
        state = lemma.node(idx).state
        goals = [_goal_text(goal) for goal in self.client.goals(state, timeout=self.timeout)]
        return {
            "lemma": lemma_name,
            "state_index": idx,
            "goals": goals,
            "goal_count": len(goals),
            "proof_finished": bool(getattr(state, "proof_finished", False)) or len(goals) == 0,
        }

    def run_tac(
        self,
        lemma_name: str,
        tactic: str,
        *,
        state_index: int | None = None,
    ) -> dict[str, Any]:
        tactic = _clean_command(tactic)
        lemma = self.lemmas[lemma_name]
        idx = lemma.latest_index if state_index is None else state_index
        parent = lemma.node(idx)
        try:
            new_state = self.client.run(parent.state, tactic, timeout=self.timeout)
        except Exception as exc:
            return {
                "ok": False,
                "lemma": lemma_name,
                "source_state_index": idx,
                "tactic": tactic,
                "error": str(exc),
                "feedback": _feedback(parent.state),
            }
        new_idx = len(lemma.nodes)
        lemma.nodes.append(
            StateNode(index=new_idx, parent_index=idx, command=tactic, state=new_state)
        )
        lemma.latest_index = new_idx
        goal_info = self.goals(lemma_name, state_index=new_idx)
        return {
            "ok": True,
            "lemma": lemma_name,
            "source_state_index": idx,
            "new_state_index": new_idx,
            "tactic": tactic,
            "goals": goal_info["goals"],
            "goal_count": goal_info["goal_count"],
            "proof_finished": goal_info["proof_finished"],
            "feedback": _feedback(new_state),
        }

    def run_script(self, lemma_name: str, steps: Iterable[str]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for step in steps:
            out = self.run_tac(lemma_name, step)
            outputs.append(out)
            if not out.get("ok", False):
                break
        return outputs

    def complete_lemma(self, lemma_name: str, *, refresh_open_lemmas: bool = True) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        if lemma.completed:
            return {"ok": True, "lemma": lemma_name, "already_completed": True}
        latest = lemma.node(lemma.latest_index)
        try:
            qed_state = self.client.run(latest.state, "Qed.", timeout=self.timeout)
        except Exception as exc:
            return {
                "ok": False,
                "lemma": lemma_name,
                "error": str(exc),
                "goals": self.goals(lemma_name)["goals"],
            }
        lemma.completed = True
        lemma.qed_state = qed_state
        self.global_state = qed_state
        self.completed_order.append(lemma_name)
        self.timeline.append(("lemma", lemma_name))
        refreshed = self.refresh_open_lemmas(except_names={lemma_name}) if refresh_open_lemmas else []
        return {
            "ok": True,
            "lemma": lemma_name,
            "feedback": _feedback(qed_state),
            "refreshed_open_lemmas": refreshed,
        }

    def restart_lemma(self, lemma_name: str) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        if lemma.completed:
            raise ValueError(f"Lemma `{lemma_name}` is already completed.")
        commands = lemma.active_commands()
        proof_state = self.client.run(self.global_state, lemma.header, timeout=self.timeout)
        new_nodes = [StateNode(index=0, parent_index=None, command=None, state=proof_state)]
        latest_index = 0
        for command in commands:
            new_state = self.client.run(new_nodes[latest_index].state, command, timeout=self.timeout)
            latest_index = len(new_nodes)
            new_nodes.append(
                StateNode(
                    index=latest_index,
                    parent_index=latest_index - 1,
                    command=command,
                    state=new_state,
                )
            )
        lemma.nodes = new_nodes
        lemma.latest_index = latest_index
        return {
            "ok": True,
            "lemma": lemma_name,
            "replayed_step_count": len(commands),
            "latest_state_index": lemma.latest_index,
            "goals": self.goals(lemma_name)["goals"],
        }

    def reset_lemma(self, lemma_name: str) -> dict[str, Any]:
        lemma = self.lemmas[lemma_name]
        if lemma.completed:
            raise ValueError(f"Lemma `{lemma_name}` is already completed.")
        proof_state = self.client.run(self.global_state, lemma.header, timeout=self.timeout)
        lemma.nodes = [StateNode(index=0, parent_index=None, command=None, state=proof_state)]
        lemma.latest_index = 0
        return {
            "ok": True,
            "lemma": lemma_name,
            "latest_state_index": 0,
            "goals": self.goals(lemma_name)["goals"],
        }

    def refresh_open_lemmas(self, *, except_names: set[str] | None = None) -> list[dict[str, Any]]:
        except_names = except_names or set()
        refreshed: list[dict[str, Any]] = []
        for name, lemma in list(self.lemmas.items()):
            if lemma.completed or name in except_names:
                continue
            try:
                refreshed.append(self.restart_lemma(name))
            except Exception as exc:
                refreshed.append({"ok": False, "lemma": name, "error": str(exc)})
        return refreshed

    def proof_script(self, lemma_name: str) -> str:
        return "\n".join(self.lemmas[lemma_name].active_commands())

    def lemma_source(self, lemma_name: str, *, close: bool | None = None) -> str:
        return self.lemmas[lemma_name].source(close=close)

    def source(self, *, include_open: bool = True) -> str:
        blocks: list[str] = []
        for kind, value in self.timeline:
            if kind == "element":
                blocks.append(value)
            elif kind == "lemma":
                blocks.append(self.lemmas[value].source(close=True))
        if include_open:
            for _name, lemma in self.lemmas.items():
                if not lemma.completed:
                    blocks.append(lemma.source(close=False))
        return "\n\n".join(block.rstrip() for block in blocks if block.strip()) + "\n"


@dataclass
class TheoremSession:
    """Notebook-facing handle for an open or completed Rocq theorem."""

    document: "RocqDocument"
    name: str
    checkpoints: dict[str, int] = field(default_factory=dict)

    @property
    def _workshop(self) -> RocqWorkshop:
        return self.document.workshop

    @property
    def lemma(self) -> LemmaSession:
        return self._workshop.lemmas[self.name]

    @property
    def header(self) -> str:
        return self.lemma.header

    @property
    def completed(self) -> bool:
        return self.lemma.completed

    def goals(self) -> list[str]:
        return self._workshop.goals(self.name)["goals"]

    def run_tac(self, tactic: str) -> dict[str, Any]:
        return self._workshop.run_tac(self.name, tactic)

    def run_script(self, script: str | Iterable[str]) -> list[dict[str, Any]]:
        if isinstance(script, str):
            from .llm import split_rocq_commands

            steps = split_rocq_commands(script)
        else:
            steps = list(script)
        return self._workshop.run_script(self.name, steps)

    def checkpoint(self, name: str | None = None) -> str:
        if name is None:
            name = f"checkpoint_{len(self.checkpoints) + 1}"
        self.checkpoints[name] = self.lemma.latest_index
        return name

    def reverse(self, checkpoint: str | int = 0) -> dict[str, Any]:
        if self.completed:
            raise ValueError(f"Cannot reverse completed theorem `{self.name}`.")
        index = self.checkpoints[checkpoint] if isinstance(checkpoint, str) else checkpoint
        self.lemma.node(index)
        self.lemma.latest_index = index
        return {
            "ok": True,
            "lemma": self.name,
            "latest_state_index": index,
            "goals": self.goals(),
        }

    def reset(self) -> dict[str, Any]:
        self.checkpoints.clear()
        return self._workshop.reset_lemma(self.name)

    def qed(self) -> dict[str, Any]:
        return self._workshop.complete_lemma(self.name)

    def source(self, *, close: bool | None = None) -> str:
        return self._workshop.lemma_source(self.name, close=close)

    def proof_script(self) -> str:
        return self._workshop.proof_script(self.name)

    def as_retrieval_hit(self) -> dict[str, Any]:
        return {
            "uid": f"local:{self.name}",
            "name": self.name,
            "kind": "local_theorem",
            "library": "Current document",
            "source": "current Rocq session",
            "statement": self.header,
            "docstring": self.source(close=self.completed),
            "content": self.source(close=self.completed),
        }

    def __str__(self) -> str:
        info = self._workshop.goals(self.name)
        status = "completed" if self.completed else f"{info['goal_count']} goal(s)"
        parts = [f"{self.name}: {status}", self.header]
        if not self.completed:
            goals = info["goals"]
            if goals:
                parts.append("Goals:\n" + "\n\n".join(goals))
            else:
                parts.append("No remaining goals. Run `.qed()` to close the theorem.")
        return "\n\n".join(parts)

    __repr__ = __str__


class RocqDocument:
    """Notebook-friendly document API over :class:`RocqWorkshop`."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        timeout: float = 600.0,
        connect: bool = True,
        server_url: str | None = None,
    ):
        server_url = (
            server_url
            or os.getenv("ROCQ_SERVER_URL")
            or os.getenv("ROCQ_ML_SERVER_URL")
            or None
        )
        if server_url is None and host and "://" in host:
            server_url = host
            host = None
        host = (
            host
            or os.getenv("ROCQ_SERVER_HOST")
            or os.getenv("ROCQ_ML_SERVER_HOST")
            or "127.0.0.1"
        )
        port = port or int(
            os.getenv("ROCQ_SERVER_PORT") or os.getenv("ROCQ_ML_SERVER_PORT") or "5000"
        )
        self.workshop = RocqWorkshop(
            host=host,
            port=port,
            timeout=timeout,
            connect=connect,
            server_url=server_url,
        )

    def close(self) -> None:
        self.workshop.close()

    def add_import(self, libname: str, source: str) -> dict[str, Any]:
        return self.workshop.add_import(libname, source)

    def open_scope(self, scope: str) -> dict[str, Any]:
        return self.workshop.open_scope(scope)

    def execute(self, command: str) -> dict[str, Any]:
        return self.workshop.add_element(command)

    def add_definition(self, command: str) -> dict[str, Any]:
        return self.workshop.add_definition(command)

    def add_theorem(self, header: str, *, name: str | None = None) -> TheoremSession:
        header = _strip_code_fence(header)
        result = self.workshop.add_lemma(header, name=name)
        return TheoremSession(self, result["lemma"])

    def ensure_theorem(self, header: str, *, name: str | None = None) -> TheoremSession:
        """Return an existing theorem, or replace it if its open statement changed."""

        header = _strip_code_fence(header)
        result = self.workshop.ensure_lemma(header, name=name)
        return TheoremSession(self, result["lemma"])

    def add_artefact(self, artefact: str, *, name: str | None = None) -> TheoremSession:
        header = _strip_code_fence(artefact)
        if "Proof." in header:
            header = header.split("Proof.", 1)[0].strip()
        if name is not None and not LEMMA_HEADER_RE.search(header):
            header = f"Lemma {name} :\n  {header}."
        return self.add_theorem(header, name=name)

    def source(self, *, include_open: bool = True) -> str:
        return self.workshop.source(include_open=include_open)


def new_document(
    host: str | None = None,
    port: int | None = None,
    *,
    timeout: float = 600.0,
    connect: bool = True,
    server_url: str | None = None,
) -> RocqDocument:
    return RocqDocument(
        host=host,
        port=port,
        timeout=timeout,
        connect=connect,
        server_url=server_url,
    )
