from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import ssl
import statistics
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import requests

try:
    import websocket
except Exception as exc:  # pragma: no cover - dependency error path.
    websocket = None  # type: ignore[assignment]
    _WEBSOCKET_IMPORT_ERROR = exc
else:
    _WEBSOCKET_IMPORT_ERROR = None


DEFAULT_QUERIES = [
    "exponential is strictly positive",
    "derivative of the sum of two real functions",
    "integral of a derivative between two bounds",
    "a strictly positive real number is nonzero",
]

RESULT_MARKER = "__INTEGRAL_TP_STRESS_RESULT__"


def parse_waves(value: str) -> list[int]:
    """Parse a comma-separated group of positive virtual-user counts."""

    waves = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not waves or any(wave <= 0 for wave in waves):
        raise ValueError("All user-wave values must be positive integers.")
    return waves


def _extract_result(stdout: str, *, phase: str) -> dict[str, Any]:
    """Return the last structured kernel result for ``phase``."""

    for line in reversed(stdout.splitlines()):
        if not line.startswith(RESULT_MARKER):
            continue
        try:
            result = json.loads(line.removeprefix(RESULT_MARKER))
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("phase") == phase:
            return result
    raise ValueError(f"Kernel output did not contain a {phase!r} result marker.")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def without_install_commands(code: str) -> str:
    """Remove notebook package-install commands from an already provisioned server."""

    kept: list[str] = []
    skip_continuation = False
    for line in code.splitlines():
        stripped = line.lstrip()
        starts_install = stripped.startswith(("%pip ", "!pip ", "pip install "))
        if starts_install or skip_continuation:
            skip_continuation = line.rstrip().endswith("\\")
            continue
        kept.append(line)
    return "\n".join(kept)


def websocket_url(base_url: str, kernel_id: str, session_id: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/api/kernels/{kernel_id}/channels"
    return urlunsplit((scheme, parsed.netloc, path, urlencode({"session_id": session_id}), ""))


@dataclass
class ExecutionResult:
    ok: bool
    latency_s: float
    output: str = ""
    error: str = ""


@dataclass
class VirtualUserResult:
    user_id: int
    ok: bool
    wave: int = 0
    kernel_create_s: float = 0.0
    setup_s: float = 0.0
    model_load_s: float = 0.0
    search_latencies_s: list[float] = field(default_factory=list)
    search_roundtrip_s: list[float] = field(default_factory=list)
    searches_completed: int = 0
    rss_mb: float = 0.0
    pss_mb: float = 0.0
    private_mb: float = 0.0
    error_phase: str = ""
    error: str = ""


class JupyterApi:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        verify_tls: bool | str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.verify_tls = verify_tls
        self.timeout = timeout
        self.headers = {"Authorization": f"token {token}"}

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            verify=self.verify_tls,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def server_info(self) -> dict[str, Any]:
        return dict(self.request("GET", "/api").json())

    def server_status(self) -> dict[str, Any] | None:
        try:
            return dict(self.request("GET", "/api/status").json())
        except (requests.RequestException, ValueError):
            return None

    def notebook(self, path: str) -> dict[str, Any]:
        encoded_path = quote(path.strip("/"), safe="/")
        payload = self.request(
            "GET",
            f"/api/contents/{encoded_path}",
            # Jupyter Server 2.7 rejects ``format=json`` for notebooks even
            # though directory responses describe their format as JSON.
            params={"content": 1},
        ).json()
        content = payload.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("cells"), list):
            raise RuntimeError(f"Jupyter did not return notebook content for {path!r}.")
        return content

    def create_kernel(self, kernel_name: str) -> str:
        response = self.request("POST", "/api/kernels", json={"name": kernel_name})
        kernel_id = str(response.json().get("id") or "")
        if not kernel_id:
            raise RuntimeError("Jupyter created a kernel without returning its id.")
        return kernel_id

    def delete_kernel(self, kernel_id: str) -> None:
        try:
            self.request("DELETE", f"/api/kernels/{kernel_id}")
        except requests.HTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 404:
                raise


class KernelChannels:
    def __init__(
        self,
        *,
        api: JupyterApi,
        kernel_id: str,
        execution_timeout: float,
    ) -> None:
        if websocket is None:
            raise RuntimeError(
                "Install websocket-client to run the Jupyter stress test."
            ) from _WEBSOCKET_IMPORT_ERROR
        self.api = api
        self.kernel_id = kernel_id
        self.execution_timeout = execution_timeout
        self.session_id = uuid.uuid4().hex
        sslopt: dict[str, Any] = {}
        if api.verify_tls is False:
            sslopt = {"cert_reqs": ssl.CERT_NONE, "check_hostname": False}
        elif isinstance(api.verify_tls, str):
            sslopt = {"ca_certs": api.verify_tls, "cert_reqs": ssl.CERT_REQUIRED}
        self.socket = websocket.create_connection(
            websocket_url(api.base_url, kernel_id, self.session_id),
            header=[f"Authorization: token {api.token}"],
            timeout=execution_timeout,
            sslopt=sslopt,
        )

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception:
            pass

    def execute(self, code: str) -> ExecutionResult:
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": message_id,
                "username": "integral-tp-stress",
                "session": self.session_id,
                "date": datetime.now(timezone.utc).isoformat(),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        started = time.perf_counter()
        self.socket.send(json.dumps(message))
        reply_received = False
        idle_received = False
        outputs: list[str] = []
        error = ""
        deadline = started + self.execution_timeout

        while time.perf_counter() < deadline:
            self.socket.settimeout(max(0.1, deadline - time.perf_counter()))
            try:
                raw = self.socket.recv()
            except websocket.WebSocketTimeoutException:
                break
            if not raw:
                continue
            if isinstance(raw, bytes):
                return ExecutionResult(
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error="Jupyter returned an unsupported binary WebSocket message.",
                )
            incoming = json.loads(raw)
            parent_id = str(incoming.get("parent_header", {}).get("msg_id") or "")
            if parent_id != message_id:
                continue
            message_type = str(incoming.get("msg_type") or incoming.get("header", {}).get("msg_type") or "")
            content = incoming.get("content", {})

            if message_type == "stream":
                outputs.append(str(content.get("text") or ""))
            elif message_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                if isinstance(data, dict) and data.get("text/plain") is not None:
                    outputs.append(str(data["text/plain"]))
            elif message_type == "error":
                traceback = content.get("traceback") or []
                error = "\n".join(str(line) for line in traceback) or str(content)
            elif message_type == "execute_reply":
                reply_received = True
                if content.get("status") != "ok" and not error:
                    error = str(content)
            elif message_type == "status" and content.get("execution_state") == "idle":
                idle_received = True

            if reply_received and idle_received:
                latency = time.perf_counter() - started
                return ExecutionResult(
                    ok=not error,
                    latency_s=latency,
                    output="".join(outputs),
                    error=error,
                )

        return ExecutionResult(
            ok=False,
            latency_s=time.perf_counter() - started,
            output="".join(outputs),
            error=f"Kernel execution timed out after {self.execution_timeout:.1f}s.",
        )


@dataclass
class PreparedUser:
    user_id: int
    kernel_id: str
    channels: KernelChannels
    result: VirtualUserResult


def find_setup_code(
    notebook: dict[str, Any],
    *,
    setup_cell: int | None,
    include_install: bool,
    torch_threads: int | None = None,
    torch_interop_threads: int | None = None,
) -> tuple[int, str]:
    cells = notebook["cells"]
    if setup_cell is None:
        candidates = [
            index
            for index, cell in enumerate(cells)
            if cell.get("cell_type") == "code"
            and "retriever = RetrievalClient.from_env" in source_text(cell)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not identify one retrieval setup cell; pass --setup-cell explicitly. "
                f"Candidates: {candidates}"
            )
        setup_cell = candidates[0]
    if setup_cell < 0 or setup_cell >= len(cells):
        raise IndexError(f"Setup cell {setup_cell} is outside the notebook's cell range.")
    cell = cells[setup_cell]
    if cell.get("cell_type") != "code":
        raise TypeError(f"Notebook cell {setup_cell} is not a code cell.")
    code = source_text(cell)
    if not include_install:
        code = without_install_commands(code)
    torch_setup = ""
    if torch_threads is not None or torch_interop_threads is not None:
        torch_setup = "import torch\n"
        if torch_threads is not None:
            torch_setup += f"torch.set_num_threads({torch_threads})\n"
        if torch_interop_threads is not None:
            torch_setup += f"torch.set_num_interop_threads({torch_interop_threads})\n"
    instrumented = (
        "import json as _stress_json, time as _stress_time\n"
        "_stress_setup_started = _stress_time.perf_counter()\n"
        + torch_setup
        + code
        + "\n"
        + "try:\n"
        + "    with open('/proc/self/status', encoding='utf-8') as _stress_status:\n"
        + "        _stress_rss_mb = next(float(_line.split()[1]) / 1024 "
        + "for _line in _stress_status if _line.startswith('VmRSS:'))\n"
        + "except Exception:\n"
        + "    _stress_rss_mb = 0.0\n"
        + "try:\n"
        + "    _stress_smaps = {}\n"
        + "    with open('/proc/self/smaps_rollup', encoding='utf-8') as _stress_rollup:\n"
        + "        for _line in _stress_rollup:\n"
        + "            _parts = _line.split()\n"
        + "            _key = _parts[0].rstrip(':') if _parts else ''\n"
        + "            if len(_parts) >= 2 and _key in "
        + "{'Pss', 'Private_Clean', 'Private_Dirty'}:\n"
        + "                _stress_smaps[_key] = float(_parts[1]) / 1024\n"
        + "    _stress_pss_mb = _stress_smaps.get('Pss', 0.0)\n"
        + "    _stress_private_mb = (_stress_smaps.get('Private_Clean', 0.0) + "
        + "_stress_smaps.get('Private_Dirty', 0.0))\n"
        + "except Exception:\n"
        + "    _stress_pss_mb = 0.0\n"
        + "    _stress_private_mb = 0.0\n"
        + f"print({RESULT_MARKER!r} + _stress_json.dumps({{\n"
        + "    'phase': 'setup',\n"
        + "    'model_load_s': _stress_time.perf_counter() - _stress_setup_started,\n"
        + "    'rss_mb': _stress_rss_mb,\n"
        + "    'pss_mb': _stress_pss_mb,\n"
        + "    'private_mb': _stress_private_mb,\n"
        + "}))\n"
    )
    return setup_cell, instrumented


def search_code(query: str, *, library: str, kind: str, k: int) -> str:
    return (
        "import json as _stress_json, time as _stress_time\n"
        f"_stress_query = {query!r}\n"
        "_stress_started = _stress_time.perf_counter()\n"
        "_stress_hits = retriever.search(\n"
        "    _stress_query,\n"
        f"    library={library!r},\n"
        f"    kind={kind!r},\n"
        f"    k={k},\n"
        ")\n"
        f"print({RESULT_MARKER!r} + _stress_json.dumps({{\n"
        "    'phase': 'search',\n"
        "    'query': _stress_query,\n"
        "    'search_s': _stress_time.perf_counter() - _stress_started,\n"
        "    'hits': len(_stress_hits),\n"
        "}))\n"
    )


def prepare_user(
    api: JupyterApi,
    *,
    user_id: int,
    kernel_name: str,
    setup_code: str,
    execution_timeout: float,
    wave: int = 0,
) -> tuple[PreparedUser | None, VirtualUserResult]:
    result = VirtualUserResult(user_id=user_id, ok=False, wave=wave)
    kernel_id = ""
    channels: KernelChannels | None = None
    try:
        started = time.perf_counter()
        kernel_id = api.create_kernel(kernel_name)
        result.kernel_create_s = time.perf_counter() - started
        channels = KernelChannels(
            api=api,
            kernel_id=kernel_id,
            execution_timeout=execution_timeout,
        )
        setup = channels.execute(setup_code)
        result.setup_s = setup.latency_s
        if not setup.ok:
            result.error_phase = "setup"
            raise RuntimeError(setup.error or f"Setup failed. Output: {setup.output[-1000:]}")
        try:
            setup_data = _extract_result(setup.output, phase="setup")
        except ValueError:
            result.error_phase = "setup"
            raise RuntimeError(f"Setup result missing. Output: {setup.output[-1000:]}")
        result.model_load_s = float(setup_data.get("model_load_s") or 0.0)
        result.rss_mb = float(setup_data.get("rss_mb") or 0.0)
        result.pss_mb = float(setup_data.get("pss_mb") or 0.0)
        result.private_mb = float(setup_data.get("private_mb") or 0.0)
        return (
            PreparedUser(
                user_id=user_id,
                kernel_id=kernel_id,
                channels=channels,
                result=result,
            ),
            result,
        )
    except Exception as exc:
        if not result.error_phase:
            result.error_phase = "kernel"
        result.error = " ".join(str(exc).split())[:2000]
        if channels is not None:
            channels.close()
        if kernel_id:
            try:
                api.delete_kernel(kernel_id)
            except Exception:
                pass
        return None, result


def run_searches(
    prepared: PreparedUser,
    *,
    barrier: threading.Barrier,
    queries: list[str],
    searches_per_user: int,
    library: str,
    kind: str,
    k: int,
) -> VirtualUserResult:
    result = prepared.result
    try:
        barrier.wait()
        for search_id in range(searches_per_user):
            query = queries[(prepared.user_id + search_id) % len(queries)]
            execution = prepared.channels.execute(
                search_code(query, library=library, kind=kind, k=k)
            )
            result.search_roundtrip_s.append(execution.latency_s)
            if not execution.ok:
                result.error_phase = "search"
                raise RuntimeError(execution.error or f"Search failed. Output: {execution.output[-1000:]}")
            try:
                search_data = _extract_result(execution.output, phase="search")
            except ValueError:
                result.error_phase = "search"
                raise RuntimeError(f"Search result missing. Output: {execution.output[-1000:]}")
            result.search_latencies_s.append(float(search_data.get("search_s") or 0.0))
            result.searches_completed += 1
        result.ok = True
    except Exception as exc:
        if not result.error_phase:
            result.error_phase = "search"
        result.error = " ".join(str(exc).split())[:2000]
    return result


def cleanup_users(api: JupyterApi, users: list[PreparedUser]) -> list[str]:
    errors: list[str] = []

    def cleanup(user: PreparedUser) -> None:
        user.channels.close()
        api.delete_kernel(user.kernel_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(users))) as executor:
        futures = {executor.submit(cleanup, user): user for user in users}
        for future, user in [(future, futures[future]) for future in futures]:
            try:
                future.result()
            except Exception as exc:
                errors.append(f"user {user.user_id}: {' '.join(str(exc).split())[:500]}")
    return errors


def summarize_wave(
    results: list[VirtualUserResult],
    before_status: dict[str, Any] | None,
    after_status: dict[str, Any] | None,
) -> dict[str, Any]:
    search_latencies = [
        latency for result in results for latency in result.search_latencies_s
    ]
    return {
        "users": len(results),
        "ok": sum(result.ok for result in results),
        "failed": sum(not result.ok for result in results),
        "kernel_create_s": summarize_values(
            [result.kernel_create_s for result in results if result.kernel_create_s]
        ),
        "setup_s": summarize_values(
            [result.setup_s for result in results if result.setup_s]
        ),
        "model_load_s": summarize_values(
            [result.model_load_s for result in results if result.model_load_s]
        ),
        "search_s": summarize_values(search_latencies),
        "search_roundtrip_s": summarize_values(
            [latency for result in results for latency in result.search_roundtrip_s]
        ),
        "kernel_rss_mb": summarize_values(
            [result.rss_mb for result in results if result.rss_mb]
        ),
        "aggregate_kernel_rss_mb": sum(result.rss_mb for result in results),
        "kernel_pss_mb": summarize_values(
            [result.pss_mb for result in results if result.pss_mb]
        ),
        "aggregate_kernel_pss_mb": sum(result.pss_mb for result in results),
        "kernel_private_mb": summarize_values(
            [result.private_mb for result in results if result.private_mb]
        ),
        "aggregate_kernel_private_mb": sum(result.private_mb for result in results),
        "error_phases": dict(Counter(
            result.error_phase for result in results if result.error_phase
        )),
        "server_status_before": before_status,
        "server_status_after": after_status,
    }


def wave_summary(
    *,
    target_users: int,
    results: list[VirtualUserResult],
    wall_time_s: float,
    search_wall_time_s: float,
    cleanup_errors: list[str],
    before_status: dict[str, Any] | None = None,
    after_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    search_latencies = [
        latency
        for result in results
        for latency in result.search_latencies_s
    ]
    completed = sum(result.searches_completed for result in results)
    summary = summarize_wave(results, before_status, after_status)
    summary.update({
        "target_users": target_users,
        "successful_users": sum(result.ok for result in results),
        "failed_users": sum(not result.ok for result in results),
        "searches_completed": completed,
        "wall_time_s": wall_time_s,
        "search_wall_time_s": search_wall_time_s,
        "searches_per_second": completed / search_wall_time_s if search_wall_time_s else 0.0,
        "kernel_create_latency_s": summarize_values(
            [result.kernel_create_s for result in results if result.kernel_create_s]
        ),
        "setup_latency_s": summarize_values(
            [result.setup_s for result in results if result.setup_s]
        ),
        "search_latency_s": summarize_values(search_latencies),
        "errors": [
            {"user_id": result.user_id, "error": result.error}
            for result in results
            if result.error
        ][:10],
        "cleanup_errors": cleanup_errors[:10],
    })
    return summary


def run_wave(
    api: JupyterApi,
    *,
    users: int,
    kernel_name: str,
    setup_code: str,
    execution_timeout: float,
    queries: list[str],
    searches_per_user: int,
    library: str,
    kind: str,
    k: int,
) -> dict[str, Any]:
    wave_started = time.perf_counter()
    before_status = api.server_status()
    prepared_users: list[PreparedUser] = []
    results: list[VirtualUserResult] = []
    print(f"wave users={users}: creating and warming kernels", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=users) as executor:
        futures = [
            executor.submit(
                prepare_user,
                api,
                user_id=user_id,
                kernel_name=kernel_name,
                setup_code=setup_code,
                execution_timeout=execution_timeout,
                wave=users,
            )
            for user_id in range(users)
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            prepared, result = future.result()
            results.append(result)
            if prepared is not None:
                prepared_users.append(prepared)
            print(
                f"wave users={users}: prepared {completed}/{users} "
                f"(ready={len(prepared_users)})",
                flush=True,
            )

    search_started = time.perf_counter()
    if prepared_users:
        print(
            f"wave users={users}: starting synchronized searches "
            f"with {len(prepared_users)} ready kernels",
            flush=True,
        )
        barrier = threading.Barrier(len(prepared_users))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared_users)) as executor:
            futures = [
                executor.submit(
                    run_searches,
                    prepared,
                    barrier=barrier,
                    queries=queries,
                    searches_per_user=searches_per_user,
                    library=library,
                    kind=kind,
                    k=k,
                )
                for prepared in prepared_users
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    search_wall_time_s = time.perf_counter() - search_started
    cleanup_errors = cleanup_users(api, prepared_users)
    after_status = api.server_status()
    summary = wave_summary(
        target_users=users,
        results=sorted(results, key=lambda result: result.user_id),
        wall_time_s=time.perf_counter() - wave_started,
        search_wall_time_s=search_wall_time_s,
        cleanup_errors=cleanup_errors,
        before_status=before_status,
        after_status=after_status,
    )
    print("WAVE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def token_from_args(args: argparse.Namespace) -> str:
    if args.token_file:
        token = Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    else:
        token = os.getenv(args.token_env, "").strip()
    if not token:
        raise RuntimeError(
            f"Set {args.token_env} or pass --token-file with a short-lived JupyterHub token."
        )
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress a remote Jupyter notebook retrieval workflow with isolated "
            "kernels and synchronized search bursts."
        )
    )
    parser.add_argument(
        "--base-url",
        default="https://localhost:8888/user/stoskopf",
        help="Jupyter single-user server base URL.",
    )
    parser.add_argument("--token-env", default="JUPYTER_TOKEN")
    parser.add_argument("--token-file")
    parser.add_argument("--notebook", default="integral_workshop.ipynb")
    parser.add_argument("--setup-cell", type=int)
    parser.add_argument(
        "--kernel-name",
        help="Kernel spec to use (default: the notebook kernelspec, then python3).",
    )
    parser.add_argument(
        "--users",
        nargs="+",
        default=["5", "10", "20", "40"],
        help="User waves as spaces and/or comma-separated values (default: 5 10 20 40).",
    )
    parser.add_argument("--searches-per-user", type=int, default=3)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--library", default="Coquelicot")
    parser.add_argument("--kind", default="theorem")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--execution-timeout", type=float, default=600.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="Call torch.set_num_threads() before loading the notebook setup cell.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        help="Call torch.set_num_interop_threads() before loading the setup cell.",
    )
    parser.add_argument(
        "--include-install",
        action="store_true",
        help="Execute package-install commands from the setup cell in every kernel.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for a trusted localhost tunnel.",
    )
    parser.add_argument("--ca-bundle")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.users = [wave for group in args.users for wave in parse_waves(group)]
    if any(users <= 0 for users in args.users):
        raise ValueError("All --users values must be positive.")
    if args.searches_per_user <= 0:
        raise ValueError("--searches-per-user must be positive.")
    if args.torch_threads is not None and args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive.")
    if args.torch_interop_threads is not None and args.torch_interop_threads <= 0:
        raise ValueError("--torch-interop-threads must be positive.")
    if args.insecure and args.ca_bundle:
        raise ValueError("Use either --insecure or --ca-bundle, not both.")
    verify_tls: bool | str = not args.insecure
    if args.ca_bundle:
        verify_tls = str(Path(args.ca_bundle).expanduser())
    if args.insecure:
        requests.packages.urllib3.disable_warnings(  # type: ignore[attr-defined]
            requests.packages.urllib3.exceptions.InsecureRequestWarning  # type: ignore[attr-defined]
        )

    api = JupyterApi(
        base_url=args.base_url,
        token=token_from_args(args),
        verify_tls=verify_tls,
        timeout=args.request_timeout,
    )
    server_info = api.server_info()
    notebook = api.notebook(args.notebook)
    kernel_name = args.kernel_name
    if not kernel_name:
        metadata = notebook.get("metadata")
        if isinstance(metadata, dict):
            kernelspec = metadata.get("kernelspec")
            if isinstance(kernelspec, dict):
                kernel_name = str(kernelspec.get("name") or "")
    kernel_name = kernel_name or "python3"
    setup_cell, setup_code = find_setup_code(
        notebook,
        setup_cell=args.setup_cell,
        include_install=args.include_install,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
    )
    queries = args.queries or DEFAULT_QUERIES
    print(
        json.dumps(
            {
                "base_url": args.base_url,
                "server": server_info,
                "notebook": args.notebook,
                "setup_cell": setup_cell,
                "kernel_name": kernel_name,
                "users": args.users,
                "searches_per_user": args.searches_per_user,
                "include_install": args.include_install,
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for wave_index, users in enumerate(args.users):
        summaries.append(
            run_wave(
                api,
                users=users,
                kernel_name=kernel_name,
                setup_code=setup_code,
                execution_timeout=args.execution_timeout,
                queries=queries,
                searches_per_user=args.searches_per_user,
                library=args.library,
                kind=args.kind,
                k=args.k,
            )
        )
        if wave_index + 1 < len(args.users) and args.cooldown > 0:
            print(f"cooldown {args.cooldown:.1f}s", flush=True)
            time.sleep(args.cooldown)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "notebook": args.notebook,
        "waves": summaries,
    }
    print("FINAL_SUMMARY " + json.dumps(report, sort_keys=True), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
