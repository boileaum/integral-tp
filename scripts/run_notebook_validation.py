from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the workshop notebook as one successful participant."
    )
    parser.add_argument("--notebook", default="integral_workshop.ipynb")
    parser.add_argument("--output", default="/tmp/integral_workshop_local_executed.ipynb")
    parser.add_argument("--rocq-url", default="http://127.0.0.1:5000")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8010")
    parser.add_argument("--cache-dir", default="/tmp/integral-tp-cache")
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    source = Path(args.notebook).resolve()
    output = Path(args.output).resolve()
    notebook = nbformat.read(source, as_version=4)
    replacements = {
        "/scratch/integral-tp/rocq-doc-cache": args.cache_dir,
        "http://integral-tp.math.unistra.fr:5000": args.rocq_url,
        "http://integral-tp.math.unistra.fr:8010": args.llm_url,
        "  [TO FILL].": "  is_derive F2 x ((sech (10 * x - 2)) ^ 2).",
    }
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in replacements.items():
            cell.source = cell.source.replace(old, new)

    notebook.metadata.kernelspec.name = args.kernel
    notebook.metadata.kernelspec.display_name = "Python (local TP validation)"
    started = time.perf_counter()

    def on_cell_start(cell, cell_index, **_kwargs):
        first = cell.source.splitlines()[0] if cell.source.splitlines() else ""
        print(f"[notebook] cell={cell_index} start {first[:90]}", flush=True)

    def on_cell_complete(cell, cell_index, **_kwargs):
        del cell
        elapsed = time.perf_counter() - started
        print(f"[notebook] cell={cell_index} complete elapsed={elapsed:.1f}s", flush=True)

    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(source.parent)}},
        on_cell_start=on_cell_start,
        on_cell_complete=on_cell_complete,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        elapsed = time.perf_counter() - started
        print(f"[notebook] saved={output} elapsed={elapsed:.1f}s", flush=True)
    print("[notebook] FULL_EXECUTION_OK", flush=True)


if __name__ == "__main__":
    main()
