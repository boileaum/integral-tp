#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "qwen/qwen3-embedding-4b"
DEFAULT_HF_DATASET = "theostos/pile-of-rocq"
DEFAULT_HF_CONFIG = "coq-coquelicot-toc_nodes"
DEFAULT_HF_SPLIT = "train"
DEFAULT_LIBRARIES = ("Stdlib", "Coquelicot")
STDLIB_PREFIXES = {"Bignums", "Corelib"}
DEFAULT_INDEX_FACTORY = "Flat"
DEFAULT_INDEX_FILE = "index.faiss"
DEFAULT_EMBEDDINGS_FILE = "embeddings.npy"
PROGRESS_FILE = "embedding_progress.json"
KIND_ALIASES = {
    "start_theorem_proof": "theorem",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _library_from_path(path: str | None) -> str:
    if not path:
        return "Unknown"
    prefix = path.split("/", 1)[0]
    if prefix == "Coquelicot":
        return "Coquelicot"
    if prefix in STDLIB_PREFIXES:
        return "Stdlib"
    if prefix == "mathcomp":
        return "MathComp"
    return prefix or "Unknown"


def _query_prompt_name(model_name: str) -> str | None:
    if model_name.startswith("Qwen/Qwen3-Embedding"):
        return "query"
    return None


def _openrouter_model_name(model_name: str) -> str | None:
    if model_name.lower() == DEFAULT_EMBEDDING_MODEL.lower():
        return DEFAULT_OPENROUTER_EMBEDDING_MODEL
    return None


def _normalize_library_name(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "coq": "Stdlib",
        "coq-stdlib": "Stdlib",
        "corelib": "Stdlib",
        "bignums": "Stdlib",
        "stdlib": "Stdlib",
        "std": "Stdlib",
        "coquelicot": "Coquelicot",
        "mathcomp": "MathComp",
        "math-comp": "MathComp",
    }
    return aliases.get(key, name.strip())


def _normalize_kind_name(name: object) -> str:
    key = str(name or "").strip().lower()
    return KIND_ALIASES.get(key, key)


def _normalize_libraries(values: list[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_LIBRARIES)
    out: set[str] = set()
    for value in values:
        for raw in value.split(","):
            raw = raw.strip()
            if raw:
                out.add(_normalize_library_name(raw))
    if not out:
        return set(DEFAULT_LIBRARIES)
    return out


def _library_allowed(entry: dict[str, Any], libraries: set[str]) -> bool:
    if not libraries:
        return True
    return str(entry.get("library") or "Unknown") in libraries


def _load_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _entry_from_hf_toc_row(row: dict[str, Any]) -> dict[str, Any] | None:
    extra = _load_json_object(row.get("data_extra_json"))
    source_path = row.get("source_path_canonical")
    docstring = row.get("docstring")
    content = extra.get("content") or row.get("content") or row.get("statement")
    name = row.get("name") or extra.get("name") or row.get("data_uid")
    if not any(isinstance(value, str) and value.strip() for value in (docstring, content, name)):
        return None
    span_bp = row.get("span_bp")
    span_ep = row.get("span_ep")
    return {
        "uid": row.get("data_uid") or extra.get("uid"),
        "data_uid": row.get("data_uid"),
        "name": name,
        "kind": _normalize_kind_name(row.get("kind")),
        "range": [span_bp, span_ep] if span_bp is not None and span_ep is not None else None,
        "docstring": docstring,
        "content": content,
        "source": source_path,
        "library": _library_from_path(source_path),
        "env_id": row.get("env_id"),
        "source_id": row.get("source_id"),
        "preorder_index": row.get("preorder_index"),
        "depth": row.get("depth"),
    }


def _text_for_embedding(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("name") or ""),
        str(entry.get("kind") or ""),
        str(entry.get("content") or ""),
        str(entry.get("docstring") or ""),
        str(entry.get("source") or ""),
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _stable_uid(entry: dict[str, Any]) -> str:
    text = json.dumps(
        {
            "uid": entry.get("uid"),
            "library": entry.get("library"),
            "source": entry.get("source"),
            "name": entry.get("name"),
            "kind": entry.get("kind"),
            "content": entry.get("content"),
            "docstring": entry.get("docstring"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _entries_fingerprint(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry.get("uid") or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_entries(
    entries: list[dict[str, Any]],
    *,
    libraries: set[str],
    docstrings_only: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry.get("library"):
            entry = dict(entry)
            entry["library"] = _library_from_path(str(entry.get("source") or ""))
        if not _library_allowed(entry, libraries):
            continue
        docstring = entry.get("docstring")
        content = entry.get("content")
        if docstrings_only and not (isinstance(docstring, str) and docstring.strip()):
            continue
        if not any(isinstance(value, str) and value.strip() for value in (docstring, content, entry.get("name"))):
            continue
        uid = _stable_uid(entry)
        if uid in seen:
            continue
        seen.add(uid)
        normalized = dict(entry)
        normalized["uid"] = uid
        normalized["text"] = _text_for_embedding(normalized)
        out.append(normalized)
    return out


def collect_hf_entries(
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    libraries: set[str],
    docstrings_only: bool,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Loading from Hugging Face requires `datasets`. "
            "Install with: pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, config_name, split=split)
    entries: list[dict[str, Any]] = []
    for row in dataset:
        entry = _entry_from_hf_toc_row(dict(row))
        if entry is not None:
            entries.append(entry)
    return _normalize_entries(entries, libraries=libraries, docstrings_only=docstrings_only)


def _load_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = _utc_now()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def encode_embeddings(
    *,
    entries: list[dict[str, Any]],
    output_dir: Path,
    embeddings_file: str,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    resume: bool,
) -> tuple[Any, int]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Embedding computation requires `numpy` and `sentence-transformers`.\n"
            "Install with: pip install 'integral-tp[retrieval]'\n"
            "or: pip install datasets 'sentence-transformers>=2.7.0' 'transformers>=4.51.0'"
        ) from exc

    embeddings_path = output_dir / embeddings_file
    progress_path = output_dir / PROGRESS_FILE
    fingerprint = _entries_fingerprint(entries)
    model = SentenceTransformer(model_name)
    completed = 0
    embeddings = None

    if resume and embeddings_path.exists():
        progress = _load_progress(progress_path)
        if progress is None:
            raise RuntimeError(f"Cannot resume: {progress_path} is missing.")
        expected = {
            "model_name": model_name,
            "entry_count": len(entries),
            "entries_fingerprint": fingerprint,
            "normalize_embeddings": normalize_embeddings,
            "embeddings_file": embeddings_file,
        }
        mismatches = [
            f"{key}: expected {value!r}, found {progress.get(key)!r}"
            for key, value in expected.items()
            if progress.get(key) != value
        ]
        if mismatches:
            raise RuntimeError("Cannot resume embeddings; progress metadata mismatch:\n" + "\n".join(mismatches))
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        completed = int(progress.get("completed", 0))
        if embeddings.shape[0] != len(entries):
            raise RuntimeError(
                f"Cannot resume: {embeddings_path} has {embeddings.shape[0]} rows, expected {len(entries)}."
            )
        completed = min(completed, len(entries))
        print(f"resuming embeddings at row {completed}/{len(entries)}")

    while completed < len(entries):
        stop = min(completed + batch_size, len(entries))
        batch_texts = [entry["text"] for entry in entries[completed:stop]]
        batch = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        ).astype("float32")
        if batch.ndim != 2:
            raise ValueError(f"Unexpected embedding batch shape: {batch.shape}")
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype="float32",
                shape=(len(entries), int(batch.shape[1])),
            )
        if embeddings.shape[1] != batch.shape[1]:
            raise ValueError(
                f"Embedding dimension changed: memmap has {embeddings.shape[1]}, "
                f"batch has {batch.shape[1]}."
            )
        embeddings[completed:stop] = batch
        embeddings.flush()
        completed = stop
        _write_progress(
            progress_path,
            {
                "version": 1,
                "model_name": model_name,
                "entry_count": len(entries),
                "entries_fingerprint": fingerprint,
                "normalize_embeddings": normalize_embeddings,
                "embeddings_file": embeddings_file,
                "dimension": int(embeddings.shape[1]),
                "completed": completed,
            },
        )
        print(f"embedded {completed}/{len(entries)}")

    if embeddings is None:
        raise ValueError("No embeddings were produced.")
    return embeddings, int(embeddings.shape[1])


def build_faiss_index(
    *,
    embeddings: Any,
    output_dir: Path,
    index_file: str,
    index_factory: str,
) -> None:
    try:
        import faiss
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "FAISS index building requires `faiss-cpu` and `numpy`.\n"
            "Install with: pip install 'integral-tp[retrieval]'"
        ) from exc

    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError(f"Unexpected embeddings shape: {vectors.shape}")
    dim = int(vectors.shape[1])
    if index_factory == "Flat":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, str(output_dir / index_file))


def build_cache(
    *,
    entries: list[dict[str, Any]],
    output_dir: Path,
    model_name: str,
    batch_size: int,
    source_roots: list[str],
    libraries: set[str],
    index_factory: str,
    index_file: str,
    embeddings_file: str,
    resume: bool,
    dataset_source: dict[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_embeddings = True
    embeddings, dim = encode_embeddings(
        entries=entries,
        output_dir=output_dir,
        embeddings_file=embeddings_file,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        resume=resume,
    )
    build_faiss_index(
        embeddings=embeddings,
        output_dir=output_dir,
        index_file=index_file,
        index_factory=index_factory,
    )

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            stored = {key: value for key, value in entry.items() if key != "text"}
            handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "version": 1,
        "generated_at": _utc_now(),
        "model_name": model_name,
        "entry_count": len(entries),
        "dimension": dim,
        "metric": "cosine",
        "normalize_embeddings": normalize_embeddings,
        "source_roots": source_roots,
        "libraries": sorted(libraries),
        "index_file": index_file,
        "index_factory": index_factory,
        "embeddings_file": embeddings_file,
        "embeddings_dtype": "float32",
        "entries_fingerprint": _entries_fingerprint(entries),
    }
    query_prompt_name = _query_prompt_name(model_name)
    if query_prompt_name is not None:
        manifest["query_prompt_name"] = query_prompt_name
    openrouter_model_name = _openrouter_model_name(model_name)
    if openrouter_model_name is not None:
        manifest["openrouter_model_name"] = openrouter_model_name
        manifest["openrouter_query_input_type"] = "search_query"
    if dataset_source is not None:
        manifest["dataset_source"] = dataset_source
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def zip_cache(output_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = zip_path.resolve()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path:
                archive.write(path, arcname=path.relative_to(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local FAISS docstring retrieval cache for the Rocq workshop."
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--hf-config",
        default=DEFAULT_HF_CONFIG,
        help="Hugging Face dataset config.",
    )
    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help="Hugging Face dataset split.",
    )
    parser.add_argument(
        "--library",
        action="append",
        default=[],
        help=(
            "Library to include in the cache. Repeatable or comma-separated. "
            "Defaults to Stdlib and Coquelicot. Supported aliases include "
            "Corelib/Bignums for Stdlib."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="build/retrieval_cache",
        help="Directory where manifest.json, metadata.jsonl and index.faiss are written.",
    )
    parser.add_argument(
        "--zip",
        default="build/retrieval_cache.zip",
        help="Zip archive to publish/download in Colab.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model used for both index and query embeddings.",
    )
    parser.add_argument(
        "--index-factory",
        default=DEFAULT_INDEX_FACTORY,
        help=(
            "FAISS index factory string. Default `Flat` builds exact IndexFlatIP. "
            "Approximate strategies can be tested later, e.g. `HNSW32` or `IVF128,Flat`."
        ),
    )
    parser.add_argument(
        "--index-file",
        default=DEFAULT_INDEX_FILE,
        help="FAISS index filename written inside --output-dir.",
    )
    parser.add_argument(
        "--embeddings-file",
        default=DEFAULT_EMBEDDINGS_FILE,
        help="NumPy .npy filename for the normalized float32 embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means no limit.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume embedding computation from an existing embeddings file and progress metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete --output-dir before building. Not allowed together with --resume.",
    )
    parser.add_argument(
        "--docstrings-only",
        action="store_true",
        default=None,
        help="Drop entries that have content but no docstring. Enabled by default.",
    )
    parser.add_argument(
        "--include-undocumented",
        action="store_true",
        help="Also embed entries without docstrings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force and args.resume:
        raise SystemExit("--force and --resume cannot be used together.")
    libraries = _normalize_libraries(args.library)
    docstrings_only = bool(args.docstrings_only) if args.docstrings_only is not None else not bool(args.include_undocumented)
    entries = collect_hf_entries(
        dataset_name=args.hf_dataset,
        config_name=args.hf_config,
        split=args.hf_split,
        libraries=libraries,
        docstrings_only=docstrings_only,
    )
    dataset_source = {
        "dataset": args.hf_dataset,
        "config": args.hf_config,
        "split": args.hf_split,
    }
    source_roots = [f"hf://datasets/{args.hf_dataset}/{args.hf_config}/{args.hf_split}"]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("No entries found. Check the dataset/config/split and --library filters.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    zip_path = Path(args.zip).expanduser().resolve()
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"{output_dir} already exists and is not empty. Use --resume to continue "
            "embedding computation, or --force to delete it before rebuilding."
        )
    build_cache(
        entries=entries,
        output_dir=output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        source_roots=source_roots,
        libraries=libraries,
        index_factory=args.index_factory,
        index_file=args.index_file,
        embeddings_file=args.embeddings_file,
        resume=bool(args.resume),
        dataset_source=dataset_source,
    )
    zip_cache(output_dir, zip_path)
    by_library: dict[str, int] = {}
    for entry in entries:
        library = str(entry.get("library") or "Unknown")
        by_library[library] = by_library.get(library, 0) + 1
    print(f"entries: {len(entries)}")
    for library, count in sorted(by_library.items()):
        print(f"  {library}: {count}")
    print(f"cache:   {output_dir}")
    print(f"zip:     {zip_path}")


if __name__ == "__main__":
    main()
