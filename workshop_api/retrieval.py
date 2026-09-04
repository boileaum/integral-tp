from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin

import requests


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "qwen/qwen3-embedding-4b"
KIND_ALIASES = {
    "start_theorem_proof": "theorem",
}


def _canonical_embedding_model_name(model_name: str) -> str:
    """Normalize equivalent Hugging Face/OpenRouter model identifiers."""

    return model_name.strip().lower().replace("_", "-")


def _openrouter_model_for_cache(model_name: str) -> str | None:
    aliases = {
        _canonical_embedding_model_name(DEFAULT_EMBEDDING_MODEL): (
            DEFAULT_OPENROUTER_EMBEDDING_MODEL
        ),
        _canonical_embedding_model_name(DEFAULT_OPENROUTER_EMBEDDING_MODEL): (
            DEFAULT_OPENROUTER_EMBEDDING_MODEL
        ),
    }
    return aliases.get(_canonical_embedding_model_name(model_name))


@dataclass
class RemoteEmbeddingClient:
    """OpenAI-compatible embeddings client used through the workshop proxy."""

    server_url: str
    model_name: str
    server_token: str | None = None
    timeout: float = 120.0

    def encode(
        self,
        texts: Sequence[str],
        *,
        input_type: str | None = None,
    ) -> list[list[float]]:
        inputs = [str(text) for text in texts]
        if not inputs or any(not text.strip() for text in inputs):
            raise ValueError("Embedding inputs must contain non-empty text.")

        payload: dict[str, Any] = {
            "input": inputs,
            "model": self.model_name,
            "encoding_format": "float",
        }
        if input_type:
            payload["input_type"] = input_type
        headers = {"Content-Type": "application/json"}
        if self.server_token:
            headers["Authorization"] = f"Bearer {self.server_token}"

        response = requests.post(
            urljoin(self.server_url.rstrip("/") + "/", "embeddings"),
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(inputs):
            raise ValueError("Embedding server returned an invalid number of vectors.")

        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Embedding server returned an invalid vector.")
            vectors.append([float(value) for value in embedding])
        return vectors


def normalize_kind_name(name: object) -> str:
    key = str(name or "").strip().lower()
    return KIND_ALIASES.get(key, key)


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


def _normalize_libraries(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    raw_values = [value] if isinstance(value, str) else list(value)
    out: set[str] = set()
    for item in raw_values:
        for raw in str(item).split(","):
            raw = raw.strip()
            if raw:
                out.add(_normalize_library_name(raw))
    return out or None


def _normalize_kinds(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    raw_values = [value] if isinstance(value, str) else list(value)
    out: set[str] = set()
    for item in raw_values:
        for raw in str(item).split(","):
            raw = raw.strip().lower()
            if raw:
                out.add(normalize_kind_name(raw))
    return out or None


def _library_matches(item: dict[str, Any], libraries: set[str] | None) -> bool:
    if libraries is None:
        return True
    library = item.get("library")
    if not isinstance(library, str) or not library.strip():
        return False
    return _normalize_library_name(library) in libraries


def _kind_matches(item: dict[str, Any], kinds: set[str] | None) -> bool:
    if kinds is None:
        return True
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        return False
    return normalize_kind_name(kind) in kinds


def _normalize_hit_kind(item: dict[str, Any]) -> dict[str, Any]:
    kind = item.get("kind")
    if not isinstance(kind, str):
        return item
    normalized = normalize_kind_name(kind)
    if normalized == kind:
        return item
    out = dict(item)
    out["kind"] = normalized
    return out


def _shorten(text: object, *, limit: int) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(limit - 3, 0)].rstrip() + "..."


def format_retrieval_hits(
    hits: list[dict[str, Any]],
    *,
    statement_chars: int = 500,
    docstring_chars: int = 260,
) -> str:
    """Render retrieval hits as workshop-friendly plain text."""

    if not hits:
        return "No retrieval hits."

    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        name = hit.get("name") or hit.get("uid") or "<unnamed>"
        kind = normalize_kind_name(hit.get("kind"))
        library = hit.get("library")
        source = hit.get("source")
        score = hit.get("score")
        heading_parts = [f"{index}. {name}"]
        if kind:
            heading_parts.append(f"({kind})")
        if score is not None:
            try:
                heading_parts.append(f"[score {float(score):.3f}]")
            except (TypeError, ValueError):
                heading_parts.append(f"[score {score}]")
        lines.append(" ".join(heading_parts))
        if library:
            lines.append(f"   library: {_shorten(library, limit=80)}")
        if source:
            lines.append(f"   source: {_shorten(source, limit=140)}")

        statement = hit.get("statement") or hit.get("content")
        if statement:
            lines.append(f"   statement: {_shorten(statement, limit=statement_chars)}")

        docstring = hit.get("docstring")
        if docstring:
            lines.append(f"   docstring: {_shorten(docstring, limit=docstring_chars)}")
    return "\n".join(lines)


def _default_cache_parent() -> Path:
    if _running_in_colab():
        return Path("/content/.cache/integral-tp")
    return Path(os.getenv("WORKSHOP_CACHE_DIR", "~/.cache/integral-tp")).expanduser()


def _running_in_colab() -> bool:
    return "google.colab" in sys.modules or bool(os.getenv("COLAB_RELEASE_TAG"))


def _resolve_cache_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "manifest.json").exists():
        return path
    children = [child for child in path.iterdir() if child.is_dir()] if path.exists() else []
    matches = [child for child in children if (child / "manifest.json").exists()]
    if len(matches) == 1:
        return matches[0].resolve()
    return path


def download_retrieval_cache(
    cache_url: str,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download and unpack a retrieval cache zip.

    `cache_url` can be a direct HTTP(S) zip URL, a local zip path, or a local
    directory. The returned directory contains `manifest.json`, `metadata.jsonl`,
    and `index.faiss`.
    """

    if not cache_url:
        raise ValueError("cache_url is empty.")

    raw = cache_url.strip()
    if raw.startswith("file://"):
        raw = raw[len("file://") :]

    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_dir():
        return _resolve_cache_dir(candidate)

    target = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_parent() / "retrieval"
    if target.exists() and force:
        shutil.rmtree(target)
    resolved = _resolve_cache_dir(target) if target.exists() else target
    if (resolved / "manifest.json").exists() and not force:
        return resolved

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "retrieval_cache.zip"

    if candidate.exists() and candidate.is_file():
        shutil.copyfile(candidate, zip_path)
    else:
        with requests.get(raw, stream=True, timeout=120) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target)
    except zipfile.BadZipFile as exc:
        raise ValueError(
            f"Downloaded file is not a valid zip: {zip_path}. "
            "Use a direct download URL for retrieval_cache.zip; public preview pages "
            "from cloud drives usually download HTML instead of the archive."
        ) from exc
    zip_path.unlink(missing_ok=True)
    resolved = _resolve_cache_dir(target)
    if not (resolved / "manifest.json").exists():
        raise FileNotFoundError(
            f"Downloaded cache did not contain manifest.json at {target}. "
            "The zip should contain manifest.json, metadata.jsonl, and index.faiss."
        )
    return resolved


def prepare_colab_retrieval_cache(
    cache_url: str | None = None,
    *,
    cache_dir: str | Path = "/content/rocq-doc-cache",
    force: bool = False,
) -> Path:
    """Prepare the retrieval cache inside the Colab runtime.

    In the workshop notebook, the FAISS index is meant to be local to Colab:
    the cache zip is downloaded once into `/content`, unpacked, and queried
    without recomputing library embeddings.
    """

    explicit_dir = os.getenv("DOCSTRING_CACHE_DIR")
    if explicit_dir and not cache_url:
        return _resolve_cache_dir(Path(explicit_dir))

    url = cache_url or os.getenv("DOCSTRING_CACHE_URL")
    if not url:
        raise ValueError(
            "No retrieval cache URL configured. Set DOCSTRING_CACHE_URL to the "
            "public zip URL, or set DOCSTRING_CACHE_DIR to an already-unpacked cache."
        )
    cache_path = download_retrieval_cache(url, cache_dir=cache_dir, force=force)
    os.environ["DOCSTRING_CACHE_DIR"] = str(cache_path)
    return cache_path


@dataclass
class LocalFaissRetriever:
    cache_dir: str | Path
    model_name: str | None = None
    embedding_server_url: str | None = None
    embedding_server_token: str | None = None
    remote_model_name: str | None = None
    embedding_timeout: float = 120.0
    top_k_multiplier: int = 4

    def __post_init__(self) -> None:
        try:
            import faiss
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency error path.
            raise RuntimeError(
                "Retrieval requires `faiss-cpu` and `numpy`. Install "
                "`integral-tp[colab]`, or run `pip install -q faiss-cpu numpy`."
            ) from exc

        self._faiss = faiss
        self._np = np
        self.cache_dir = _resolve_cache_dir(Path(self.cache_dir))
        manifest_path = self.cache_dir / "manifest.json"
        metadata_path = self.cache_dir / "metadata.jsonl"
        if not manifest_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Invalid retrieval cache at {self.cache_dir}; expected manifest.json, "
                "metadata.jsonl, and the manifest index file."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = self.cache_dir / str(self.manifest.get("index_file", "index.faiss"))
        if not index_path.exists():
            raise FileNotFoundError(f"Retrieval cache manifest points to missing index file: {index_path}")
        manifest_model = self.manifest.get("model_name")
        if self.model_name and manifest_model and self.model_name != manifest_model:
            raise ValueError(
                f"Embedding model mismatch: cache manifest uses {manifest_model!r}, "
                f"but query model was set to {self.model_name!r}."
            )
        self.model_name = self.model_name or manifest_model or DEFAULT_EMBEDDING_MODEL
        self.query_prompt_name = self.manifest.get("query_prompt_name")
        self.metadata = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index/metadata mismatch: index has {self.index.ntotal}, "
                f"metadata has {len(self.metadata)}."
            )
        self.remote_embedder: RemoteEmbeddingClient | None = None
        self.model: Any | None = None
        if self.embedding_server_url:
            expected_remote_model = _openrouter_model_for_cache(self.model_name)
            manifest_remote_model = self.manifest.get("openrouter_model_name")
            if isinstance(manifest_remote_model, str) and manifest_remote_model.strip():
                expected_remote_model = manifest_remote_model.strip()
            active_remote_model = self.remote_model_name or expected_remote_model
            if not active_remote_model:
                raise ValueError(
                    f"Cache model {self.model_name!r} has no known OpenRouter equivalent. "
                    "Set DOCSTRING_OPENROUTER_EMBEDDING_MODEL only after confirming that "
                    "it produces vectors compatible with this cache."
                )
            if expected_remote_model and (
                _canonical_embedding_model_name(active_remote_model)
                != _canonical_embedding_model_name(expected_remote_model)
            ):
                raise ValueError(
                    f"Remote embedding model mismatch: cache expects "
                    f"{expected_remote_model!r}, but the client was configured with "
                    f"{active_remote_model!r}."
                )
            self.remote_model_name = active_remote_model
            self.remote_embedder = RemoteEmbeddingClient(
                server_url=self.embedding_server_url,
                server_token=self.embedding_server_token,
                model_name=active_remote_model,
                timeout=self.embedding_timeout,
            )
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - dependency error path.
                raise RuntimeError(
                    "Local query embeddings require `sentence-transformers`. Install "
                    "`integral-tp[retrieval]`, configure WORKSHOP_EMBEDDING_SERVER_URL, "
                    "or run `pip install -q 'sentence-transformers>=2.7.0' "
                    "'transformers>=4.51.0'`."
                ) from exc
            self.model = SentenceTransformer(self.model_name)

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        library: str | Sequence[str] | None = None,
        kind: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        k = int(k)
        if not query or k <= 0:
            return []
        libraries = _normalize_libraries(library)
        kinds = _normalize_kinds(kind)
        normalize_embeddings = bool(self.manifest.get("normalize_embeddings", True))
        if self.remote_embedder is not None:
            query_input_type = self.manifest.get("openrouter_query_input_type")
            if not isinstance(query_input_type, str) or not query_input_type.strip():
                query_input_type = "search_query"
            vector = self._np.asarray(
                self.remote_embedder.encode([query], input_type=query_input_type),
                dtype="float32",
            )
            if normalize_embeddings:
                norms = self._np.linalg.norm(vector, axis=1, keepdims=True)
                if self._np.any(norms == 0):
                    raise ValueError("Embedding server returned a zero-length query vector.")
                vector = vector / norms
        else:
            assert self.model is not None
            encode_kwargs: dict[str, Any] = {}
            if isinstance(self.query_prompt_name, str) and self.query_prompt_name.strip():
                encode_kwargs["prompt_name"] = self.query_prompt_name
            vector = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
                **encode_kwargs,
            ).astype("float32")
        if vector.ndim != 2 or vector.shape[0] != 1 or vector.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding shape {vector.shape} does not match FAISS index "
                f"dimension {self.index.d}."
            )
        search_k = self.index.ntotal if libraries is not None or kinds is not None else max(k * self.top_k_multiplier, k)
        scores, ids = self.index.search(vector, min(int(search_k), int(self.index.ntotal)))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0:
                continue
            item = _normalize_hit_kind(dict(self.metadata[int(idx)]))
            if not _library_matches(item, libraries):
                continue
            if not _kind_matches(item, kinds):
                continue
            key = str(item.get("uid") or (item.get("source"), item.get("name"), item.get("content")))
            if key in seen:
                continue
            seen.add(key)
            item["score"] = float(score)
            out.append(item)
            if len(out) >= k:
                break
        return out


@dataclass
class RetrievalClient:
    """FAISS docstring retriever with local or proxy-side query embeddings.

    The FAISS index and metadata stay local to the notebook. Query embeddings
    can be computed by the workshop proxy, so participant kernels do not each
    load a multi-gigabyte model. Configure either `cache_dir` or `cache_url`;
    otherwise the first search raises a clear setup error. Pass `library="Stdlib"` or
    `library="Coquelicot"` to restrict matches to one source library. Pass
    `kind="definition"` or `kind=["definition", "theorem"]` to
    restrict matches by Rocq element kind.
    """

    cache_dir: str | Path | None = None
    cache_url: str | None = None
    model_name: str | None = None
    embedding_server_url: str | None = None
    embedding_server_token: str | None = None
    remote_model_name: str | None = None
    embedding_timeout: float = 120.0
    libraries: str | Sequence[str] | None = None
    kinds: str | Sequence[str] | None = None
    _local: LocalFaissRetriever | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        cache_url: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "RetrievalClient":
        embedding_server_url = (
            os.getenv("WORKSHOP_EMBEDDING_SERVER_URL")
            or os.getenv("WORKSHOP_LLM_SERVER_URL")
            or os.getenv("LLM_SERVER_URL")
            or None
        )
        return cls(
            cache_url=cache_url or os.getenv("DOCSTRING_CACHE_URL") or None,
            cache_dir=cache_dir or os.getenv("DOCSTRING_CACHE_DIR") or None,
            model_name=os.getenv("DOCSTRING_EMBEDDING_MODEL") or None,
            embedding_server_url=embedding_server_url,
            embedding_server_token=(
                os.getenv("WORKSHOP_EMBEDDING_SERVER_TOKEN")
                or os.getenv("WORKSHOP_LLM_SERVER_TOKEN")
                or os.getenv("LLM_SERVER_TOKEN")
                or None
            ),
            remote_model_name=(
                os.getenv("DOCSTRING_OPENROUTER_EMBEDDING_MODEL") or None
            ),
            embedding_timeout=float(
                os.getenv("WORKSHOP_EMBEDDING_TIMEOUT_SECONDS", "120")
            ),
            libraries=os.getenv("DOCSTRING_LIBRARIES") or None,
            kinds=os.getenv("DOCSTRING_KINDS") or None,
        )

    def _local_retriever(self) -> LocalFaissRetriever:
        if self._local is not None:
            return self._local
        cache_dir = self.cache_dir
        if cache_dir is None and self.cache_url:
            cache_dir = download_retrieval_cache(self.cache_url)
        if cache_dir is None:
            raise ValueError(
                "Local retrieval cache is not configured. Set DOCSTRING_CACHE_URL "
                "to a direct retrieval_cache.zip URL, set DOCSTRING_CACHE_DIR to an "
                "unpacked cache directory, or pass cache_url/cache_dir explicitly."
            )
        self._local = LocalFaissRetriever(
            cache_dir=cache_dir,
            model_name=self.model_name,
            embedding_server_url=self.embedding_server_url,
            embedding_server_token=self.embedding_server_token,
            remote_model_name=self.remote_model_name,
            embedding_timeout=self.embedding_timeout,
        )
        return self._local

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        library: str | Sequence[str] | None = None,
        kind: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        active_library = library if library is not None else self.libraries
        active_kind = kind if kind is not None else self.kinds
        return self._local_retriever().search(query, k=k, library=active_library, kind=active_kind)
