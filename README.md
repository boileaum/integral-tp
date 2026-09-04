# integral-tp

Hands-on Rocq workshop material for certified numerical analysis.

## Participant Quick Link

Open the workshop notebook in Colab:

<a target="_blank" href="https://colab.research.google.com/github/theostos/integral-tp/blob/main/integral_workshop.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The notebook expects the Rocq server and the LLM proxy to be running before the
session starts.

## Start The Servers

Start the Rocq inference server:

```bash
docker run --rm -it --network host \
  theostos/coq-tp:8.20 \
  rocq-ml-server
```

Start the LLM proxy in another terminal, so participants never see the
OpenRouter API key:

```bash
export WORKSHOP_LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="z-ai/glm-5.3-flash"
export OPENROUTER_EMBEDDING_MODEL="qwen/qwen3-embedding-4b"
export WORKSHOP_LLM_SERVER_CONCURRENCY=8
export WORKSHOP_LLM_SERVER_WORKERS=8
export WORKSHOP_LLM_SERVER_MIN_INTERVAL_SECONDS=0.1
export WORKSHOP_EMBEDDING_SERVER_CONCURRENCY=8
export WORKSHOP_EMBEDDING_SERVER_MIN_INTERVAL_SECONDS=0.1
export WORKSHOP_EMBEDDING_SERVER_MAX_RETRIES=8
integral-tp-llm-server --host 0.0.0.0 --port 8010
```

For a Colab test on the free ngrok plan, expose one local reverse proxy and
route paths to the two services:

```caddyfile
:8080 {
  handle_path /llm/* {
    reverse_proxy 127.0.0.1:8010
  }

  handle_path /rocq/* {
    reverse_proxy 127.0.0.1:5000
  }
}
```

```bash
caddy run --config Caddyfile
ngrok http 8080
```

In the first notebook cell, set:

```python
os.environ["ROCQ_SERVER_URL"] = "https://<ngrok-host>/rocq"
os.environ["WORKSHOP_LLM_SERVER_URL"] = "https://<ngrok-host>/llm"
```

## Files

- `integral.v`: compact reference development.
- `workshop_api/`: small Python API over `rocq-ml-server` for adding Rocq elements, opening lemmas, running tactics, local retrieval hooks, and optional LLM calls.
- `integral_workshop.ipynb`: 90-minute notebook decomposing `integral.v` step by step.
- `img/`: legacy screenshots from the former Colab GPU setup path.
- `scripts/build_retrieval_cache.py`: offline builder for the FAISS docstring retrieval cache.
- `pyproject.toml`: package metadata and Colab extras used by the notebook setup.
- `requirements-colab.txt`: one-line remote install spec for Colab.

## Colab Run

The intended workshop notebook runs in Google Colab. In this setup, "local"
retrieval means local to the Colab runtime: the precomputed FAISS cache is
downloaded into `/content` and queried there.

Open `integral_workshop.ipynb` in Colab from this repository. The setup cells:

- install `integral-tp[colab]` from GitHub with `%pip`;
- connect to a remote `rocq-ml-server` endpoint;
- download the FAISS retrieval cache into `/content`;
- then all Rocq interaction goes through that remote endpoint. The FAISS index
  stays in the notebook process, while query embeddings are computed through
  the OpenRouter proxy so participant kernels do not load the 4B model.

The install spec used by default is:

```bash
integral-tp[colab] @ git+https://github.com/theostos/integral-tp.git
```

The notebook uses this Colab cell:

```python
%pip install -q integral-tp[colab]@git+https://github.com/theostos/integral-tp.git
```

The `colab` extra depends on `rocq-ml-toolbox[client]` from the public
`llm4rocq/rocq-ml-toolbox` repository, plus public `pytanque` and retrieval
packages. The Rocq server stack still comes from the Docker image.

For a branch, fork, or local development variant, edit the `%pip install` line
in the notebook.

Run the Rocq server once on a remote machine with Docker, then point all Colab
notebooks at that host:

```bash
docker run -d -p 5000:5000 \
  theostos/coq-tp:8.20 \
  rocq-ml-server --host 0.0.0.0 --port 5000 \
    --num-pet-server 16 --workers 16 --timeout 600
```

In the notebook, configure:

```python
os.environ["ROCQ_SERVER_HOST"] = "rocq-workshop.example.org"
os.environ["ROCQ_SERVER_PORT"] = "5000"
```

If the server is behind a path-routing proxy, use the URL form instead:

```python
os.environ["ROCQ_SERVER_URL"] = "https://workshop.example.org/rocq"
```

For a larger group, run several server instances on different ports or
machines and assign participants to endpoints.

## LLM Proxy

Do not put the OpenRouter API key in participant notebooks. Run the small proxy
on the workshop server instead:

```bash
export WORKSHOP_LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="z-ai/glm-5.3-flash"
export OPENROUTER_EMBEDDING_MODEL="qwen/qwen3-embedding-4b"
export WORKSHOP_LLM_SERVER_CONCURRENCY=8
export WORKSHOP_LLM_SERVER_WORKERS=8
export WORKSHOP_LLM_SERVER_MIN_INTERVAL_SECONDS=0.1
export WORKSHOP_LLM_SERVER_MAX_RETRIES=8
export WORKSHOP_LLM_SERVER_QUEUE_SIZE=500
export WORKSHOP_EMBEDDING_SERVER_CONCURRENCY=8
export WORKSHOP_EMBEDDING_SERVER_MIN_INTERVAL_SECONDS=0.1
export WORKSHOP_EMBEDDING_SERVER_MAX_RETRIES=8
integral-tp-llm-server --host 0.0.0.0 --port 8010
```

The proxy accepts participant requests immediately, puts them in an internal
queue, and lets a small pool of workers call OpenRouter with global pacing.
When the upstream provider returns a transient error such as `429 Rate limit
exceeded`, the proxy backs off and retries the job before reporting a failure
to the notebook.

Create the OpenRouter key from <https://openrouter.ai/settings/keys>. Add
credits or a payment method in the OpenRouter dashboard before the workshop,
then store the key only on the proxy machine as `OPENROUTER_API_KEY`.

Useful tuning variables:

```bash
export WORKSHOP_LLM_SERVER_CONCURRENCY=8          # simultaneous upstream calls
export WORKSHOP_LLM_SERVER_WORKERS=8
export WORKSHOP_LLM_SERVER_MIN_INTERVAL_SECONDS=0.1
export WORKSHOP_LLM_SERVER_MAX_RETRIES=8
export WORKSHOP_LLM_SERVER_RATE_LIMIT_BACKOFF_INITIAL_SECONDS=10
export WORKSHOP_LLM_SERVER_BACKOFF_MAX_SECONDS=120
export WORKSHOP_LLM_SERVER_QUEUE_SIZE=500
export WORKSHOP_LLM_SERVER_JOB_TTL_SECONDS=3600
export WORKSHOP_EMBEDDING_SERVER_CONCURRENCY=8
export WORKSHOP_EMBEDDING_SERVER_MIN_INTERVAL_SECONDS=0.1
export WORKSHOP_EMBEDDING_SERVER_MAX_RETRIES=8
```

To use Mistral directly instead of OpenRouter, set:

```bash
export WORKSHOP_LLM_PROVIDER="mistral"
export MISTRAL_API_KEY="..."
export MISTRAL_MODEL="mistral-medium-latest"
```

Operational endpoints:

```text
GET  /health
GET  /queue
POST /jobs
GET  /jobs/{job_id}
POST /chat
POST /embeddings
```

`/chat` remains blocking for existing notebook code. The Python client uses
`/jobs` by default when talking to the proxy, polls the job status, and prints
queue position, retries, and final wait time when `verbose=True`. Set
`WORKSHOP_LLM_SERVER_USE_JOBS=0` to force the old blocking `/chat` path.

By default, retrieval reuses the LLM proxy URL for embeddings, so the notebook
only needs one proxy URL:

```python
os.environ["WORKSHOP_LLM_SERVER_URL"] = "http://llm-workshop.example.org:8010"
```

Set `WORKSHOP_EMBEDDING_SERVER_URL` only when embeddings are served at a
different proxy URL. The OpenRouter key remains on the proxy machine.

Behind the single ngrok/Caddy setup above, use:

```python
os.environ["WORKSHOP_LLM_SERVER_URL"] = "https://<ngrok-host>/llm"
```

Each `verbose=True` LLM call prints input tokens, output tokens, and estimated
USD cost. The estimate knows the Mistral Medium 3.5 rates for both
`mistral-medium-latest` and OpenRouter's `mistralai/mistral-medium-3-5`, plus
OpenRouter's `z-ai/glm-5.3-flash` rates: $0.075/M uncached input, $0.015/M
cached input, and $0.25/M output. The OpenRouter dashboard remains the source
of truth when routed-provider prices change.

## Retrieval Cache

Build the retrieval cache once from the Hugging Face dataset
`theostos/pile-of-rocq`, config `coq-coquelicot-toc_nodes`:

```bash
pip install datasets faiss-cpu "sentence-transformers>=2.7.0" "transformers>=4.51.0"
python scripts/build_retrieval_cache.py \
  --hf-dataset theostos/pile-of-rocq \
  --hf-config coq-coquelicot-toc_nodes \
  --library Stdlib \
  --library Coquelicot \
  --output-dir build/retrieval_cache \
  --zip build/retrieval_cache.zip \
  --batch-size 4 \
  --force
```

Those Hugging Face dataset/config/library values are the script defaults, so
the explicit form above can be shortened to just the output paths. The builder
classifies `Corelib/...` and `Bignums/...` rows as `Stdlib`, and `Coquelicot/...`
rows as `Coquelicot`.

The default cache model is `Qwen/Qwen3-Embedding-4B`, whose matching OpenRouter
identifier is `qwen/qwen3-embedding-4b`. The query-side runtime must use a model
compatible with the one recorded in `manifest.json`; mixing embedding spaces
silently destroys retrieval quality. For Qwen3 caches, the builder records both
the local query prompt and the matching OpenRouter model/input type.

The builder writes both `embeddings.npy` and `index.faiss`. The `.npy` file is a
portable float32 matrix in the same order as `metadata.jsonl`, so you can rebuild
another FAISS index strategy later without recomputing embeddings. The default
index is exact `Flat` / inner product over normalized vectors, which is a good
fit for this cache size. Use `--index-factory` only when you want to experiment
with approximate FAISS indexes after the workshop path is stable.

For a rented GPU run, use `--resume` if the embedding process is interrupted.
The script records progress in `embedding_progress.json` and refuses to resume
if the model, entry order, or cache parameters do not match.

Upload `build/retrieval_cache.zip` to a public host. In Colab, point the
notebook to the direct zip URL before the retrieval cell:

```python
DOCSTRING_CACHE_URL = "https://example.org/retrieval_cache.zip"
```

The notebook downloads the zip and loads `index.faiss` + `metadata.jsonl`
locally. Only the small user-query embedding request goes through the workshop
proxy to OpenRouter. If the cache is already unpacked in the runtime, set
`DOCSTRING_CACHE_DIR` instead. If no embedding proxy URL is configured,
`RetrievalClient` retains a local SentenceTransformer fallback.

Minimal retrieval use in the notebook:

```python
from workshop_api import (
    RetrievalClient,
    format_retrieval_hits,
    prepare_colab_retrieval_cache,
)

cache_path = prepare_colab_retrieval_cache(DOCSTRING_CACHE_URL)
retriever = RetrievalClient.from_env(cache_dir=cache_path)
hits = retriever.search(
    "RInt fundamental theorem derivative continuous interval integral",
    library="Coquelicot",
    kind="theorem",
    k=5,
)
print(format_retrieval_hits(hits))

hits = retriever.search(
    "field_simplify ring exponential positivity exp_plus denominator nonzero",
    library="Stdlib",
    kind=["ltac", "theorem"],
    k=5,
)
print(format_retrieval_hits(hits))
```
