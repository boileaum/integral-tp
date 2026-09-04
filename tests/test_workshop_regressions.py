from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from workshop_api import llm_server
from workshop_api.retrieval import (
    LocalFaissRetriever,
    RemoteEmbeddingClient,
    RetrievalClient,
)
from workshop_api.rocq import RocqDocument, RocqWorkshop
from workshop_api.widgets import RetrievalExplorer


def test_notebook_uses_glm_medium_and_relies_on_20k_proof_default():
    notebook_path = Path(__file__).resolve().parents[1] / "integral_workshop.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'os.environ["WORKSHOP_LLM_PROVIDER"] = "openrouter"' in code
    assert 'os.environ["OPENROUTER_MODEL"] = "z-ai/glm-5.3-flash"' in code
    assert 'os.environ["OPENROUTER_REASONING_EFFORT"] = "medium"' in code
    assert "max_tokens=" not in code
    assert "assert result_direct" not in code
    assert "assert result_feedback" not in code


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._body


def test_remote_embedding_client_uses_proxy_and_preserves_input_order(monkeypatch):
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            }
        )

    monkeypatch.setattr("workshop_api.retrieval.requests.post", fake_post)
    client = RemoteEmbeddingClient(
        server_url="https://workshop.example/llm",
        server_token="participant-token",
        model_name="qwen/qwen3-embedding-4b",
    )

    vectors = client.encode(["first", "second"], input_type="search_query")

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["url"] == "https://workshop.example/llm/embeddings"
    assert captured["json"] == {
        "input": ["first", "second"],
        "model": "qwen/qwen3-embedding-4b",
        "encoding_format": "float",
        "input_type": "search_query",
    }
    assert captured["headers"]["Authorization"] == "Bearer participant-token"


def test_retrieval_client_reuses_llm_proxy_for_embeddings(monkeypatch):
    monkeypatch.delenv("WORKSHOP_EMBEDDING_SERVER_URL", raising=False)
    monkeypatch.setenv("WORKSHOP_LLM_SERVER_URL", "https://workshop.example/llm")
    monkeypatch.setenv("WORKSHOP_LLM_SERVER_TOKEN", "participant-token")

    client = RetrievalClient.from_env(cache_dir="cache")

    assert client.embedding_server_url == "https://workshop.example/llm"
    assert client.embedding_server_token == "participant-token"


def test_faiss_search_uses_remote_query_embedding_without_local_model(
    monkeypatch,
    tmp_path,
):
    import numpy as np

    manifest = {
        "model_name": "Qwen/Qwen3-Embedding-4B",
        "index_file": "index.faiss",
        "normalize_embeddings": True,
        "openrouter_model_name": "qwen/qwen3-embedding-4b",
        "openrouter_query_input_type": "search_query",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"uid": "one", "name": "one", "library": "Stdlib"}) + "\n"
    )
    (tmp_path / "index.faiss").touch()

    class Index:
        ntotal = 1
        d = 2

        def search(self, vectors, count):
            assert count == 1
            np.testing.assert_allclose(vectors, [[0.6, 0.8]])
            return np.asarray([[0.9]], dtype="float32"), np.asarray([[0]])

    fake_faiss = types.ModuleType("faiss")
    fake_faiss.read_index = lambda path: Index()
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    captured = {}

    def fake_encode(self, texts, *, input_type):
        captured.update(texts=texts, input_type=input_type)
        return [[3.0, 4.0]]

    monkeypatch.setattr(RemoteEmbeddingClient, "encode", fake_encode)
    retriever = LocalFaissRetriever(
        cache_dir=tmp_path,
        embedding_server_url="https://workshop.example/llm",
    )

    hits = retriever.search("needle", library="Stdlib", k=1)

    assert [hit["name"] for hit in hits] == ["one"]
    assert captured == {"texts": ["needle"], "input_type": "search_query"}
    assert retriever.model is None


def test_embedding_proxy_keeps_openrouter_key_server_side(monkeypatch):
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _FakeResponse(
            {
                "data": [{"index": 0, "embedding": [0.5, 0.25]}],
                "model": "qwen/qwen3-embedding-4b",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "server-only-test-key")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-4b")
    monkeypatch.setattr(llm_server.requests, "post", fake_post)
    request = llm_server.EmbeddingRequest(
        input=["query"],
        model="qwen/qwen3-embedding-4b",
        input_type="search_query",
    )

    body = llm_server._embed_once(request)

    assert body["data"][0]["embedding"] == [0.5, 0.25]
    assert captured["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer server-only-test-key"
    assert captured["json"]["input_type"] == "search_query"

    with pytest.raises(ValueError, match="not enabled"):
        llm_server._embed_once(
            llm_server.EmbeddingRequest(input="query", model="another/model")
        )


class _FakeRocqClient:
    def __init__(self):
        self.commands = []
        self.fail_on = None

    def run(self, state, command, *, timeout):
        del state, timeout
        self.commands.append(command)
        if command == self.fail_on:
            raise RuntimeError("invalid corrected statement")
        return SimpleNamespace(command=command, proof_finished=False)

    def goals(self, state, *, timeout):
        del timeout
        return [SimpleNamespace(pp=f"goal for {state.command}")]


def _workshop() -> RocqWorkshop:
    workshop = object.__new__(RocqWorkshop)
    workshop.client = _FakeRocqClient()
    workshop.timeout = 30
    workshop.global_state = object()
    workshop.lemmas = {}
    workshop.elements = []
    workshop.completed_order = []
    workshop.timeline = []
    return workshop


def test_ensure_theorem_reuses_or_replaces_only_open_declarations():
    workshop = _workshop()
    document = object.__new__(RocqDocument)
    document.workshop = workshop
    original = "Lemma demo : True."
    corrected = "Lemma demo : 1 = 1."

    first = document.ensure_theorem(original)
    command_count = len(workshop.client.commands)
    same = document.ensure_theorem(original)
    corrected_session = document.ensure_theorem(corrected)

    assert same.name == first.name == corrected_session.name == "demo"
    assert len(workshop.client.commands) == command_count + 1
    assert corrected_session.header == corrected
    assert corrected_session.lemma.latest_index == 0

    old_lemma = corrected_session.lemma
    invalid = "Lemma demo : INVALID."
    workshop.client.fail_on = invalid
    with pytest.raises(RuntimeError, match="invalid corrected statement"):
        document.ensure_theorem(invalid)
    assert workshop.lemmas["demo"] is old_lemma

    old_lemma.completed = True
    with pytest.raises(ValueError, match="already completed"):
        document.ensure_theorem("Lemma demo : False.")


def _fake_widgets_module():
    module = types.ModuleType("ipywidgets")

    class Layout:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Widget:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ValueWidget(Widget):
        def __init__(self, value="", **kwargs):
            super().__init__(value=value, **kwargs)

    class Button(Widget):
        def __init__(self, description="", icon="", **kwargs):
            super().__init__(description=description, icon=icon, disabled=False, **kwargs)
            self._callbacks = []

        def on_click(self, callback):
            self._callbacks.append(callback)

        def click(self):
            # Deliberately invoke callbacks even while disabled. This simulates
            # click messages that reached the kernel queue before the disabled
            # trait update reached the browser.
            for callback in self._callbacks:
                callback(self)

    class Box(Widget):
        def __init__(self, children=(), **kwargs):
            super().__init__(children=tuple(children), **kwargs)

    module.Layout = Layout
    module.Textarea = ValueWidget
    module.Dropdown = ValueWidget
    module.Text = ValueWidget
    module.IntSlider = ValueWidget
    module.HTML = ValueWidget
    module.Button = Button
    module.VBox = Box
    module.HBox = Box
    return module


def test_search_button_disables_during_search_and_ignores_queued_duplicate(monkeypatch):
    monkeypatch.setitem(sys.modules, "ipywidgets", _fake_widgets_module())

    class Retriever:
        calls = 0
        explorer = None

        def search(self, query, *, library, kind, k):
            self.calls += 1
            assert query == "needle"
            assert self.explorer._search_button.disabled is True
            assert self.explorer._search_button.description == "Searching..."
            return [{"uid": "hit", "name": "demo", "content": "Lemma demo : True."}]

    retriever = Retriever()
    explorer = RetrievalExplorer(retriever, default_query="needle")
    retriever.explorer = explorer
    explorer.render()

    explorer._search_button.click()
    explorer._search_button.click()

    assert retriever.calls == 1
    assert explorer._search_button.disabled is False
    assert explorer._search_button.description == "Search"
    assert "Duplicate search ignored" in explorer._search_status.value
