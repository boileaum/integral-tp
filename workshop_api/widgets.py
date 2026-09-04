from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .retrieval import format_retrieval_hits, normalize_kind_name


def _hit_key(hit: dict[str, Any]) -> str:
    return str(hit.get("uid") or (hit.get("source"), hit.get("name"), hit.get("content")))


def _shorten(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _highlight_rocq(text: object) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name

        formatter = HtmlFormatter(nowrap=False)
        return highlight(raw, get_lexer_by_name("coq"), formatter)
    except Exception:
        return f"<pre>{html.escape(raw)}</pre>"


def _hit_html(hit: dict[str, Any]) -> str:
    name = html.escape(str(hit.get("name") or hit.get("uid") or "<unnamed>"))
    kind = html.escape(normalize_kind_name(hit.get("kind")) or "?")
    library = html.escape(str(hit.get("library") or "?"))
    source = html.escape(_shorten(hit.get("source"), limit=120))
    score = hit.get("score")
    score_text = ""
    if score is not None:
        try:
            score_text = f" score={float(score):.3f}"
        except (TypeError, ValueError):
            score_text = f" score={html.escape(str(score))}"

    statement = hit.get("statement") or hit.get("content")
    docstring = hit.get("docstring")
    doc_html = ""
    if docstring:
        doc_html = (
            "<div class='rocq-docstring'>"
            f"{html.escape(_shorten(docstring, limit=1800))}"
            "</div>"
        )

    return (
        "<div class='rocq-hit'>"
        f"<div class='rocq-hit-head'><b>{name}</b> "
        f"<span>{kind}</span> <span>{library}</span><span>{score_text}</span></div>"
        f"<div class='rocq-source'>{source}</div>"
        f"{_highlight_rocq(statement)}"
        f"{doc_html}"
        "</div>"
    )


def _style_html() -> str:
    try:
        from pygments.formatters import HtmlFormatter

        pygments_css = HtmlFormatter().get_style_defs(".rocq-hit .highlight")
    except Exception:
        pygments_css = ""
    return f"""
<style>
{pygments_css}
.rocq-retrieval {{
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.rocq-hit {{
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 8px 10px;
  margin: 0 0 8px 0;
  background: #ffffff;
  width: 100%;
  box-sizing: border-box;
}}
.rocq-hit-head {{
  display: flex;
  gap: 8px;
  align-items: baseline;
  flex-wrap: wrap;
  margin-bottom: 3px;
}}
.rocq-hit-head span,
.rocq-source {{
  color: #57606a;
  font-size: 12px;
}}
.rocq-source {{
  margin-bottom: 6px;
}}
.rocq-docstring {{
  color: #24292f;
  background: #f6f8fa;
  border-radius: 4px;
  padding: 6px 8px;
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.rocq-hit pre,
.rocq-hit .highlight {{
  margin: 0;
  overflow: visible;
  background: #f6f8fa;
  border-radius: 4px;
  padding: 8px;
  font-size: 13px;
  line-height: 1.45;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.rocq-hit .highlight pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
</style>
"""


@dataclass
class RetrievalExplorer:
    """Small ipywidgets UI for searching and curating retrieval hits."""

    retriever: Any
    selected_hits: list[dict[str, Any]] = field(default_factory=list)
    default_query: str = ""
    default_library: str | None = None
    default_kind: str = ""
    default_k: int = 8
    duplicate_click_window_seconds: float = 2.0
    _ui: Any = field(default=None, init=False, repr=False)
    _results_box: Any = field(default=None, init=False, repr=False)
    _selected_box: Any = field(default=None, init=False, repr=False)
    _context_box: Any = field(default=None, init=False, repr=False)
    _selected_label: Any = field(default=None, init=False, repr=False)
    _search_button: Any = field(default=None, init=False, repr=False)
    _search_status: Any = field(default=None, init=False, repr=False)
    _search_in_progress: bool = field(default=False, init=False, repr=False)
    _last_search_signature: tuple[Any, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_search_finished_at: float = field(default=0.0, init=False, repr=False)

    @property
    def hits(self) -> list[dict[str, Any]]:
        return self.selected_hits

    def context(self, *, statement_chars: int = 1000, docstring_chars: int = 500) -> str:
        return format_retrieval_hits(
            self.selected_hits,
            statement_chars=statement_chars,
            docstring_chars=docstring_chars,
        )

    def clear(self) -> None:
        self.selected_hits.clear()
        self._render_selected()

    def add(self, hit: dict[str, Any]) -> None:
        key = _hit_key(hit)
        if not any(_hit_key(old) == key for old in self.selected_hits):
            self.selected_hits.append(hit)
        self._render_selected()

    def remove(self, key: str) -> None:
        self.selected_hits[:] = [hit for hit in self.selected_hits if _hit_key(hit) != key]
        self._render_selected()

    def render(self) -> Any:
        if self._ui is not None:
            return self._ui

        try:
            import ipywidgets as widgets
        except Exception as exc:
            raise RuntimeError(
                "RetrievalExplorer requires ipywidgets. In Colab, install "
                "`integral-tp[colab]` and rerun the runtime."
            ) from exc

        try:
            from google.colab import output

            output.enable_custom_widget_manager()
        except Exception:
            pass

        query = widgets.Textarea(
            value=self.default_query,
            placeholder="Search Stdlib / Coquelicot...",
            layout=widgets.Layout(width="100%", height="76px"),
        )
        library = widgets.Dropdown(
            options=[
                ("All libraries", None),
                ("Stdlib", "Stdlib"),
                ("Coquelicot", "Coquelicot"),
            ],
            value=self.default_library,
            description="Library",
            layout=widgets.Layout(width="230px"),
        )
        kind = widgets.Text(
            value=self.default_kind,
            placeholder="definition,theorem,ltac",
            description="Kind",
            layout=widgets.Layout(width="360px"),
        )
        k = widgets.IntSlider(
            value=int(self.default_k),
            min=1,
            max=20,
            step=1,
            description="Hits",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
        )
        search_button = widgets.Button(description="Search", icon="search")
        self._search_button = search_button
        self._search_status = widgets.HTML()
        clear_button = widgets.Button(description="Clear selection", icon="trash")
        self._selected_label = widgets.HTML()
        self._results_box = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
                overflow="visible",
                border="1px solid #d0d7de",
                padding="6px",
            )
        )
        self._selected_box = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
                overflow="visible",
                border="1px solid #d0d7de",
                padding="6px",
            )
        )
        self._context_box = widgets.Textarea(
            description="Context",
            layout=widgets.Layout(width="100%", height="96px"),
        )

        def run_search(_: object = None) -> None:
            signature = (
                query.value.strip(),
                library.value,
                kind.value.strip(),
                int(k.value),
            )
            now = time.monotonic()
            duplicate_just_finished = (
                signature == self._last_search_signature
                and now - self._last_search_finished_at
                < max(float(self.duplicate_click_window_seconds), 0.0)
            )
            if self._search_in_progress or duplicate_just_finished:
                self._search_status.value = (
                    "<span style='color:#57606a'>Duplicate search ignored.</span>"
                )
                return

            self._search_in_progress = True
            search_button.disabled = True
            search_button.description = "Searching..."
            search_button.icon = "spinner"
            self._search_status.value = (
                "<span style='color:#57606a'>Computing query embedding and searching...</span>"
            )
            try:
                hits = self.retriever.search(
                    signature[0],
                    library=signature[1],
                    kind=signature[2] or None,
                    k=signature[3],
                )
                self._render_results(hits)
                self._search_status.value = (
                    f"<span style='color:#1a7f37'>{len(hits)} hit(s).</span>"
                )
            except Exception as exc:
                self._search_status.value = (
                    "<span style='color:#cf222e'><b>Search failed:</b> "
                    f"{html.escape(str(exc))}</span>"
                )
            finally:
                self._last_search_signature = signature
                self._last_search_finished_at = time.monotonic()
                self._search_in_progress = False
                search_button.disabled = False
                search_button.description = "Search"
                search_button.icon = "search"

        search_button.on_click(run_search)
        clear_button.on_click(lambda _: self.clear())

        self._ui = widgets.VBox(
            [
                widgets.HTML(_style_html()),
                query,
                widgets.HBox([library, kind, k, search_button]),
                self._search_status,
                widgets.HTML("<b>Search results</b>"),
                self._results_box,
                widgets.HBox([widgets.HTML("<b>Selected context</b>"), clear_button]),
                self._selected_label,
                self._selected_box,
                self._context_box,
            ]
        )
        self._render_selected()
        return self._ui

    def display(self) -> None:
        try:
            from IPython.display import display
        except Exception as exc:
            raise RuntimeError("RetrievalExplorer.display() requires IPython.") from exc
        display(self.render())

    def _render_results(self, hits: Sequence[dict[str, Any]]) -> None:
        if self._results_box is None:
            return
        import ipywidgets as widgets

        rows: list[Any] = []
        for hit in hits:
            button = widgets.Button(
                description="Add",
                icon="plus",
                layout=widgets.Layout(width="80px"),
            )
            def add_current(_: object, current: dict[str, Any] = hit) -> None:
                self.add(current)

            button.on_click(add_current)
            card = widgets.HTML(
                _hit_html(dict(hit)),
                layout=widgets.Layout(width="100%", flex="1 1 auto"),
            )
            rows.append(
                widgets.HBox(
                    [button, card],
                    layout=widgets.Layout(width="100%", align_items="flex-start"),
                )
            )
        self._results_box.children = tuple(rows) if rows else (widgets.HTML("<em>No hits.</em>"),)

    def _render_selected(self) -> None:
        if self._selected_box is None or self._context_box is None or self._selected_label is None:
            return
        import ipywidgets as widgets

        self._selected_label.value = f"{len(self.selected_hits)} selected item(s)"
        self._context_box.value = self.context()
        rows: list[Any] = []
        for hit in self.selected_hits:
            key = _hit_key(hit)
            button = widgets.Button(
                description="Remove",
                icon="minus",
                layout=widgets.Layout(width="100px"),
            )

            def remove_current(_: object, current_key: str = key) -> None:
                self.remove(current_key)

            button.on_click(remove_current)
            card = widgets.HTML(
                _hit_html(dict(hit)),
                layout=widgets.Layout(width="100%", flex="1 1 auto"),
            )
            rows.append(
                widgets.HBox(
                    [button, card],
                    layout=widgets.Layout(width="100%", align_items="flex-start"),
                )
            )
        self._selected_box.children = tuple(rows) if rows else (widgets.HTML("<em>No selected items.</em>"),)
