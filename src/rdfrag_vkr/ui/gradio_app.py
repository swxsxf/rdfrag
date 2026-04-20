"""Gradio chat UI for the RDFRAG project."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from typing import Callable

import gradio as gr

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.hybrid_retriever import HybridRetriever
from rdfrag_vkr.modules.llm_service import LLMService
from rdfrag_vkr.modules.pdf_parser import PDFParser
from rdfrag_vkr.modules.sparql_service import SparqlService

LOCAL_STORAGE_KEY = "rdfrag_vkr_question_history"

CUSTOM_CSS = """
:root {
  color-scheme: dark;
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
    radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.10), transparent 24%),
    #09090b;
  color: #f4f4f5;
}

.gradio-container {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

#app-shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 20px 14px 24px;
}

.app-header {
  margin-bottom: 18px;
  padding: 18px 22px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 22px;
  background: rgba(15, 23, 42, 0.68);
  backdrop-filter: blur(10px);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
}

.app-title {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0 0 8px;
}

.app-subtitle {
  margin: 0;
  color: #cbd5e1;
  font-size: 15px;
}

.status-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 16px;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 13px;
  color: #e2e8f0;
}

.status-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  display: inline-block;
}

.status-dot.ok {
  background: #22c55e;
  box-shadow: 0 0 10px rgba(34, 197, 94, 0.75);
}

.status-dot.warn {
  background: #f59e0b;
  box-shadow: 0 0 10px rgba(245, 158, 11, 0.70);
}

.status-dot.off {
  background: #ef4444;
  box-shadow: 0 0 10px rgba(239, 68, 68, 0.65);
}

.sidebar-shell,
.chat-shell {
  min-height: 720px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 24px;
  background: rgba(10, 14, 23, 0.78);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
}

.sidebar-shell {
  padding: 16px;
}

.chat-shell {
  padding: 16px 16px 8px;
}

.sidebar-title,
.chat-title {
  font-size: 15px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
  margin-bottom: 12px;
}

#question-history-panel {
  height: 620px;
  overflow-y: auto;
  padding-right: 4px;
}

.history-empty {
  color: #64748b;
  font-size: 14px;
  line-height: 1.5;
}

.history-item {
  width: 100%;
  margin-bottom: 10px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.07);
  background: rgba(255, 255, 255, 0.04);
  color: #e2e8f0;
  text-align: left;
  font-size: 13px;
  line-height: 1.45;
  cursor: pointer;
  transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}

.history-item:hover {
  transform: translateY(-1px);
  border-color: rgba(56, 189, 248, 0.45);
  background: rgba(56, 189, 248, 0.08);
}

#new-chat-btn button {
  width: 100%;
  min-height: 46px;
  border-radius: 16px;
  background: linear-gradient(135deg, #0f172a, #111827);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: #f8fafc;
  font-weight: 600;
}

#top-k-slider {
  margin-bottom: 10px;
}

#top-k-slider .wrap {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 18px;
  padding: 8px 12px;
}

#rag-chatbot {
  border: none;
}

#rag-chatbot .bubble-wrap {
  margin-bottom: 16px;
}

#rag-chatbot .message {
  border-radius: 20px !important;
  line-height: 1.6;
  font-size: 14px;
}

#rag-chatbot .message.user {
  background: linear-gradient(135deg, #0f172a, #1d4ed8) !important;
  border: 1px solid rgba(96, 165, 250, 0.25) !important;
}

#rag-chatbot .message.bot {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

#question-input textarea {
  min-height: 88px !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  color: #f8fafc !important;
}

#send-btn button {
  min-height: 88px;
  border-radius: 18px;
  background: linear-gradient(135deg, #0284c7, #2563eb);
  border: none;
  font-weight: 700;
}

.pdf-list {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.10);
}

.pdf-list-title {
  color: #94a3b8;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.pdf-list ul {
  margin: 0;
  padding-left: 18px;
}

.pdf-list li {
  margin: 0 0 6px;
  color: #cbd5e1;
}

#question-history-data {
  display: none !important;
}
"""

HEAD_HTML = f"""
<script>
(() => {{
  const storageKey = "{LOCAL_STORAGE_KEY}";

  function escapeHtml(text) {{
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }}

  function getQuestions() {{
    try {{
      const raw = window.localStorage.getItem(storageKey) || "[]";
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    }} catch (_error) {{
      return [];
    }}
  }}

  function renderSidebar() {{
    const host = document.getElementById("question-history-panel");
    if (!host) {{
      return;
    }}
    const questions = getQuestions();
    if (!questions.length) {{
      host.innerHTML = '<div class="history-empty">История вопросов появится здесь и сохранится в браузере.</div>';
      return;
    }}
    const items = [...questions].reverse().map((question) => {{
      const safeQuestion = escapeHtml(question);
      const payload = JSON.stringify(question);
      return `<button class="history-item" onclick='window.rdfragFillQuestion(${{payload}})'>${{safeQuestion}}</button>`;
    }});
    host.innerHTML = items.join("");
  }}

  function syncFromHiddenField() {{
    const field = document.querySelector("#question-history-data textarea, #question-history-data input");
    if (!field) {{
      window.setTimeout(syncFromHiddenField, 350);
      return;
    }}
    let lastValue = field.value || "[]";
    try {{
      window.localStorage.setItem(storageKey, lastValue);
    }} catch (_error) {{
      // Ignore browser storage issues.
    }}
    renderSidebar();
    window.setInterval(() => {{
      const nextValue = field.value || "[]";
      if (nextValue === lastValue) {{
        return;
      }}
      lastValue = nextValue;
      try {{
        window.localStorage.setItem(storageKey, nextValue);
      }} catch (_error) {{
        // Ignore browser storage issues.
      }}
      renderSidebar();
    }}, 300);
  }}

  window.rdfragFillQuestion = function(question) {{
    const textbox = document.querySelector("#question-input textarea");
    if (!textbox) {{
      return;
    }}
    textbox.value = question;
    textbox.dispatchEvent(new Event("input", {{ bubbles: true }}));
    textbox.focus();
  }};

  window.addEventListener("load", () => {{
    renderSidebar();
    syncFromHiddenField();
  }});
}})();
</script>
"""


def _status_badge(label: str, value: bool) -> str:
    status_class = "ok" if value else "off"
    text = "online" if value else "offline"
    return (
        '<div class="status-pill">'
        f'<span class="status-dot {status_class}"></span>'
        f"<strong>{label}</strong> {text}"
        "</div>"
    )


def _safe_status(check: Callable[[], bool]) -> bool:
    try:
        return bool(check())
    except Exception:
        return False


def _dedupe_pdf_names(hits: list[dict]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for hit in hits:
        if isinstance(hit, dict):
            source_file = str(hit.get("source_file", "")).strip()
        else:
            source_file = str(getattr(hit, "source_file", "")).strip()
        if not source_file or source_file in seen:
            continue
        seen.add(source_file)
        names.append(source_file)
    return names


def _format_answer(answer: str, hits: list[dict]) -> str:
    pdf_names = _dedupe_pdf_names(hits)
    if not pdf_names:
        return answer.strip()
    pdf_markup = "\n".join(f"<li>{name}</li>" for name in pdf_names)
    return (
        f"{answer.strip()}\n\n"
        '<div class="pdf-list">'
        '<div class="pdf-list-title">PDF-источники</div>'
        f"<ul>{pdf_markup}</ul>"
        "</div>"
    )


class RAGChatController:
    """Thin controller for the Gradio chat UI."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(self.settings)
        self.llm_service = LLMService(self.settings)
        self.pdf_parser = PDFParser(self.settings)
        self.sparql_service = SparqlService(self.settings)

    def build_status_markup(self) -> str:
        try:
            pdf_count = len(list(self.settings.raw_pdf_dir.glob("*.pdf")))
        except OSError:
            pdf_count = 0
        return (
            '<div class="app-header">'
            '<div class="app-title">RDFRAG Chat</div>'
            '<p class="app-subtitle">'
            "Гибридный graph + vector RAG по корпусу научных PDF для демонстрации ВКР."
            "</p>"
            '<div class="status-bar">'
            f'<div class="status-pill"><strong>PDF</strong> {pdf_count}</div>'
            + _status_badge("GROBID", _safe_status(self.pdf_parser.is_grobid_available))
            + _status_badge("Fuseki", _safe_status(self.sparql_service.is_fuseki_available))
            + _status_badge("Ollama", _safe_status(self.llm_service.is_ollama_available))
            + "</div>"
            "</div>"
        )

    def _query(self, question: str, top_k: int) -> tuple[str, list[dict]]:
        hits, graph_context = self.retriever.search(question, top_k=top_k)
        answer = self.llm_service.generate_answer(question, hits, graph_context)
        serialized_hits = [
            hit.model_dump() if hasattr(hit, "model_dump") else dict(hit)
            for hit in hits
        ]
        return _format_answer(answer, hits), serialized_hits

    def clear_chat(self) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], str, str]:
        return [], [], [], "[]", ""

    def stream_answer(
        self,
        question: str,
        top_k: int,
        chat_history: list[dict[str, str]] | None,
        question_history: list[str] | None,
    ) -> Generator[tuple[list[dict[str, str]], list[dict[str, str]], list[str], str, str], None, None]:
        prompt = question.strip()
        chat_rows = [dict(row) for row in (chat_history or [])]
        asked_questions = list(question_history or [])
        if not prompt:
            payload = json.dumps(asked_questions, ensure_ascii=False)
            yield chat_rows, chat_rows, asked_questions, payload, ""
            return

        asked_questions.append(prompt)
        payload = json.dumps(asked_questions, ensure_ascii=False)
        chat_rows.append({"role": "user", "content": prompt})
        chat_rows.append({"role": "assistant", "content": "Думаю."})
        yield chat_rows, chat_rows, asked_questions, payload, ""

        result: dict[str, object] = {}
        error: dict[str, str] = {}

        def worker() -> None:
            try:
                answer, hits = self._query(prompt, int(top_k))
                result["answer"] = answer
                result["hits"] = hits
            except Exception as exc:  # pragma: no cover - UI resilience
                error["message"] = str(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        frames = ("Думаю.", "Думаю..", "Думаю...")
        frame_index = 0
        while thread.is_alive():
            chat_rows[-1]["content"] = frames[frame_index % len(frames)]
            yield chat_rows, chat_rows, asked_questions, payload, ""
            time.sleep(0.35)
            frame_index += 1

        if error:
            chat_rows[-1]["content"] = (
                "Не удалось получить ответ.\n\n"
                f"Техническая ошибка: `{error['message']}`"
            )
        else:
            chat_rows[-1]["content"] = str(result["answer"])
        yield chat_rows, chat_rows, asked_questions, payload, ""


def create_demo(settings: Settings | None = None) -> gr.Blocks:
    """Create the Gradio chat application."""
    controller = RAGChatController(settings)

    with gr.Blocks(title="RDFRAG Chat") as demo:
        chat_state = gr.State([])
        question_state = gr.State([])

        with gr.Column(elem_id="app-shell"):
            gr.HTML(controller.build_status_markup())
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=280):
                    with gr.Column(elem_classes=["sidebar-shell"]):
                        gr.Markdown("История вопросов", elem_classes=["sidebar-title"])
                        gr.HTML('<div id="question-history-panel"></div>')
                        new_chat = gr.Button("Новый чат", elem_id="new-chat-btn")
                        question_history_json = gr.Textbox(
                            value="[]",
                            visible=False,
                            elem_id="question-history-data",
                            show_label=False,
                        )
                with gr.Column(scale=4, min_width=780):
                    with gr.Column(elem_classes=["chat-shell"]):
                        gr.Markdown("Диалог с системой", elem_classes=["chat-title"])
                        top_k = gr.Slider(
                            minimum=5,
                            maximum=15,
                            step=1,
                            value=5,
                            label="Top-K retrieval",
                            elem_id="top-k-slider",
                        )
                        chatbot = gr.Chatbot(
                            value=[],
                            label="",
                            height=560,
                            elem_id="rag-chatbot",
                        )
                        with gr.Row():
                            question_box = gr.Textbox(
                                placeholder="Задай вопрос по корпусу статей...",
                                show_label=False,
                                lines=4,
                                max_lines=6,
                                elem_id="question-input",
                                scale=8,
                            )
                            send_button = gr.Button("Отправить", elem_id="send-btn", scale=1)

        submit_outputs = [chatbot, chat_state, question_state, question_history_json, question_box]
        submit_inputs = [question_box, top_k, chat_state, question_state]

        send_button.click(
            fn=controller.stream_answer,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )
        question_box.submit(
            fn=controller.stream_answer,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )
        new_chat.click(
            fn=controller.clear_chat,
            outputs=submit_outputs,
        )

    demo.queue(default_concurrency_limit=2)
    return demo
