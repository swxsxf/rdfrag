"""Answer generation service separated from retrieval logic."""

from __future__ import annotations

import logging
import re

import httpx

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.schemas import RetrievalHit


class LLMService:
    """Generate a grounded answer from retrieved graph and vector context."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def generate_answer(self, question: str, hits: list[RetrievalHit], graph_context: list[dict]) -> str:
        """Generate an answer with a local Ollama model when available."""
        if not hits:
            return (
                "No relevant materials were retrieved from the current corpus. "
                "Run ingestion first or refine the question using terms present in the PDF collection."
            )

        if self.settings.llm_provider.lower() == "ollama" and self.is_ollama_available():
            prompt = self._build_prompt(question, hits, graph_context)
            answer = self._generate_with_ollama(prompt)
            if answer and self._is_usable_answer(answer, question):
                if self._is_russian(question) and not self._looks_like_russian_answer(answer):
                    self.logger.warning("Ollama returned a non-Russian answer for a Russian question; retrying.")
                    retry_prompt = (
                        "/no_think\n"
                        "Перепиши следующий ответ строго на русском языке. "
                        "Сохрани смысл, формат 'Ответ:' и 'Ключевые пункты:', не добавляй новые факты.\n\n"
                        f"Вопрос: {question}\n\n"
                        f"Ответ для перевода/исправления:\n{answer}"
                    )
                    retry_answer = self._generate_with_ollama(retry_prompt)
                    if retry_answer and self._looks_like_russian_answer(retry_answer):
                        return retry_answer
                else:
                    return answer
            retry_prompt = self._build_prompt(question, hits[:2], graph_context=[], compact=True)
            retry_answer = self._generate_with_ollama(retry_prompt)
            if retry_answer and self._is_usable_answer(retry_answer, question):
                return retry_answer

        raise RuntimeError(
            "Ollama недоступна или не смогла сгенерировать ответ. "
            f"Проверь, что сервер Ollama работает на {self.settings.ollama_url} "
            f"и что модель {self.settings.ollama_model} доступна."
        )

    def is_ollama_available(self) -> bool:
        """Check whether the configured Ollama endpoint is reachable."""
        try:
            with httpx.Client(trust_env=False, timeout=min(self.settings.ollama_timeout_seconds, 15)) as client:
                response = client.get(f"{self.settings.ollama_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def _generate_with_ollama(self, prompt: str) -> str | None:
        try:
            with httpx.Client(trust_env=False, timeout=self.settings.ollama_timeout_seconds) as client:
                response = client.post(
                    f"{self.settings.ollama_url}/api/generate",
                    json={
                        "model": self.settings.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": self.settings.ollama_temperature,
                            "num_predict": max(700, self.settings.ollama_num_predict),
                        },
                    },
                )
            response.raise_for_status()
            payload = response.json()
            answer = str(payload.get("response", "")).strip()
            if not answer:
                return None
            return self._normalize_answer_text(answer)
        except Exception as exc:
            self.logger.warning("Ollama answer generation failed, using fallback synthesis: %s", exc)
            return None

    def _build_prompt(
        self,
        question: str,
        hits: list[RetrievalHit],
        graph_context: list[dict],
        compact: bool = False,
    ) -> str:
        evidence_blocks = []
        max_hits = 2 if compact else 3
        max_chars = 420 if compact else 620
        for index, hit in enumerate(self._select_evidence_hits(hits)[:max_hits], start=1):
            clean_text = self._prepare_context_text(hit.text, question)
            if not clean_text:
                continue
            evidence_blocks.append(
                f"Фрагмент {index}:\n"
                f"{clean_text[:max_chars]}"
            )
        evidence_part = "\n\n".join(evidence_blocks)
        target_language = self._target_language(question)
        if self._is_russian(question):
            output_format = (
                "/no_think\n"
                "Формат ответа:\n"
                "Ответ: один содержательный абзац из 4-6 предложений с прямым ответом на вопрос.\n"
                "В абзаце раскрой суть, принцип работы/механизм и практическое значение, если это подтверждается фрагментами.\n"
                "Затем обязательно добавь отдельный блок 'Ключевые пункты:' с 3-4 короткими пунктами через дефис.\n"
                "Если уместно, добавь 1 короткое ограничение или оговорку.\n"
                "Не ограничивайся только абзацем: блок 'Ключевые пункты:' обязателен.\n"
                "Не используй длинные разделы и не превращай ответ в текст главы ВКР.\n\n"
            )
        else:
            output_format = (
                "/no_think\n"
                "Response format:\n"
                "Answer: one informative paragraph of 4-6 sentences that directly answers the question.\n"
                "Cover the core idea, mechanism, and practical meaning when supported by the fragments.\n"
                "Then always add a separate 'Key points:' block with 3-4 concise hyphen bullets.\n"
                "When relevant, add 1 short limitation or caveat.\n"
                "Do not return only the paragraph: the 'Key points:' block is required.\n"
                "Do not use long sections and do not write a thesis chapter.\n\n"
            )
        if self._is_russian(question):
            system_prompt = (
                "Ты — эксперт-аналитик. Тебе предоставлены фрагменты научных PDF-документов.\n\n"
                "ПРАВИЛА:\n"
                "1. Отвечай компактно, структурированно и по делу, но не слишком сухо.\n"
                "2. Используй предоставленные фрагменты и не игнорируй релевантные детали.\n"
                "3. Приводи конкретные термины, механизмы, методы, примеры и ограничения из документов, но без лишнего объема.\n"
                "4. Если фрагменты не дают полного ответа, прямо укажи, что вывод ограничен имеющимся контекстом.\n"
                "5. Если фрагменты противоречат друг другу, укажи это явно.\n"
                "6. Не выдумывай факты, которых нет в предоставленных фрагментах.\n"
                "7. Не пиши как retrieval-отчёт: не упоминай score, graph context, файл, chunk, API или техническую диагностику.\n"
                "8. Пиши весь ответ на русском языке, даже если отдельные фрагменты или названия статей даны на английском.\n\n"
                "ВАЖНО: вопрос задан на русском, поэтому весь итоговый ответ обязан быть на русском языке.\n\n"
                "Итоговый ответ обязательно должен содержать строки 'Ответ:' и 'Ключевые пункты:'.\n\n"
            )
            user_prompt = (
                f"Вопрос: {question}\n\n"
                f"Фрагменты научных документов:\n{evidence_part}\n\n"
                "Дай компактный экспертный ответ, опираясь на фрагменты выше.\n"
            )
        else:
            system_prompt = (
                "You are an expert analyst. You are given fragments from scientific PDF documents.\n\n"
                "RULES:\n"
                "1. Answer compactly, structurally, and directly, but not too tersely.\n"
                "2. Use all provided fragments and do not ignore relevant evidence.\n"
                "3. Include concrete terms, mechanisms, methods, examples, and limitations from the documents, but avoid unnecessary length.\n"
                "4. If the fragments do not fully support a conclusion, state that clearly.\n"
                "5. If fragments contradict each other, mention that explicitly.\n"
                "6. Do not invent facts not grounded in the provided fragments.\n"
                "7. Do not write like a retrieval report: do not mention scores, graph context, files, chunks, API, or technical diagnostics.\n"
                f"8. Write the answer in {target_language}.\n\n"
            )
            user_prompt = (
                f"Question: {question}\n\n"
                f"Scientific document fragments:\n{evidence_part}\n\n"
                "Provide a compact expert answer grounded in the fragments above.\n"
            )

        return system_prompt + output_format + user_prompt

    @staticmethod
    def _prepare_context_text(text: str, question: str) -> str:
        cleaned = " ".join(text.replace("\n", " ").split())
        if cleaned == "Matched through graph entities.":
            return ""
        cleaned = cleaned.replace("•", " ")
        cleaned = cleaned.replace("[", "").replace("]", "")
        cleaned = " ".join(cleaned.split())
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
        terms = LLMService._query_terms_for_context(question)
        selected = [
            part
            for part in parts
            if 45 <= len(part) <= 420 and any(term in part.lower() for term in terms)
        ]
        if not selected:
            selected = [part for part in parts if 45 <= len(part) <= 420]
        return " ".join(selected[:3]) or cleaned[:620]

    @staticmethod
    def _query_terms_for_context(question: str) -> set[str]:
        lowered = question.lower()
        terms = {token for token in re.findall(r"[\w-]+", lowered) if len(token) > 3}
        aliases = {
            "блокчейн": {"blockchain", "dlt", "реестр", "реестра"},
            "цифров": {"digital", "twin", "twins", "dt", "двойник", "двойники", "двойниках"},
            "iot": {"internet", "things", "интернет", "вещей"},
            "метавселен": {"metaverse"},
        }
        for key, values in aliases.items():
            if key in lowered:
                terms.update(values)
        return terms

    @staticmethod
    def _is_usable_answer(answer: str, question: str) -> bool:
        stripped = answer.strip()
        if len(stripped) < 80:
            return False
        if LLMService._is_russian(question) and not LLMService._looks_like_russian_answer(stripped):
            return False
        bad_chars = sum(1 for char in stripped if char in "#$%&*+/<=>@\\^_|~")
        if bad_chars / max(1, len(stripped)) > 0.08:
            return False
        if not any(marker in stripped for marker in ("Ответ", "Ключевые пункты", "Answer", "Key points")):
            return False
        return True

    @staticmethod
    def _normalize_answer_text(answer: str) -> str:
        return answer.replace("Ключевые пункы", "Ключевые пункты")

    @staticmethod
    def _fallback_answer(question: str, hits: list[RetrievalHit], graph_context: list[dict]) -> str:
        """Deterministic retrieval summary used only by offline evaluation scripts."""
        selected_hits = LLMService._select_evidence_hits(hits)
        evidence_points = []
        key_phrases = []
        concepts = LLMService._extract_concepts(selected_hits)
        for hit in selected_hits[:3]:
            snippet = hit.text.replace("\n", " ").strip()
            if snippet == "Matched through graph entities.":
                continue
            cleaned = LLMService._clean_snippet(snippet)
            if cleaned:
                evidence_points.append(f"- {cleaned}")
            key_phrases.extend(LLMService._extract_key_phrases(hit))

        if LLMService._is_russian(question):
            short_answer = LLMService._build_fallback_summary_ru(concepts, key_phrases)
            bullet_points = LLMService._build_fallback_points_ru(concepts, evidence_points)
            sections = [
                f"Ответ: {short_answer}",
                "Ключевые пункты:",
                *(bullet_points or ["- Прямые текстовые фрагменты по вопросу ограничены, поэтому вывод основан на ближайшем релевантном контексте."]),
            ]
            return "\n".join(sections)

        short_answer = (
            "The available papers do not always provide a single explicit answer, "
            "but the closest sources still support a focused summary."
        )
        if key_phrases:
            phrase_text = ", ".join(dict.fromkeys(key_phrases[:5]))
            short_answer = (
                "The closest sources connect the question to the following themes: "
                f"{phrase_text}."
            )
        sections = [
            f"Answer: {short_answer}",
            "Key points:",
            *(evidence_points or ["- Direct textual evidence is limited, so the answer is based on the closest relevant papers."]),
        ]
        return "\n".join(sections)

    @staticmethod
    def _select_evidence_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
        content_hits = [hit for hit in hits if hit.text.strip() != "Matched through graph entities."]
        graph_hits = [hit for hit in hits if hit.text.strip() == "Matched through graph entities."]
        ranked = sorted(content_hits, key=lambda item: item.score, reverse=True)
        if graph_hits:
            ranked.extend(sorted(graph_hits, key=lambda item: item.score, reverse=True)[:2])
        return ranked[:5] if ranked else hits[:5]

    @staticmethod
    def _is_russian(text: str) -> bool:
        return any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in text)

    @staticmethod
    def _target_language(question: str) -> str:
        return "Russian" if LLMService._is_russian(question) else "English"

    @staticmethod
    def _extract_key_phrases(hit: RetrievalHit) -> list[str]:
        matched_entities = hit.metadata.get("matched_entities", [])
        if matched_entities:
            normalized = [LLMService._normalize_phrase(str(item)) for item in matched_entities[:6]]
            return [item for item in dict.fromkeys(normalized) if item][:4]
        lowered_title = hit.title.lower()
        phrases = []
        for token in ("blockchain", "digital twin", "metaverse", "low-code", "6g", "smart city", "iot"):
            if token in lowered_title:
                phrases.append(LLMService._normalize_phrase(token))
        return phrases

    @staticmethod
    def _looks_like_russian_answer(answer: str) -> bool:
        cyrillic_chars = sum(1 for char in answer if "а" <= char.lower() <= "я" or char.lower() == "ё")
        latin_chars = sum(1 for char in answer if "a" <= char.lower() <= "z")
        if cyrillic_chars == 0:
            return False
        return cyrillic_chars >= max(20, latin_chars * 0.35)

    @staticmethod
    def _clean_snippet(snippet: str) -> str:
        cleaned = " ".join(snippet.split())
        cleaned = cleaned.replace("[", "").replace("]", "")
        cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
        cleaned = cleaned.replace("•", " ")
        cleaned = " ".join(cleaned.split())
        sentences = [item.strip() for item in cleaned.split(".") if item.strip()]
        if sentences:
            cleaned = sentences[0]
        cleaned = cleaned[:220].rstrip(",;:- ")
        return cleaned + ("." if cleaned and not cleaned.endswith(".") else "")

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        lowered = text.strip().lower()
        alias_map = {
            "blockchain": "блокчейн",
            "blockchain technology": "блокчейн",
            "distributed ledger": "распределенный реестр",
            "digital twin": "цифровые двойники",
            "digital twins": "цифровые двойники",
            "low-code platforms": "low-code платформы",
            "low-code development": "low-code разработка",
            "smart city": "умный город",
            "smart cities": "умные города",
            "internet of things": "iot",
        }
        return alias_map.get(lowered, lowered)

    @staticmethod
    def _extract_concepts(hits: list[RetrievalHit]) -> list[str]:
        text = " ".join(f"{hit.title} {hit.text}" for hit in hits[:5]).lower()
        concept_aliases = {
            "блокчейн": ["блокчейн", "blockchain", "distributed ledger", "dlt"],
            "цифровые двойники": ["цифров", "digital twin", "digital twins", "dt)"],
            "6g": ["6g"],
            "edge ai": ["edge ai", "пограничн", "искусственн", "artificial intelligence"],
            "iot": ["iot", "интернет вещей", "internet of things"],
            "прогнозная аналитика": ["прогноз", "predictive"],
            "безопасный обмен данными": ["безопас", "security", "trusted", "integrity"],
            "интероперабельность": ["interoperab", "совместим", "совместн"],
            "low-code": ["low-code", "low code", "низк", "малым кодом"],
            "тестирование и контроль качества": ["testing", "quality assurance", "качество", "тест"],
            "impact analysis": ["impact analysis", "эволюц", "version", "изменен"],
        }
        found = []
        for concept, aliases in concept_aliases.items():
            if any(alias in text for alias in aliases):
                found.append(concept)
        return found

    @staticmethod
    def _build_fallback_summary_ru(concepts: list[str], key_phrases: list[str]) -> str:
        topics = list(dict.fromkeys([*concepts, *key_phrases]))
        if topics:
            topic_text = ", ".join(topics[:5])
            return (
                "По найденным фрагментам можно сделать ограниченный, но предметный вывод: "
                f"основной контекст ответа связан с темами {topic_text}. "
                "Ниже приведены ключевые наблюдения, извлечённые из наиболее релевантных фрагментов корпуса."
            )
        return (
            "По найденным фрагментам можно сделать только ограниченный вывод: "
            "релевантный контекст есть, но он не содержит достаточно явного объяснения для полного ответа."
        )

    @staticmethod
    def _build_fallback_points_ru(concepts: list[str], evidence_points: list[str]) -> list[str]:
        if evidence_points:
            return evidence_points[:4]
        if concepts:
            return [f"- В найденном контексте выделяется тема: {concept}." for concept in concepts[:4]]
        return ["- Прямые текстовые фрагменты по вопросу ограничены, поэтому вывод требует уточнения запроса или расширения корпуса."]
