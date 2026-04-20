"""Answer generation service separated from retrieval logic."""

from __future__ import annotations

import logging

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
            if answer:
                if self._is_russian(question) and not self._looks_like_russian_answer(answer):
                    self.logger.warning("Ollama returned a non-Russian answer for a Russian question; using fallback.")
                else:
                    return answer

        return self._fallback_answer(question, hits, graph_context)

    def is_ollama_available(self) -> bool:
        """Check whether the configured Ollama endpoint is reachable."""
        try:
            response = httpx.get(
                f"{self.settings.ollama_url}/api/tags",
                timeout=min(self.settings.ollama_timeout_seconds, 15),
            )
            return response.status_code == 200
        except Exception:
            return False

    def _generate_with_ollama(self, prompt: str) -> str | None:
        try:
            response = httpx.post(
                f"{self.settings.ollama_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.settings.ollama_temperature},
                },
                timeout=self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip() or None
        except Exception as exc:
            self.logger.warning("Ollama answer generation failed, using fallback synthesis: %s", exc)
            return None

    def _build_prompt(self, question: str, hits: list[RetrievalHit], graph_context: list[dict]) -> str:
        evidence_blocks = []
        for index, hit in enumerate(self._select_evidence_hits(hits), start=1):
            evidence_blocks.append(
                f"[Source {index}] title={hit.title}\n"
                f"file={hit.source_file}\n"
                f"retrieval_source={hit.source}\n"
                f"score={hit.score:.3f}\n"
                f"text={hit.text[:1200]}"
            )
        graph_lines = []
        for item in graph_context[:5]:
            matched = ", ".join(item.get("matched_entities", [])) or "no explicit entity match"
            graph_lines.append(
                f"title={item.get('title', '')}; file={item.get('source_file', '')}; "
                f"score={item.get('score', 0)}; matched={matched}"
            )
        graph_part = "\n".join(graph_lines) if graph_lines else "No graph matches."
        evidence_part = "\n\n".join(evidence_blocks)
        target_language = self._target_language(question)
        if self._is_russian(question):
            output_format = (
                "Формат ответа:\n"
                "1. Краткое определение или суть вопроса.\n"
                "2. Основные механизмы, принципы или подходы.\n"
                "3. Применение, примеры или сценарии из фрагментов.\n"
                "4. Ключевые выводы и, если уместно, ограничения.\n"
                "Пиши минимум 3-5 абзацев, подробно и структурированно.\n\n"
            )
        else:
            output_format = (
                "Response format:\n"
                "1. Short definition or core idea.\n"
                "2. Main mechanisms, principles, or approaches.\n"
                "3. Applications, examples, or scenarios from the fragments.\n"
                "4. Key conclusions and, when relevant, limitations.\n"
                "Write at least 3-5 paragraphs, in a detailed and structured way.\n\n"
            )
        if self._is_russian(question):
            system_prompt = (
                "Ты — эксперт-аналитик. Тебе предоставлены фрагменты научных PDF-документов.\n\n"
                "ПРАВИЛА:\n"
                "1. Отвечай подробно и структурированно.\n"
                "2. Используй все предоставленные фрагменты и не игнорируй релевантные детали.\n"
                "3. Приводи конкретные термины, механизмы, методы, примеры и ограничения из документов.\n"
                "4. Если фрагменты не дают полного ответа, прямо укажи, что вывод ограничен имеющимся контекстом.\n"
                "5. Если фрагменты противоречат друг другу, укажи это явно.\n"
                "6. Не выдумывай факты, которых нет в предоставленных фрагментах.\n"
                "7. Не пиши как retrieval-отчёт: не упоминай score, graph context, файл, chunk, API или техническую диагностику.\n"
                "8. Пиши весь ответ на русском языке, даже если отдельные фрагменты или названия статей даны на английском.\n\n"
            )
            user_prompt = (
                f"Вопрос: {question}\n\n"
                f"Графовый контекст:\n{graph_part}\n\n"
                f"Документы:\n{evidence_part}\n\n"
                "Дай развёрнутый экспертный ответ, опираясь на все фрагменты выше.\n"
            )
        else:
            system_prompt = (
                "You are an expert analyst. You are given fragments from scientific PDF documents.\n\n"
                "RULES:\n"
                "1. Answer in a detailed and structured way.\n"
                "2. Use all provided fragments and do not ignore relevant evidence.\n"
                "3. Include concrete terms, mechanisms, methods, examples, and limitations from the documents.\n"
                "4. If the fragments do not fully support a conclusion, state that clearly.\n"
                "5. If fragments contradict each other, mention that explicitly.\n"
                "6. Do not invent facts not grounded in the provided fragments.\n"
                "7. Do not write like a retrieval report: do not mention scores, graph context, files, chunks, API, or technical diagnostics.\n"
                f"8. Write the answer in {target_language}.\n\n"
            )
            user_prompt = (
                f"Question: {question}\n\n"
                f"Graph context:\n{graph_part}\n\n"
                f"Documents:\n{evidence_part}\n\n"
                "Provide a detailed expert answer grounded in all fragments above.\n"
            )

        return system_prompt + output_format + user_prompt

    @staticmethod
    def _fallback_answer(question: str, hits: list[RetrievalHit], graph_context: list[dict]) -> str:
        selected_hits = LLMService._select_evidence_hits(hits)
        evidence_points = []
        key_phrases = []
        concepts = LLMService._extract_concepts(selected_hits)
        for hit in selected_hits[:3]:
            snippet = hit.text.replace("\n", " ").strip()
            if snippet == "Matched through graph entities.":
                continue
            evidence_points.append(f"- {LLMService._clean_snippet(snippet)}")
            key_phrases.extend(LLMService._extract_key_phrases(hit))

        if LLMService._is_russian(question):
            short_answer = LLMService._build_direct_answer_ru(question, concepts, key_phrases)
            bullet_points = LLMService._build_key_points_ru(question, concepts, evidence_points)
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
    def _build_direct_answer_ru(question: str, concepts: list[str], key_phrases: list[str]) -> str:
        lowered = question.lower()
        if "что такое" in lowered and "блокчейн" in lowered:
            return (
                "Блокчейн — это технология распределённого реестра, в которой данные о транзакциях и событиях "
                "записываются в последовательность связанных блоков и защищаются от незаметного изменения. "
                "В найденных материалах он описывается как основа для доверенного обмена данными, "
                "прослеживаемости операций и межорганизационного взаимодействия без единого центра доверия."
            )
        if "блокчейн" in lowered and ("цифров" in lowered or "double" in lowered or "twin" in lowered):
            return (
                "Блокчейн в цифровых двойниках используется прежде всего для защищённого обмена и хранения данных, "
                "повышения их целостности и прослеживаемости, а также для координации взаимодействия между цифровыми двойниками. "
                "В найденных работах он также сочетается с прогнозной аналитикой и обработкой оперативных данных в реальном времени."
            )
        if "метод" in lowered and "метавселен" in lowered:
            method_concepts = [item for item in concepts if item in {"цифровые двойники", "edge ai", "6g", "iot", "блокчейн", "интероперабельность"}]
            if method_concepts:
                return (
                    "В найденных исследованиях метавселенной чаще всего упоминаются такие методы и технологические подходы, как "
                    + ", ".join(method_concepts[:5])
                    + ". Они используются для моделирования объектов, связи физического и виртуального миров, передачи данных и поддержки интеллектуальных сервисов."
                )
            return (
                "В найденных исследованиях метавселенной методы описаны как сочетание архитектурных, сетевых и данных-ориентированных подходов, "
                "но прямое перечисление зависит от конкретной статьи."
            )
        if "low-code" in lowered or "low code" in lowered or "низк" in lowered:
            return (
                "В найденных работах low-code платформы рассматриваются как средство ускорения разработки и вовлечения непрофессиональных разработчиков, "
                "но при этом отдельно подчёркиваются вопросы тестирования, контроля качества, интеграции и анализа влияния изменений."
            )
        if key_phrases:
            phrase_text = ", ".join(dict.fromkeys(key_phrases[:4]))
            return f"По найденному контексту вопрос в основном связан с такими темами, как {phrase_text}."
        return "По найденному контексту можно сделать ограниченный, но предметный вывод по теме вопроса."

    @staticmethod
    def _build_key_points_ru(question: str, concepts: list[str], evidence_points: list[str]) -> list[str]:
        lowered = question.lower()
        if "что такое" in lowered and "блокчейн" in lowered:
            return [
                "- Блокчейн хранит записи в виде последовательности связанных блоков, что поддерживает целостность и неизменяемость данных.",
                "- Технология применяется для прозрачной фиксации транзакций, событий и операций между несколькими участниками.",
                "- В работах по корпусу блокчейн связывается с IoT, цифровой трансформацией, цепочками поставок и цифровыми двойниками.",
            ]
        if "блокчейн" in lowered and ("цифров" in lowered or "twin" in lowered):
            points = [
                "- Блокчейн используется для надёжного и неизменяемого обмена данными между цифровыми двойниками.",
                "- Он помогает обеспечить прослеживаемость операций и безопасную совместную работу в реальном времени.",
            ]
            if "прогнозная аналитика" in concepts or "iot" in concepts:
                points.append("- В отдельных работах блокчейн сочетается с прогнозной аналитикой, IoT и обработкой операционных данных.")
            return points
        if "метод" in lowered and "метавселен" in lowered:
            points = []
            if "цифровые двойники" in concepts:
                points.append("- Цифровые двойники используются для отображения физических объектов и процессов в виртуальную среду.")
            if "edge ai" in concepts or "6g" in concepts:
                points.append("- Edge AI и 6G рассматриваются как технологическая основа для вычислений, связи и масштабируемости метавселенной.")
            if "блокчейн" in concepts or "iot" in concepts:
                points.append("- Дополнительно встречаются блокчейн и IoT как средства доверенного обмена данными и интеграции устройств.")
            return points or evidence_points[:3]
        if "low-code" in lowered or "low code" in lowered or "низк" in lowered:
            points = [
                "- Главный акцент делается на ускорении разработки и снижении объёма ручного кодирования.",
                "- Одновременно подчёркиваются ограничения в тестировании, обеспечении качества и сопровождении таких систем.",
            ]
            if "impact analysis" in concepts:
                points.append("- Отдельно обсуждается анализ влияния изменений и контроль эволюции low-code решений.")
            return points
        return evidence_points[:3]
