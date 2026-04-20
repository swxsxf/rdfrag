"""LLM-assisted structured extraction for entities and relations."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.schemas import ParsedDocument
from rdfrag_vkr.utils.artifacts import write_json_summary


SUPPORTED_ENTITY_TYPES = {"Author", "Topic", "Method", "Dataset", "Metric", "Year"}
SUPPORTED_PREDICATES = {
    "hasAuthor",
    "hasTopic",
    "mentionsMethod",
    "usesDataset",
    "evaluatedByMetric",
    "publishedInYear",
}


class OllamaKnowledgeExtractor:
    """Extract structured article knowledge with a local Ollama model and caching."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        self.cache_dir = self.settings.artifacts_models_tuned_dir / "knowledge_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._availability_checked_at = 0.0
        self._availability_cache: bool | None = None
        self._disabled_for_run = False
        self._consecutive_failures = 0

    def extract(self, document: ParsedDocument) -> dict:
        """Return cached or freshly generated entity/relation JSON for a document."""
        cache_path = self.cache_dir / f"{document.metadata.doc_id}.json"
        cached_payload = self._read_cache(cache_path)
        if cached_payload is not None:
            return cached_payload

        if self._disabled_for_run or not self._is_available():
            payload = {"entities": [], "relations": [], "backend": "unavailable"}
            write_json_summary(cache_path, payload)
            return payload

        attempts = [
            {"max_entities": 3, "max_relations": 3, "excerpt_chars": 450, "num_predict": min(self.settings.ollama_num_predict, 180)},
            {"max_entities": 3, "max_relations": 3, "excerpt_chars": 350, "num_predict": min(max(self.settings.ollama_num_predict, 220), 220)},
        ]
        try:
            payload = None
            last_error: Exception | None = None
            for attempt in attempts:
                try:
                    payload = self._request_payload(document, **attempt)
                    break
                except Exception as exc:
                    last_error = exc
                    self.logger.warning(
                        "LLM knowledge extraction retry for %s failed: %s",
                        document.metadata.source_file,
                        exc,
                    )
            if payload is None:
                raise last_error or RuntimeError("LLM knowledge extraction returned no payload.")
            self._consecutive_failures = 0
        except Exception as exc:
            if self._should_count_as_connectivity_failure(exc):
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0
            self.logger.warning("LLM knowledge extraction failed for %s: %s", document.metadata.source_file, exc)
            if (
                self._consecutive_failures >= self.settings.knowledge_llm_failures_before_disable
                and self._should_count_as_connectivity_failure(exc)
            ):
                self._disabled_for_run = True
                self._availability_cache = False
                self._availability_checked_at = time.monotonic()
                self.logger.warning(
                    "Ollama extraction disabled for the rest of this run after %s consecutive failures.",
                    self._consecutive_failures,
                )
            payload = {"entities": [], "relations": [], "backend": "error", "error": str(exc)}
            write_json_summary(cache_path, payload)
            return payload

        payload["backend"] = "ollama"
        write_json_summary(cache_path, payload)
        return payload

    def _is_available(self) -> bool:
        now = time.monotonic()
        ttl_seconds = 20 if self._availability_cache else self.settings.ollama_unavailable_ttl_seconds
        if self._availability_cache is not None and now - self._availability_checked_at < ttl_seconds:
            return self._availability_cache
        if self.settings.llm_provider.lower() != "ollama":
            self._availability_cache = False
            self._availability_checked_at = now
            return False
        try:
            response = httpx.get(
                f"{self.settings.ollama_url}/api/tags",
                timeout=min(self.settings.ollama_timeout_seconds, 15),
            )
            self._availability_cache = response.status_code == 200
        except Exception:
            self._availability_cache = False
        self._availability_checked_at = now
        return bool(self._availability_cache)

    def _build_prompt(self, document: ParsedDocument, max_entities: int, max_relations: int, excerpt_chars: int) -> str:
        abstract = (document.metadata.abstract or "").strip()
        abstract = abstract[:250]
        body_excerpt = document.text[: min(self.settings.knowledge_llm_max_chars, excerpt_chars)]
        context = (
            f"Title: {document.metadata.title}\n"
            f"Authors: {', '.join(document.metadata.authors) or 'unknown'}\n"
            f"Year: {document.metadata.year or 'unknown'}\n"
            f"Abstract: {abstract}\n"
            f"Body excerpt: {body_excerpt}"
        )
        return (
            "/no_think\n"
            "Extract compact structured knowledge from the scientific article excerpt.\n"
            "Return one valid JSON object only with keys entities and relations.\n"
            "Focus on semantic article content only.\n"
            "Allowed entity_type values: Topic, Method, Dataset, Metric.\n"
            "Allowed predicate values: hasTopic, mentionsMethod, usesDataset, evaluatedByMetric.\n"
            "Each entity must have keys: entity_type, label, evidence.\n"
            "Each relation must have keys: predicate, object_label, evidence.\n"
            "Use only information explicitly present in the text.\n"
            "Labels must be short, maximum 5 words.\n"
            "Evidence must be short, maximum 80 characters.\n"
            "If a field is unknown, omit it. Do not explain anything. Do not wrap the JSON in markdown.\n"
            f"Return at most {max_entities} entities and at most {max_relations} relations.\n\n"
            f"{context}"
        )

    def _response_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_type": {"type": "string"},
                            "label": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["entity_type", "label", "evidence"],
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "predicate": {"type": "string"},
                            "object_label": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["predicate", "object_label", "evidence"],
                    },
                },
            },
            "required": ["entities", "relations"],
        }

    def _request_payload(
        self,
        document: ParsedDocument,
        max_entities: int,
        max_relations: int,
        excerpt_chars: int,
        num_predict: int,
    ) -> dict:
        prompt = self._build_prompt(document, max_entities=max_entities, max_relations=max_relations, excerpt_chars=excerpt_chars)
        response = httpx.post(
            f"{self.settings.ollama_url}/api/generate",
            json={
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": self._response_schema(),
                "options": {
                    "temperature": 0.0,
                    "num_predict": num_predict,
                    "num_ctx": 4096,
                },
            },
            timeout=self.settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        raw_response = str(data.get("response", "")).strip()
        done_reason = str(data.get("done_reason", "")).strip().lower()
        if done_reason == "length":
            raise RuntimeError("LLM output was truncated by num_predict limit.")
        return self._parse_payload(raw_response)

    def _parse_payload(self, raw_response: str) -> dict:
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        payload = json.loads(self._extract_json_object(cleaned))
        entities = []
        for entity in payload.get("entities", []):
            entity_type = str(entity.get("entity_type", "")).strip()
            label = str(entity.get("label", "")).strip()
            if entity_type in SUPPORTED_ENTITY_TYPES and label and self._is_valid_label(label):
                entities.append(
                    {
                        "entity_type": entity_type,
                        "label": label[:80],
                        "evidence": str(entity.get("evidence", "")).strip() or None,
                    }
                )
        relations = []
        for relation in payload.get("relations", []):
            predicate = str(relation.get("predicate", "")).strip()
            object_label = str(relation.get("object_label", "")).strip()
            if predicate in SUPPORTED_PREDICATES and object_label and self._is_valid_label(object_label):
                relations.append(
                    {
                        "predicate": predicate,
                        "object_label": object_label[:80],
                        "evidence": str(relation.get("evidence", "")).strip() or None,
                    }
                )
        return {"entities": entities, "relations": relations}

    def _read_cache(self, cache_path: Path) -> dict | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        backend = str(payload.get("backend", "")).strip().lower()
        if backend == "ollama":
            return payload
        return None

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text
        return text[start : end + 1]

    @staticmethod
    def _is_valid_label(label: str) -> bool:
        stripped = label.strip()
        if len(stripped) < 2:
            return False
        if stripped.count("!") > 2:
            return False
        if re.fullmatch(r"[\W_]+", stripped):
            return False
        return True

    @staticmethod
    def _should_count_as_connectivity_failure(exc: Exception) -> bool:
        text = str(exc).lower()
        connectivity_markers = (
            "timed out",
            "connection refused",
            "connecterror",
            "readerror",
            "server disconnected",
            "nodename nor servname provided",
            "name or service not known",
            "http 500",
            "500 internal server error",
        )
        return any(marker in text for marker in connectivity_markers)
