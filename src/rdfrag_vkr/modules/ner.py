"""Rule-based NER for the diploma MVP."""

from __future__ import annotations

import re

import spacy

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.knowledge_llm import OllamaKnowledgeExtractor
from rdfrag_vkr.schemas import Entity, ParsedDocument
from rdfrag_vkr.utils.io import make_entity_id


TOPIC_KEYWORDS = [
    "digital economy",
    "цифровая экономика",
    "metaverse",
    "метавселен",
    "blockchain",
    "блокчейн",
    "internet of things",
    "iot",
    "интернет вещей",
    "digital twin",
    "цифровой двойник",
    "smart city",
    "умный город",
    "web3",
    "supply chain",
    "цепочк",
]

METHOD_KEYWORDS = [
    "survey",
    "review",
    "case study",
    "systematic review",
    "machine learning",
    "deep learning",
    "federated learning",
    "ontology",
    "semantic web",
    "bibliometric",
    "swot",
]

DATASET_KEYWORDS = [
    "dataset",
    "benchmark",
    "hybrid dataset",
    "corpus",
    "data set",
    "набора данных",
]

METRIC_KEYWORDS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency",
    "throughput",
    "qos",
    "qoe",
]


class RuleBasedNER:
    """Minimal rule-based entity extractor with extendable interfaces."""

    def extract(self, document: ParsedDocument) -> list[Entity]:
        """Extract a minimal set of entities from one parsed document."""
        entities: list[Entity] = []
        entities.append(
            Entity(
                entity_id=make_entity_id("Article", document.metadata.doc_id),
                entity_type="Article",
                label=document.metadata.title,
                normalized_label=document.metadata.doc_id,
            )
        )
        for author in document.metadata.authors:
            entities.append(self._entity("Author", author, author))
        if document.metadata.year:
            entities.append(self._entity("Year", str(document.metadata.year), str(document.metadata.year)))

        source_text = f"{document.metadata.title}\n{document.metadata.abstract or ''}\n{document.text[:7000]}"
        entities.extend(self._match_keywords("Topic", TOPIC_KEYWORDS, source_text))
        entities.extend(self._match_keywords("Method", METHOD_KEYWORDS, source_text))
        entities.extend(self._match_keywords("Dataset", DATASET_KEYWORDS, source_text))
        entities.extend(self._match_keywords("Metric", METRIC_KEYWORDS, source_text))
        return self._deduplicate(entities)

    def _match_keywords(self, entity_type: str, keywords: list[str], text: str) -> list[Entity]:
        lowered = text.lower()
        matches: list[Entity] = []
        for keyword in keywords:
            if keyword.lower() in lowered:
                evidence = self._find_evidence(text, keyword)
                matches.append(self._entity(entity_type, keyword, evidence))
        return matches

    @staticmethod
    def _find_evidence(text: str, keyword: str) -> str:
        pattern = re.compile(rf"(.{{0,60}}{re.escape(keyword)}.{{0,60}})", flags=re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if not match:
            return keyword
        return re.sub(r"\s+", " ", match.group(1)).strip()

    @staticmethod
    def _entity(entity_type: str, label: str, evidence: str | None) -> Entity:
        normalized = label.strip().lower()
        return Entity(
            entity_id=make_entity_id(entity_type, normalized),
            entity_type=entity_type,
            label=label.strip(),
            normalized_label=normalized,
            evidence=evidence,
        )

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        unique: dict[str, Entity] = {}
        for entity in entities:
            unique[entity.entity_id] = entity
        return list(unique.values())


class SpacyNERStub:
    """Placeholder for a future spaCy-based NER implementation."""

    def extract(self, document: ParsedDocument) -> list[Entity]:
        """Return an empty result until the spaCy pipeline is integrated."""
        _ = document
        return []


KEYWORDS_BY_TYPE = {
    "Topic": [
        "digital economy",
        "цифровая экономика",
        "metaverse",
        "метавселенная",
        "blockchain",
        "блокчейн",
        "internet of things",
        "iot",
        "интернет вещей",
        "digital twin",
        "цифровой двойник",
        "smart city",
        "умный город",
        "web3",
        "supply chain",
        "цепочки поставок",
    ],
    "Method": [
        "survey",
        "review",
        "case study",
        "systematic review",
        "machine learning",
        "deep learning",
        "federated learning",
        "ontology",
        "semantic web",
        "bibliometric",
        "swot",
        "систематический обзор",
        "онтология",
        "семантический веб",
    ],
    "Dataset": [
        "dataset",
        "benchmark",
        "hybrid dataset",
        "corpus",
        "data set",
        "набор данных",
        "датасет",
    ],
    "Metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "latency",
        "throughput",
        "qos",
        "qoe",
        "точность",
    ],
}


class SpacyPatternNER:
    """spaCy-based pattern NER with thesis-friendly heuristics."""

    def __init__(self) -> None:
        self.nlp = spacy.blank("xx")
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        self.patterns = KEYWORDS_BY_TYPE

    def extract(self, document: ParsedDocument) -> list[Entity]:
        """Extract entities using spaCy sentence segmentation plus keyword patterns."""
        entities: list[Entity] = [
            Entity(
                entity_id=make_entity_id("Article", document.metadata.doc_id),
                entity_type="Article",
                label=document.metadata.title,
                normalized_label=document.metadata.doc_id,
            )
        ]
        for author in document.metadata.authors:
            entities.append(self._entity("Author", author, author))
        if document.metadata.year:
            entities.append(self._entity("Year", str(document.metadata.year), str(document.metadata.year)))

        source_text = f"{document.metadata.title}\n{document.metadata.abstract or ''}\n{document.text[:10000]}"
        doc = self.nlp(source_text)
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        for entity_type, keywords in self.patterns.items():
            for keyword in keywords:
                evidence = self._find_sentence_evidence(sentences, keyword) or self._find_evidence(source_text, keyword)
                if evidence:
                    entities.append(self._entity(entity_type, keyword, evidence))
        return self._deduplicate(entities)

    @staticmethod
    def _find_sentence_evidence(sentences: list[str], keyword: str) -> str | None:
        keyword_lower = keyword.lower()
        for sentence in sentences:
            if keyword_lower in sentence.lower():
                return re.sub(r"\s+", " ", sentence).strip()[:280]
        return None

    @staticmethod
    def _find_evidence(text: str, keyword: str) -> str | None:
        pattern = re.compile(rf"(.{{0,80}}{re.escape(keyword)}.{{0,80}})", flags=re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if not match:
            return None
        return re.sub(r"\s+", " ", match.group(1)).strip()

    @staticmethod
    def _entity(entity_type: str, label: str, evidence: str | None) -> Entity:
        normalized = label.strip().lower()
        return Entity(
            entity_id=make_entity_id(entity_type, normalized),
            entity_type=entity_type,
            label=label.strip(),
            normalized_label=normalized,
            evidence=evidence,
        )

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        unique: dict[str, Entity] = {}
        for entity in entities:
            unique[entity.entity_id] = entity
        return list(unique.values())


class FinalHybridNER(SpacyPatternNER):
    """Pattern NER enriched with cached Ollama entities for the final project."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.llm_extractor = OllamaKnowledgeExtractor(self.settings)

    def extract(self, document: ParsedDocument, llm_payload: dict | None = None) -> list[Entity]:
        entities = super().extract(document)
        if self.settings.knowledge_backend not in {"ollama", "ollama_hybrid"}:
            return entities
        llm_payload = llm_payload or self.llm_extractor.extract(document)
        for row in llm_payload.get("entities", []):
            entities.append(self._entity(row["entity_type"], row["label"], row.get("evidence")))
        return self._deduplicate(entities)


RuleBasedNER = FinalHybridNER
