"""Rule-based relation extraction for the diploma MVP."""

from __future__ import annotations

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.knowledge_llm import OllamaKnowledgeExtractor
from rdfrag_vkr.schemas import Entity, ParsedDocument, Relation


PREDICATE_BY_ENTITY = {
    "Author": "hasAuthor",
    "Topic": "hasTopic",
    "Method": "mentionsMethod",
    "Dataset": "usesDataset",
    "Metric": "evaluatedByMetric",
    "Year": "publishedInYear",
}


class RuleBasedRelationExtractor:
    """Create simple article-centric relations from extracted entities."""

    def extract(self, document: ParsedDocument, entities: list[Entity]) -> list[Relation]:
        """Build relations from article to each supported entity type."""
        article_entity = next((entity for entity in entities if entity.entity_type == "Article"), None)
        if article_entity is None:
            return []
        relations: list[Relation] = []
        for entity in entities:
            if entity.entity_type == "Article":
                continue
            predicate = PREDICATE_BY_ENTITY.get(entity.entity_type)
            if not predicate:
                continue
            relations.append(
                Relation(
                    relation_id=f"{article_entity.entity_id}-{predicate}-{entity.entity_id}",
                    subject_id=article_entity.entity_id,
                    predicate=predicate,
                    object_id=entity.entity_id,
                    evidence=entity.evidence or document.metadata.title,
                )
            )
        return relations


class LLMRelationExtractorStub:
    """Placeholder for a future LLM-based relation extraction stage."""

    def extract(self, document: ParsedDocument, entities: list[Entity]) -> list[Relation]:
        """Return no relations until the LLM-based stage is implemented."""
        _ = document, entities
        return []


class FinalHybridRelationExtractor(RuleBasedRelationExtractor):
    """Rule-based article relations enriched with cached Ollama relations."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm_extractor = OllamaKnowledgeExtractor(self.settings)

    def extract(self, document: ParsedDocument, entities: list[Entity], llm_payload: dict | None = None) -> list[Relation]:
        relations = super().extract(document, entities)
        if self.settings.knowledge_backend not in {"ollama", "ollama_hybrid"}:
            return relations

        article_entity = next((entity for entity in entities if entity.entity_type == "Article"), None)
        if article_entity is None:
            return relations

        entities_by_label = {entity.label.strip().lower(): entity for entity in entities}
        llm_payload = llm_payload or self.llm_extractor.extract(document)
        for row in llm_payload.get("relations", []):
            object_entity = entities_by_label.get(str(row["object_label"]).strip().lower())
            if object_entity is None:
                continue
            relation_id = f"{article_entity.entity_id}-{row['predicate']}-{object_entity.entity_id}"
            relations.append(
                Relation(
                    relation_id=relation_id,
                    subject_id=article_entity.entity_id,
                    predicate=row["predicate"],
                    object_id=object_entity.entity_id,
                    evidence=row.get("evidence") or object_entity.evidence or document.metadata.title,
                )
            )

        unique: dict[str, Relation] = {}
        for relation in relations:
            unique[relation.relation_id] = relation
        return list(unique.values())


RuleBasedRelationExtractor = FinalHybridRelationExtractor
