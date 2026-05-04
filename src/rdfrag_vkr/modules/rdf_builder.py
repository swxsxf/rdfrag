"""RDF graph builder for extracted article knowledge."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.schemas import KnowledgeDocument
from rdfrag_vkr.utils.io import write_json, write_jsonl

try:  # pragma: no cover - depends on optional package
    from rdflib import Graph, Literal, Namespace, RDF
except ImportError:  # pragma: no cover - exercised via fallback
    Graph = None
    Literal = None
    Namespace = None
    RDF = None


class RDFBuilder:
    """Build RDF triples and persist them to data/rdf."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_uri = "http://example.org/rdfrag/"

    def build_corpus_graph(self, documents: list[KnowledgeDocument]) -> Path:
        """Build RDF for the full corpus and persist TTL plus JSON artifacts."""
        triples = self._build_triples(documents)
        ttl_path = self.settings.rdf_dir / "knowledge_graph.ttl"
        if Graph is not None:
            ttl_text = self._serialize_with_rdflib(documents)
        else:
            ttl_text = self._serialize_fallback(triples)
        ttl_path.write_text(ttl_text, encoding="utf-8")
        write_jsonl(self.settings.rdf_dir / "triples.jsonl", triples)
        write_jsonl(self.settings.rdf_dir / "knowledge_documents.jsonl", [doc.model_dump() for doc in documents])
        write_json(
            self.settings.rdf_dir / "manifest.json",
            {
                "document_count": len(documents),
                "triple_count": len(triples),
                "serializer": "rdflib" if Graph is not None else "fallback_turtle",
            },
        )
        return ttl_path

    def _build_triples(self, documents: list[KnowledgeDocument]) -> list[dict[str, str]]:
        triples: list[dict[str, str]] = []
        for document in documents:
            article_uri = self._resource_uri(document.metadata.doc_id)
            triples.extend(
                [
                    {"subject": article_uri, "predicate": "rdf:type", "object": self._resource_uri("Article")},
                    {"subject": article_uri, "predicate": "title", "object": document.metadata.title},
                    {"subject": article_uri, "predicate": "sourceFile", "object": document.metadata.source_file},
                ]
            )
            if document.metadata.year:
                triples.append(
                    {"subject": article_uri, "predicate": "publishedInYearLiteral", "object": str(document.metadata.year)}
                )
            for entity in document.entities:
                entity_uri = self._resource_uri(entity.entity_id)
                triples.append({"subject": entity_uri, "predicate": "rdf:type", "object": self._resource_uri(entity.entity_type)})
                triples.append({"subject": entity_uri, "predicate": "label", "object": entity.label})
            for relation in document.relations:
                triples.append(
                    {
                        "subject": article_uri,
                        "predicate": relation.predicate,
                        "object": self._resource_uri(relation.object_id),
                    }
                )
        return triples

    def _serialize_with_rdflib(self, documents: list[KnowledgeDocument]) -> str:
        assert Graph is not None and Namespace is not None and Literal is not None and RDF is not None
        graph = Graph()
        ns = Namespace(self.base_uri)
        for document in documents:
            article_ref = ns[quote(document.metadata.doc_id)]
            graph.add((article_ref, RDF.type, ns.Article))
            graph.add((article_ref, ns.title, Literal(document.metadata.title)))
            graph.add((article_ref, ns.sourceFile, Literal(document.metadata.source_file)))
            if document.metadata.year:
                graph.add((article_ref, ns.publishedInYearLiteral, Literal(document.metadata.year)))
            for entity in document.entities:
                entity_ref = ns[quote(entity.entity_id)]
                graph.add((entity_ref, RDF.type, ns[entity.entity_type]))
                graph.add((entity_ref, ns.label, Literal(entity.label)))
            for relation in document.relations:
                graph.add((article_ref, ns[relation.predicate], ns[quote(relation.object_id)]))
        return graph.serialize(format="turtle")

    def _serialize_fallback(self, triples: list[dict[str, str]]) -> str:
        lines = [
            "@prefix rdfrag: <http://example.org/rdfrag/> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "",
        ]
        for triple in triples:
            subject = self._format_node(triple["subject"])
            predicate = "rdf:type" if triple["predicate"] == "rdf:type" else f"rdfrag:{triple['predicate']}"
            object_value = self._format_node(triple["object"])
            lines.append(f"{subject} {predicate} {object_value} .")
        lines.append("")
        return "\n".join(lines)

    def _format_node(self, value: str) -> str:
        if value.startswith("http://example.org/rdfrag/"):
            return f"<{value}>"
        if value.startswith("http://www.w3.org/"):
            return f"<{value}>"
        return json.dumps(value, ensure_ascii=False)

    def _resource_uri(self, value: str) -> str:
        return f"{self.base_uri}{quote(value)}"
