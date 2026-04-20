"""Hybrid retrieval combining graph and vector signals."""

from __future__ import annotations

import re

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.sparql_service import SparqlService
from rdfrag_vkr.modules.vector_retriever import VectorRetriever
from rdfrag_vkr.schemas import RetrievalHit


QUERY_EXPANSIONS = {
    "цифровой двойник": ["digital twin", "digital twins"],
    "цифровые двойники": ["digital twin", "digital twins"],
    "умный город": ["smart city", "smart cities"],
    "низкий код": ["low-code", "low code", "low-code platforms"],
    "метавселенная": ["metaverse"],
    "блокчейн": ["blockchain"],
    "интернет вещей": ["internet of things", "iot"],
    "умный транспорт": ["smart transportation", "smart mobility"],
    "цифровая экономика": ["digital economy"],
}


class HybridRetriever:
    """Combine graph retrieval and vector retrieval for MVP answers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sparql_service = SparqlService(self.settings)
        self.vector_retriever = VectorRetriever(self.settings)

    def search(self, query: str, top_k: int = 5) -> tuple[list[RetrievalHit], list[dict]]:
        """Merge graph and vector results into a final ranked result set."""
        expanded_query = self._expand_query(query)
        query_terms = self._query_terms(expanded_query)
        graph_hits = self.sparql_service.search_articles_by_keyword(expanded_query, limit=top_k * 2)
        vector_hits = self.vector_retriever.search(expanded_query, top_k=top_k * 3)

        merged: dict[str, RetrievalHit] = {}
        for rank, hit in enumerate(vector_hits):
            text_overlap = self._overlap_score(query_terms, f"{hit.title} {hit.text[:600]}")
            title_overlap = self._overlap_score(query_terms, hit.title) * 0.8
            combined_score = hit.score + max(0.0, 1 - rank * 0.05) + text_overlap + title_overlap
            merged[hit.doc_id] = hit.model_copy(
                update={
                    "score": combined_score,
                    "metadata": {
                        **hit.metadata,
                        "expanded_query": expanded_query,
                        "title_overlap": round(title_overlap, 3),
                        "text_overlap": round(text_overlap, 3),
                    },
                }
            )

        for graph_hit in graph_hits:
            existing = merged.get(graph_hit["doc_id"])
            matched_entities = graph_hit.get("matched_entities", [])
            entity_overlap = len(matched_entities) * 0.8
            title_overlap = self._overlap_score(query_terms, graph_hit["title"])
            empty_graph_penalty = -1.0 if not matched_entities else 0.0
            boost = graph_hit["score"] * 0.35 + entity_overlap + title_overlap + empty_graph_penalty
            if existing is None:
                merged[graph_hit["doc_id"]] = RetrievalHit(
                    doc_id=graph_hit["doc_id"],
                    chunk_id=None,
                    title=graph_hit["title"],
                    score=boost,
                    source_file=graph_hit["source_file"],
                    text="Matched through graph entities.",
                    source="graph",
                    metadata={
                        "matched_entities": matched_entities,
                        "expanded_query": expanded_query,
                        "graph_title_overlap": round(title_overlap, 3),
                    },
                )
            else:
                existing_matched = existing.metadata.get("matched_entities", [])
                merged[graph_hit["doc_id"]] = existing.model_copy(
                    update={
                        "score": existing.score + boost + 0.75,
                        "metadata": {
                            **existing.metadata,
                            "matched_entities": list(dict.fromkeys([*existing_matched, *matched_entities])),
                            "graph_title_overlap": round(title_overlap, 3),
                            "hybrid_boosted": True,
                        },
                    }
                )

        hits = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]
        return hits, graph_hits

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {token for token in re.findall(r"[\w-]+", query.lower()) if len(token) > 2}

    def _expand_query(self, query: str) -> str:
        lowered = query.lower()
        additions: list[str] = []
        for source, aliases in QUERY_EXPANSIONS.items():
            if source in lowered:
                additions.extend(alias for alias in aliases if alias not in lowered)
        if not additions:
            return query
        return query + " " + " ".join(dict.fromkeys(additions))

    def _overlap_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = self._query_terms(text)
        if not text_terms:
            return 0.0
        overlap = len(query_terms & text_terms)
        return overlap / max(1, len(query_terms))
