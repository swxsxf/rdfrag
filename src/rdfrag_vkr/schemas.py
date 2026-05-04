"""Shared schemas for the RDFRAG modules."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArticleMetadata(BaseModel):
    """Basic article metadata extracted from PDF or fallback heuristics."""

    doc_id: str
    source_file: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    language: str | None = None
    page_count: int = 0
    parser: str = "pypdf"


class ParsedDocument(BaseModel):
    """Structured output of PDF parsing."""

    metadata: ArticleMetadata
    text: str
    pages: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """Preprocessed chunk used for retrieval."""

    chunk_id: str
    doc_id: str
    source_file: str
    title: str
    text: str
    chunk_index: int
    token_estimate: int


class Entity(BaseModel):
    """Knowledge entity extracted from a scientific article."""

    entity_id: str
    entity_type: str
    label: str
    evidence: str | None = None
    normalized_label: str | None = None


class Relation(BaseModel):
    """Relation between article and extracted entity."""

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    evidence: str | None = None


class KnowledgeDocument(BaseModel):
    """Knowledge representation for one article."""

    metadata: ArticleMetadata
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    """Single retrieval result."""

    doc_id: str
    chunk_id: str | None = None
    title: str
    score: float
    source_file: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Input model for /query endpoint."""

    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Output model for /query endpoint."""

    question: str
    answer: str
    hits: list[RetrievalHit]
    graph_context: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Output model for /health endpoint."""

    status: str
    pdf_count: int
    parsed_count: int
    chunk_files_count: int
    rdf_exists: bool
    vector_index_exists: bool
    grobid_available: bool = False
    fuseki_available: bool = False
    ollama_available: bool = False
