"""FastAPI application for the RDFRAG diploma MVP."""

from __future__ import annotations

import json
from fastapi import FastAPI

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.hybrid_retriever import HybridRetriever
from rdfrag_vkr.modules.llm_service import LLMService
from rdfrag_vkr.modules.pdf_parser import PDFParser
from rdfrag_vkr.modules.sparql_service import SparqlService
from rdfrag_vkr.schemas import HealthResponse, QueryRequest, QueryResponse


def _safe_count_pdfs(settings: Settings) -> int:
    """Count PDFs without crashing when a Docker-mounted filesystem is flaky."""
    try:
        count = len(list(settings.raw_pdf_dir.glob("*.pdf")))
        if count > 0:
            return count
    except OSError:
        pass
    manifest_path = settings.parsed_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(payload.get("pdf_count", payload.get("parsed_count", 0)))
    return 0


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a FastAPI app with project services."""
    app_settings = settings or get_settings()
    app = FastAPI(title="RDFRAG VKR MVP", version="0.1.0")
    retriever = HybridRetriever(app_settings)
    llm_service = LLMService(app_settings)
    pdf_parser = PDFParser(app_settings)
    sparql_service = SparqlService(app_settings)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pdf_count = _safe_count_pdfs(app_settings)
        parsed_count = len([path for path in app_settings.parsed_dir.glob("*.json") if path.name != "manifest.json"])
        chunk_files_count = len([path for path in app_settings.chunks_dir.glob("*.jsonl") if path.name != "all_chunks.jsonl"])
        return HealthResponse(
            status="ok",
            pdf_count=pdf_count,
            parsed_count=parsed_count,
            chunk_files_count=chunk_files_count,
            rdf_exists=(app_settings.rdf_dir / "knowledge_graph.ttl").exists(),
            vector_index_exists=(app_settings.embeddings_dir / "vector_index.json").exists(),
            grobid_available=pdf_parser.is_grobid_available(),
            fuseki_available=sparql_service.is_fuseki_available(),
            ollama_available=llm_service.is_ollama_available(),
        )

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        hits, graph_context = retriever.search(payload.question, top_k=payload.top_k)
        answer = llm_service.generate_answer(payload.question, hits, graph_context)
        return QueryResponse(
            question=payload.question,
            answer=answer,
            hits=hits,
            graph_context=graph_context,
        )

    return app


app = create_app()
