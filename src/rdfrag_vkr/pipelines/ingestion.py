"""End-to-end ingestion pipeline for the diploma MVP."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import logging
import time

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.knowledge_llm import OllamaKnowledgeExtractor
from rdfrag_vkr.modules.ner import RuleBasedNER
from rdfrag_vkr.modules.pdf_parser import PDFParser
from rdfrag_vkr.modules.preprocessing import Preprocessor
from rdfrag_vkr.modules.rdf_builder import RDFBuilder
from rdfrag_vkr.modules.relation_extraction import RuleBasedRelationExtractor
from rdfrag_vkr.modules.sparql_service import SparqlService
from rdfrag_vkr.modules.vector_retriever import VectorRetriever
from rdfrag_vkr.schemas import KnowledgeDocument
from rdfrag_vkr.utils.artifacts import write_json_summary
from rdfrag_vkr.utils.io import write_json


def run_ingestion(settings: Settings | None = None) -> dict:
    """Run the full diploma ingestion pipeline over data/raw_pdfs."""
    logger = logging.getLogger(__name__)
    started_at = time.monotonic()
    app_settings = settings or get_settings()
    parser = PDFParser(app_settings)
    preprocessor = Preprocessor(app_settings)
    ner = RuleBasedNER(app_settings)
    relation_extractor = RuleBasedRelationExtractor(app_settings)
    rdf_builder = RDFBuilder(app_settings)
    sparql_service = SparqlService(app_settings)
    vector_retriever = VectorRetriever(app_settings)
    knowledge_extractor = OllamaKnowledgeExtractor(app_settings) if app_settings.knowledge_backend in {"ollama", "ollama_hybrid"} else None
    stage_metrics: list[dict] = []

    logger.info("Ingestion pipeline started. Project root: %s", app_settings.project_root)
    stage_started_at = time.monotonic()
    parsed_documents = parser.parse_corpus()
    parse_elapsed = time.monotonic() - stage_started_at
    stage_metrics.append({"stage": "parse", "seconds": round(parse_elapsed, 3), "count": len(parsed_documents)})
    logger.info(
        "Parsing stage completed: %s documents parsed in %.1fs.",
        len(parsed_documents),
        parse_elapsed,
    )
    stage_started_at = time.monotonic()
    chunks = preprocessor.process_corpus(parsed_documents)
    preprocess_elapsed = time.monotonic() - stage_started_at
    stage_metrics.append({"stage": "preprocess", "seconds": round(preprocess_elapsed, 3), "count": len(chunks)})
    logger.info(
        "Preprocessing stage completed: %s chunks built in %.1fs.",
        len(chunks),
        preprocess_elapsed,
    )

    knowledge_documents: list[KnowledgeDocument] = []
    total_documents = len(parsed_documents)
    stage_started_at = time.monotonic()
    for index, document in enumerate(parsed_documents, start=1):
        item_started_at = time.monotonic()
        llm_payload = knowledge_extractor.extract(document) if knowledge_extractor is not None else None
        entities = ner.extract(document, llm_payload=llm_payload)
        relations = relation_extractor.extract(document, entities, llm_payload=llm_payload)
        knowledge_documents.append(
            KnowledgeDocument(metadata=document.metadata, entities=entities, relations=relations)
        )
        if index == 1 or index % 10 == 0 or index == total_documents:
            knowledge_backend = llm_payload.get("backend", "disabled") if llm_payload is not None else "disabled"
            logger.info(
                "[KNOWLEDGE %s/%s | remaining=%s] %s | backend=%s | entities=%s | relations=%s | item_elapsed=%.1fs | total_elapsed=%.1fs",
                index,
                total_documents,
                total_documents - index,
                document.metadata.source_file,
                knowledge_backend,
                len(entities),
                len(relations),
                time.monotonic() - item_started_at,
                time.monotonic() - started_at,
            )
    knowledge_elapsed = time.monotonic() - stage_started_at
    stage_metrics.append(
        {"stage": "knowledge_extraction", "seconds": round(knowledge_elapsed, 3), "count": len(knowledge_documents)}
    )

    stage_started_at = time.monotonic()
    ttl_path = rdf_builder.build_corpus_graph(knowledge_documents)
    rdf_elapsed = time.monotonic() - stage_started_at
    stage_metrics.append({"stage": "rdf_build", "seconds": round(rdf_elapsed, 3), "count": 1})
    logger.info("RDF stage completed: graph saved to %s", ttl_path)

    stage_started_at = time.monotonic()
    fuseki_uploaded = sparql_service.sync_graph(ttl_path)
    fuseki_elapsed = time.monotonic() - stage_started_at
    stage_metrics.append({"stage": "fuseki_sync", "seconds": round(fuseki_elapsed, 3), "count": int(fuseki_uploaded)})
    if app_settings.fuseki_mode == "required" and not fuseki_uploaded:
        raise RuntimeError(
            f"Fuseki sync is required but failed for {ttl_path}. "
            "Check docker compose services and dataset URL before continuing."
        )
    logger.info("Fuseki sync completed: uploaded=%s", fuseki_uploaded)

    stage_started_at = time.monotonic()
    vector_index = vector_retriever.build_index(chunks)
    vector_elapsed = time.monotonic() - stage_started_at
    stage_metrics.append({"stage": "vector_index", "seconds": round(vector_elapsed, 3), "count": len(chunks)})
    logger.info(
        "Vector stage completed: backend=%s, elapsed=%.1fs.",
        vector_index["backend"],
        vector_elapsed,
    )

    summary = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pdf_count": len(parser.list_pdfs()),
        "parsed_count": len(parsed_documents),
        "chunk_count": len(chunks),
        "knowledge_document_count": len(knowledge_documents),
        "rdf_path": str(ttl_path),
        "vector_backend": vector_index["backend"],
        "grobid_mode": app_settings.grobid_mode,
        "fuseki_uploaded": fuseki_uploaded,
        "knowledge_backend": app_settings.knowledge_backend,
        "llm_provider": app_settings.llm_provider,
        "llm_model": app_settings.ollama_model if app_settings.llm_provider == "ollama" else None,
    }
    write_json(app_settings.eval_dir / "pipeline_summary.json", summary)
    _save_ingestion_artifacts(app_settings, summary, stage_metrics)
    logger.info("Ingestion pipeline finished in %.1fs.", time.monotonic() - started_at)
    return summary


def _save_ingestion_artifacts(settings: Settings, summary: dict, stage_metrics: list[dict]) -> None:
    """Persist pipeline metrics, model manifests, and runtime plots."""
    metrics_csv = settings.artifacts_metrics_csv_dir / "pipeline_stage_metrics.csv"
    metrics_json = settings.artifacts_metrics_json_dir / "pipeline_summary.json"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "seconds", "count"])
        writer.writeheader()
        writer.writerows(stage_metrics)
    write_json_summary(metrics_json, {"summary": summary, "stage_metrics": stage_metrics})

    baseline_manifest = {
        "name": "baseline",
        "description": "Pattern-based extraction with lexical/hash fallback retrieval.",
        "vector_backend": "hash_or_lexical",
        "knowledge_backend": "pattern",
    }
    tuned_manifest = {
        "name": "tuned",
        "description": "GROBID-first parsing, Ollama-assisted extraction, FAISS retrieval, Fuseki graph sync.",
        "vector_backend": summary.get("vector_backend"),
        "knowledge_backend": summary.get("knowledge_backend"),
        "llm_provider": summary.get("llm_provider"),
        "llm_model": summary.get("llm_model"),
    }
    write_json_summary(settings.artifacts_models_baseline_dir / "model_manifest.json", baseline_manifest)
    write_json_summary(settings.artifacts_models_tuned_dir / "model_manifest.json", tuned_manifest)

    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(10, 5))
        labels = [row["stage"] for row in stage_metrics]
        values = [row["seconds"] for row in stage_metrics]
        plt.bar(labels, values, color="#2563eb")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Seconds")
        plt.title("Pipeline Stage Runtime")
        plt.tight_layout()
        figure.savefig(settings.artifacts_plots_training_dir / "pipeline_stage_runtime.png", dpi=200)
        plt.close(figure)
    except Exception:
        pass
