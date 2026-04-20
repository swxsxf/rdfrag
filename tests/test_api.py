import numpy as np
from fastapi.testclient import TestClient

from rdfrag_vkr.api.app import create_app
from rdfrag_vkr.config import Settings
from rdfrag_vkr.modules.vector_retriever import VectorRetriever
from rdfrag_vkr.utils.io import write_json, write_jsonl


def test_health_and_query_endpoints(tmp_path):
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    (settings.raw_pdf_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")
    retriever = VectorRetriever(settings)
    chunk_row = {
        "chunk_id": "doc-1-chunk-0",
        "doc_id": "doc-1",
        "source_file": "sample.pdf",
        "title": "Blockchain article",
        "text": "Blockchain in digital economy.",
        "chunk_index": 0,
        "token_estimate": 4,
    }
    vector = retriever._hash_embed_text(chunk_row["text"])
    write_json(settings.embeddings_dir / "vector_index.json", {"backend": "hash", "vector_dim": len(vector), "count": 1})
    write_jsonl(settings.embeddings_dir / "chunks.jsonl", [chunk_row])
    np.save(settings.embeddings_dir / "vector.npy", np.vstack([vector]).astype("float32"))
    write_jsonl(
        settings.rdf_dir / "knowledge_documents.jsonl",
        [
            {
                "metadata": {
                    "doc_id": "doc-1",
                    "source_file": "sample.pdf",
                    "title": "Blockchain article",
                    "authors": ["Alice"],
                    "year": 2024,
                    "abstract": None,
                    "language": None,
                    "page_count": 1,
                    "parser": "pypdf",
                },
                "entities": [
                    {
                        "entity_id": "topic-1",
                        "entity_type": "Topic",
                        "label": "blockchain",
                        "evidence": "blockchain",
                        "normalized_label": "blockchain",
                    }
                ],
                "relations": [],
            }
        ],
    )
    (settings.rdf_dir / "knowledge_graph.ttl").write_text("@prefix rdfrag: <http://example.org/rdfrag/> .", encoding="utf-8")

    client = TestClient(create_app(settings))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["pdf_count"] == 1

    response = client.post("/query", json={"question": "blockchain", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["hits"]
