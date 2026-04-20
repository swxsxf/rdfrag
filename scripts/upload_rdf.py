"""Upload the generated RDF graph to Fuseki."""

from __future__ import annotations

from pathlib import Path

from rdfrag_vkr.modules.sparql_service import SparqlService


if __name__ == "__main__":
    ttl_path = Path("data/rdf/knowledge_graph.ttl")
    success = SparqlService().sync_graph(ttl_path)
    print({"ttl_path": str(ttl_path.resolve()), "synced": success})
