"""SPARQL client, Fuseki uploader and graph retrieval helpers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib import error, request

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.utils.io import read_jsonl

try:  # pragma: no cover - optional dependency
    from rdflib import Graph
except ImportError:  # pragma: no cover
    Graph = None


class SparqlService:
    """Minimal wrapper around a local RDF file and a Fuseki endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def query_local_graph(self, sparql_query: str) -> list[dict]:
        """Run SPARQL against the local TTL if rdflib is installed."""
        ttl_path = self.settings.rdf_dir / "knowledge_graph.ttl"
        if Graph is None:
            raise RuntimeError("rdflib is not installed. Install optional dependencies to enable local SPARQL.")
        if not ttl_path.exists():
            return []
        graph = Graph()
        graph.parse(ttl_path)
        rows: list[dict] = []
        for row in graph.query(sparql_query):
            rows.append({str(key): str(value) for key, value in row.asdict().items()})
        return rows

    def search_articles_by_keyword(self, keyword: str, limit: int = 5) -> list[dict]:
        """Search graph context via Fuseki first, then local SPARQL, then JSONL heuristics."""
        if self.is_fuseki_available():
            rows = self._search_via_fuseki(keyword)
            if rows:
                return rows[:limit]
        rows = self._search_via_local_sparql(keyword)
        if rows:
            return rows[:limit]
        documents = read_jsonl(self.settings.rdf_dir / "knowledge_documents.jsonl")
        query_tokens = self._query_tokens(keyword)
        hits: list[dict] = []
        for document in documents:
            title = document["metadata"]["title"]
            source_file = document["metadata"]["source_file"]
            score = 0.0
            matched_entities: list[str] = []
            title_lower = title.lower()
            for token in query_tokens:
                if token in title_lower:
                    score += 1.5
            for entity in document.get("entities", []):
                label = entity["label"]
                label_lower = label.lower()
                if any(token in label_lower for token in query_tokens):
                    score += 1.0
                    matched_entities.append(label)
            if score > 0:
                hits.append(
                    {
                        "doc_id": document["metadata"]["doc_id"],
                        "title": title,
                        "source_file": source_file,
                        "score": score,
                        "matched_entities": matched_entities,
                    }
                )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:limit]

    def _search_via_local_sparql(self, keyword: str) -> list[dict]:
        if Graph is None or not (self.settings.rdf_dir / "knowledge_graph.ttl").exists():
            return []
        token_rows: dict[str, dict] = {}
        for token in self._query_tokens(keyword):
            query = f"""
            PREFIX rdfrag: <http://example.org/rdfrag/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?article ?title ?source_file ?label
            WHERE {{
              ?article rdf:type rdfrag:Article ;
                       rdfrag:title ?title ;
                       rdfrag:sourceFile ?source_file .
              OPTIONAL {{
                ?article ?predicate ?entity .
                ?entity rdfrag:label ?label .
              }}
              FILTER(
                CONTAINS(LCASE(STR(?title)), "{token}") ||
                (BOUND(?label) && CONTAINS(LCASE(STR(?label)), "{token}"))
              )
            }}
            """
            for row in self.query_local_graph(query):
                self._merge_graph_row(token_rows, row, token)
        return sorted(token_rows.values(), key=lambda item: item["score"], reverse=True)

    def _search_via_fuseki(self, keyword: str) -> list[dict]:
        token_rows: dict[str, dict] = {}
        for token in self._query_tokens(keyword):
            query = f"""
            PREFIX rdfrag: <http://example.org/rdfrag/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?article ?title ?source_file ?label
            WHERE {{
              ?article rdf:type rdfrag:Article ;
                       rdfrag:title ?title ;
                       rdfrag:sourceFile ?source_file .
              OPTIONAL {{
                ?article ?predicate ?entity .
                ?entity rdfrag:label ?label .
              }}
              FILTER(
                CONTAINS(LCASE(STR(?title)), "{token}") ||
                (BOUND(?label) && CONTAINS(LCASE(STR(?label)), "{token}"))
              )
            }}
            """
            for row in self.query_fuseki(query):
                self._merge_graph_row(token_rows, row, token)
        return sorted(token_rows.values(), key=lambda item: item["score"], reverse=True)

    def upload_ttl(self, ttl_path: Path) -> bool:
        """Append TTL to a Fuseki dataset if the service is available."""
        if not self.ensure_dataset_exists():
            return False
        dataset_data_url = f"{self.settings.fuseki_dataset_url}/data"
        data = ttl_path.read_bytes()
        req = request.Request(
            dataset_data_url,
            data=data,
            headers={"Content-Type": "text/turtle"},
            method="POST",
        )
        self._add_admin_auth(req)
        try:
            with request.urlopen(req, timeout=10) as response:
                return 200 <= response.status < 300
        except error.URLError:
            return False

    def clear_dataset(self) -> bool:
        """Remove the current default graph from Fuseki."""
        if not self.ensure_dataset_exists():
            return False
        endpoint = f"{self.settings.fuseki_dataset_url}/update"
        req = request.Request(
            endpoint,
            data="CLEAR DEFAULT".encode("utf-8"),
            headers={"Content-Type": "application/sparql-update"},
            method="POST",
        )
        self._add_admin_auth(req)
        try:
            with request.urlopen(req, timeout=10) as response:
                return 200 <= response.status < 300
        except error.URLError:
            return False

    def sync_graph(self, ttl_path: Path) -> bool:
        """Replace the Fuseki dataset content with the local TTL graph."""
        if not ttl_path.exists() or not self.wait_for_fuseki():
            return False
        if not self.clear_dataset():
            return False
        return self.upload_ttl(ttl_path)

    def is_fuseki_available(self) -> bool:
        """Check whether the configured Fuseki dataset responds."""
        if not self.ensure_dataset_exists():
            return False
        ping_url = f"{self.settings.fuseki_dataset_url}/query?query=ASK%20WHERE%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D"
        req = request.Request(ping_url, headers={"Accept": "application/sparql-results+json"}, method="GET")
        try:
            with request.urlopen(req, timeout=5) as response:
                return 200 <= response.status < 300
        except error.URLError:
            return False

    def wait_for_fuseki(self, timeout_seconds: int = 60) -> bool:
        """Wait for the Fuseki endpoint to become reachable."""
        started = time.monotonic()
        while time.monotonic() - started < timeout_seconds:
            if self.is_fuseki_available():
                return True
            time.sleep(2)
        return False

    def query_fuseki(self, sparql_query: str) -> list[dict]:
        """Run a remote SPARQL query against Fuseki."""
        if not self.ensure_dataset_exists():
            return []
        endpoint = f"{self.settings.fuseki_dataset_url}/query"
        body = sparql_query.encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
            method="POST",
        )
        self._add_admin_auth(req)
        try:
            with request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.URLError:
            return []
        bindings = payload.get("results", {}).get("bindings", [])
        rows: list[dict] = []
        for binding in bindings:
            rows.append({key: value.get("value") for key, value in binding.items()})
        return rows

    def ensure_dataset_exists(self) -> bool:
        """Create the configured dataset if it does not exist yet."""
        dataset_name = self._dataset_name()
        existing = self.list_datasets()
        if dataset_name in existing:
            return True

        payload = urlencode({"dbName": dataset_name, "dbType": "tdb2"}).encode("utf-8")
        req = request.Request(
            self._admin_datasets_url(),
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        self._add_admin_auth(req)
        try:
            with request.urlopen(req, timeout=10) as response:
                return 200 <= response.status < 300
        except error.HTTPError as exc:
            return exc.code == 409
        except error.URLError:
            return False

    def list_datasets(self) -> list[str]:
        """Return Fuseki dataset names from the admin API."""
        req = request.Request(self._admin_datasets_url(), method="GET")
        self._add_admin_auth(req)
        try:
            with request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.URLError:
            return []
        names: list[str] = []
        for item in payload.get("datasets", []):
            dataset_name = item.get("ds.name", "")
            if dataset_name:
                names.append(str(dataset_name).strip("/"))
        return names

    @staticmethod
    def _query_tokens(keyword: str) -> list[str]:
        tokens = [token for token in re.findall(r"[\w-]+", keyword.lower()) if len(token) > 2]
        return tokens or [keyword.lower()]

    @staticmethod
    def _merge_graph_row(rows: dict[str, dict], row: dict, token: str) -> None:
        article_uri = row.get("article", "")
        doc_id = article_uri.rsplit("/", maxsplit=1)[-1]
        label = row.get("label", "")
        item = rows.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": row.get("title", ""),
                "source_file": row.get("source_file", ""),
                "score": 0.0,
                "matched_entities": [],
            },
        )
        if token in item["title"].lower():
            item["score"] += 1.5
        if label and token in label.lower():
            item["score"] += 1.0
            if label not in item["matched_entities"]:
                item["matched_entities"].append(label)

    def _admin_datasets_url(self) -> str:
        parsed = urlparse(self.settings.fuseki_dataset_url)
        return f"{parsed.scheme}://{parsed.netloc}/$/datasets"

    def _dataset_name(self) -> str:
        return urlparse(self.settings.fuseki_dataset_url).path.strip("/").split("/")[-1]

    def _add_admin_auth(self, req: request.Request) -> None:
        import base64

        credentials = f"{self.settings.fuseki_admin_user}:{self.settings.fuseki_admin_password}".encode("utf-8")
        req.add_header("Authorization", "Basic " + base64.b64encode(credentials).decode("ascii"))
