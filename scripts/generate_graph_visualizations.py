"""Generate graph visualizations from the existing RDF knowledge graph."""

from __future__ import annotations

import itertools
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, Literal, RDF, URIRef

from rdfrag_vkr.config import get_settings


BASE_URI = "http://example.org/rdfrag/"
SKIP_PREDICATES = {"title", "sourceFile", "publishedInYearLiteral", "label"}
SEED_GROUPS = {
    "Blockchain": ["blockchain", "блокчейн"],
    "Digital Twins": ["digital twin", "digital twins", "цифров", "digital-twin"],
    "IoT": ["iot", "internet of things", "интернет вещей"],
    "Smart City": ["smart city", "smart cities", "умный город"],
}
TYPE_COLORS = {
    "Article": "#2563eb",
    "Topic": "#06b6d4",
    "Method": "#10b981",
    "Author": "#f59e0b",
    "Metric": "#ef4444",
    "Dataset": "#a855f7",
    "Year": "#64748b",
    "Unknown": "#94a3b8",
}


def local_name(uri: URIRef | str) -> str:
    value = str(uri)
    if value.startswith(BASE_URI):
        value = value.removeprefix(BASE_URI)
    elif "#" in value:
        value = value.rsplit("#", 1)[-1]
    elif "/" in value:
        value = value.rsplit("/", 1)[-1]
    return unquote(value)


def sanitize_text(value: str, width: int = 26, max_lines: int = 3) -> str:
    text = " ".join(value.replace("\n", " ").split()).strip()
    if not text:
        return "Untitled"
    wrapped = textwrap.wrap(text, width=width)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][: max(0, width - 3)].rstrip() + "..."
    return "\n".join(wrapped)


class GraphVisualizer:
    """Build focused graph plots for the diploma report."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.graph = Graph()
        self.graph.parse(self.settings.rdf_dir / "knowledge_graph.ttl", format="turtle")
        self.plot_dir = self.settings.artifacts_plots_dir / "graph"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.settings.artifacts_reports_figures_dir / "graph_visualizations.md"
        plt.rcParams["font.family"] = "DejaVu Sans"
        self.ns = URIRef(BASE_URI)
        self.labels = self._build_label_map()
        self.types = self._build_type_map()
        self.article_nodes = {
            subject
            for subject, _, object_value in self.graph.triples((None, RDF.type, None))
            if local_name(object_value) == "Article"
        }

    def _build_label_map(self) -> dict[URIRef, str]:
        label_map: dict[URIRef, str] = {}
        for predicate_name in ("title", "label", "sourceFile"):
            predicate = URIRef(f"{BASE_URI}{predicate_name}")
            for subject, _, object_value in self.graph.triples((None, predicate, None)):
                if subject in label_map or not isinstance(object_value, Literal):
                    continue
                label_map[subject] = str(object_value)
        for node in set(self.graph.subjects()) | set(self.graph.objects()):
            if isinstance(node, URIRef) and node not in label_map:
                label_map[node] = local_name(node)
        return label_map

    def _build_type_map(self) -> dict[URIRef, str]:
        type_map: dict[URIRef, str] = {}
        for subject, _, object_value in self.graph.triples((None, RDF.type, None)):
            if not isinstance(subject, URIRef):
                continue
            name = local_name(object_value)
            if name == "type":
                continue
            type_map[subject] = name
        for node in self.labels:
            type_map.setdefault(node, "Unknown")
        return type_map

    def _node_type(self, node: URIRef) -> str:
        return self.types.get(node, "Unknown")

    def _node_label(self, node: URIRef, width: int = 24) -> str:
        return sanitize_text(self.labels.get(node, local_name(node)), width=width)

    def _article_neighbors(self, article: URIRef) -> list[tuple[str, URIRef]]:
        neighbors: list[tuple[str, URIRef]] = []
        for _, predicate, object_value in self.graph.triples((article, None, None)):
            predicate_name = local_name(predicate)
            if predicate_name in SKIP_PREDICATES or predicate == RDF.type:
                continue
            if isinstance(object_value, URIRef):
                neighbors.append((predicate_name, object_value))
        return neighbors

    def _find_seed_nodes(self) -> dict[str, list[URIRef]]:
        matches: dict[str, list[URIRef]] = {}
        for group_name, aliases in SEED_GROUPS.items():
            found: list[URIRef] = []
            for node, label in self.labels.items():
                lowered = label.lower()
                if any(alias in lowered for alias in aliases):
                    found.append(node)
            matches[group_name] = found
        return matches

    def _matching_articles(self, seed_nodes: Iterable[URIRef]) -> Counter:
        article_counter: Counter = Counter()
        seed_set = set(seed_nodes)
        for article in self.article_nodes:
            connected = {neighbor for _, neighbor in self._article_neighbors(article)}
            overlap = connected & seed_set
            if overlap:
                article_counter[article] = len(overlap)
        return article_counter

    def _draw_network(
        self,
        graph: nx.Graph | nx.DiGraph,
        output_path: Path,
        *,
        title: str,
        subtitle: str | None = None,
        node_sizes: dict[str, float] | None = None,
        edge_labels: dict[tuple[str, str], str] | None = None,
        edge_widths: dict[tuple[str, str], float] | None = None,
        layout: str = "spring",
    ) -> None:
        if not graph.nodes:
            return

        figure = plt.figure(figsize=(16, 10), facecolor="#070b14")
        axis = plt.gca()
        axis.set_facecolor("#070b14")
        axis.axis("off")
        figure.subplots_adjust(top=0.84, left=0.03, right=0.98, bottom=0.03)

        if layout == "shell":
            pos = nx.shell_layout(graph)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42, k=max(0.55, 1.8 / math.sqrt(max(1, graph.number_of_nodes()))))

        node_colors = [TYPE_COLORS.get(graph.nodes[node].get("node_type", "Unknown"), TYPE_COLORS["Unknown"]) for node in graph.nodes]
        node_size_values = [
            float(node_sizes.get(node, 1800)) if node_sizes else float(graph.nodes[node].get("node_size", 1800))
            for node in graph.nodes
        ]
        edge_width_values = []
        for edge in graph.edges:
            if edge_widths and edge in edge_widths:
                edge_width_values.append(edge_widths[edge])
            elif edge_widths and (edge[1], edge[0]) in edge_widths:
                edge_width_values.append(edge_widths[(edge[1], edge[0])])
            else:
                edge_width_values.append(1.8)

        edge_kwargs = {
            "width": edge_width_values,
            "edge_color": "#64748b",
            "alpha": 0.55,
            "connectionstyle": "arc3,rad=0.08" if isinstance(graph, nx.DiGraph) else "arc3",
        }
        if isinstance(graph, nx.DiGraph):
            edge_kwargs["arrows"] = True
            edge_kwargs["arrowsize"] = 18
        nx.draw_networkx_edges(graph, pos, **edge_kwargs)
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_size_values,
            edgecolors="#e2e8f0",
            linewidths=1.0,
            alpha=0.96,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={node: graph.nodes[node].get("label", node) for node in graph.nodes},
            font_size=9,
            font_color="#e5e7eb",
        )

        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color="#cbd5e1",
                rotate=False,
                bbox={"alpha": 0.0, "pad": 0.1},
            )

        figure.text(
            0.03,
            0.955,
            title,
            fontsize=22,
            color="#f8fafc",
            ha="left",
            va="top",
            fontweight="bold",
        )
        if subtitle:
            figure.text(
                0.03,
                0.895,
                subtitle,
                fontsize=11,
                color="#94a3b8",
                ha="left",
                va="top",
            )
        figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
        plt.close(figure)

    def build_topic_subgraph(self) -> Path:
        seed_matches = self._find_seed_nodes()
        seed_nodes = {node for nodes in seed_matches.values() for node in nodes if self._node_type(node) != "Article"}
        article_counter = self._matching_articles(seed_nodes)
        selected_articles = [node for node, _ in article_counter.most_common(8)]

        graph = nx.DiGraph()
        for group_name, nodes in seed_matches.items():
            for node in nodes[:2]:
                graph.add_node(str(node), label=self._node_label(node), node_type=self._node_type(node), node_size=2300)
        for article in selected_articles:
            article_key = str(article)
            graph.add_node(article_key, label=self._node_label(article, width=22), node_type="Article", node_size=2600)
            for predicate_name, neighbor in self._article_neighbors(article):
                if neighbor not in seed_nodes:
                    continue
                neighbor_key = str(neighbor)
                if neighbor_key not in graph:
                    graph.add_node(neighbor_key, label=self._node_label(neighbor), node_type=self._node_type(neighbor), node_size=2200)
                graph.add_edge(article_key, neighbor_key, predicate=predicate_name)

        output_path = self.plot_dir / "topic_subgraph_blockchain_digital_twins_iot_smart_city.png"
        self._draw_network(
            graph,
            output_path,
            title="Подграф по теме: blockchain, digital twins, IoT, smart city",
            subtitle="Показаны статьи, связанные с ключевыми тематическими узлами графа знаний.",
            edge_labels={(u, v): graph.edges[u, v]["predicate"] for u, v in graph.edges},
            layout="spring",
        )
        return output_path

    def build_single_document_graph(self) -> tuple[Path, str]:
        preferred = None
        candidates = [
            article
            for article in self.article_nodes
            if "blockchain" in self.labels.get(article, "").lower()
            and ("digital twin" in self.labels.get(article, "").lower() or "цифров" in self.labels.get(article, "").lower())
        ]
        if candidates:
            preferred = max(candidates, key=lambda node: len(self._article_neighbors(node)))
        if preferred is None:
            preferred = max(self.article_nodes, key=lambda node: len(self._article_neighbors(node)))

        graph = nx.DiGraph()
        article_key = str(preferred)
        graph.add_node(article_key, label=self._node_label(preferred, width=28), node_type="Article", node_size=3600)
        edge_labels: dict[tuple[str, str], str] = {}
        for predicate_name, neighbor in self._article_neighbors(preferred)[:16]:
            neighbor_key = str(neighbor)
            graph.add_node(
                neighbor_key,
                label=self._node_label(neighbor, width=20),
                node_type=self._node_type(neighbor),
                node_size=2200 if self._node_type(neighbor) != "Author" else 1800,
            )
            graph.add_edge(article_key, neighbor_key)
            edge_labels[(article_key, neighbor_key)] = predicate_name

        output_path = self.plot_dir / "single_document_graph.png"
        self._draw_network(
            graph,
            output_path,
            title="Граф одного документа",
            subtitle=self.labels.get(preferred, local_name(preferred)),
            edge_labels=edge_labels,
            layout="shell",
        )
        return output_path, self.labels.get(preferred, local_name(preferred))

    def build_top_entities_graph(self, top_n: int = 12) -> Path:
        entity_article_sets: dict[URIRef, set[URIRef]] = defaultdict(set)
        for article in self.article_nodes:
            for _, neighbor in self._article_neighbors(article):
                node_type = self._node_type(neighbor)
                if node_type in {"Article", "Year"}:
                    continue
                entity_article_sets[neighbor].add(article)

        top_entities = sorted(entity_article_sets, key=lambda node: len(entity_article_sets[node]), reverse=True)[:top_n]
        top_entity_set = set(top_entities)
        graph = nx.Graph()
        node_sizes: dict[str, float] = {}
        edge_widths: dict[tuple[str, str], float] = {}

        for entity in top_entities:
            key = str(entity)
            article_count = len(entity_article_sets[entity])
            graph.add_node(
                key,
                label=self._node_label(entity),
                node_type=self._node_type(entity),
                node_size=1500 + article_count * 220,
            )
            node_sizes[key] = 1500 + article_count * 220

        co_occurrence: Counter = Counter()
        for article in self.article_nodes:
            article_entities = [neighbor for _, neighbor in self._article_neighbors(article) if neighbor in top_entity_set]
            for first, second in itertools.combinations(sorted(set(article_entities), key=str), 2):
                co_occurrence[(first, second)] += 1

        for (first, second), weight in co_occurrence.items():
            first_key, second_key = str(first), str(second)
            graph.add_edge(first_key, second_key)
            edge_widths[(first_key, second_key)] = 1.0 + weight * 1.2

        output_path = self.plot_dir / "top_entities_cooccurrence.png"
        self._draw_network(
            graph,
            output_path,
            title="Граф top-N сущностей",
            subtitle="Узлы — самые частые сущности графа; ребра показывают совместную встречаемость в статьях.",
            node_sizes=node_sizes,
            edge_widths=edge_widths,
            layout="kamada",
        )
        return output_path

    def build_relation_schema(self) -> Path:
        schema_counts: Counter = Counter()
        for subject, predicate, object_value in self.graph:
            predicate_name = local_name(predicate)
            if predicate_name in SKIP_PREDICATES or predicate == RDF.type:
                continue
            if not isinstance(subject, URIRef) or not isinstance(object_value, URIRef):
                continue
            source_type = self._node_type(subject)
            target_type = self._node_type(object_value)
            schema_counts[(source_type, target_type, predicate_name)] += 1

        graph = nx.DiGraph()
        edge_labels: dict[tuple[str, str], str] = {}
        node_sizes: dict[str, float] = {}
        for source_type, target_type, predicate_name in schema_counts:
            graph.add_node(source_type, label=source_type, node_type=source_type, node_size=2800)
            graph.add_node(target_type, label=target_type, node_type=target_type, node_size=2400)
            if graph.has_edge(source_type, target_type):
                graph.edges[source_type, target_type]["predicates"].append((predicate_name, schema_counts[(source_type, target_type, predicate_name)]))
            else:
                graph.add_edge(
                    source_type,
                    target_type,
                    predicates=[(predicate_name, schema_counts[(source_type, target_type, predicate_name)])],
                )
            node_sizes[source_type] = 3000
            node_sizes[target_type] = 2600

        for source, target in graph.edges:
            predicates = graph.edges[source, target]["predicates"]
            predicates = sorted(predicates, key=lambda item: item[1], reverse=True)
            edge_labels[(source, target)] = "\n".join(f"{name} ({count})" for name, count in predicates[:3])

        output_path = self.plot_dir / "relation_schema.png"
        self._draw_network(
            graph,
            output_path,
            title="Схема типов связей",
            subtitle="Узлы — типы сущностей, ребра — агрегированные предикаты и число соответствующих триплов.",
            edge_labels=edge_labels,
            node_sizes=node_sizes,
            layout="shell",
        )
        return output_path

    def save_report(self, generated: dict[str, str]) -> None:
        lines = ["# Graph Visualizations", ""]
        for name, path in generated.items():
            lines.append(f"- {name}: `{path}`")
        lines.append("")
        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> dict[str, str]:
        topic_path = self.build_topic_subgraph()
        document_path, document_title = self.build_single_document_graph()
        top_entities_path = self.build_top_entities_graph()
        schema_path = self.build_relation_schema()

        generated = {
            "topic_subgraph": str(topic_path),
            "single_document_graph": str(document_path),
            "single_document_title": document_title,
            "top_entities_graph": str(top_entities_path),
            "relation_schema": str(schema_path),
        }
        self.save_report(generated)
        manifest_path = self.plot_dir / "graph_visualizations_manifest.json"
        manifest_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
        return generated


def main() -> None:
    visualizer = GraphVisualizer()
    generated = visualizer.run()
    print(json.dumps(generated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
