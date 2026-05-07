"""Generate thesis-ready project visualizations from existing artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdflib import Graph, Literal, RDF, URIRef

from rdfrag_vkr.config import get_settings


BASE_URI = "http://example.org/rdfrag/"
COLOR_BG = "#ffffff"
COLOR_PANEL = "#ffffff"
COLOR_PANEL_ALT = "#f8fafc"
COLOR_TEXT = "#0f172a"
COLOR_SUBTEXT = "#475569"
COLOR_GRID = "#cbd5e1"
COLOR_BLUE = "#2563eb"
COLOR_CYAN = "#06b6d4"
COLOR_GREEN = "#10b981"
COLOR_AMBER = "#f59e0b"
COLOR_RED = "#ef4444"
COLOR_PURPLE = "#a855f7"
COLOR_SLATE = "#64748b"

MODE_LABELS = {
    "baseline": "Graph retrieval",
    "tuned": "Vector retrieval",
    "hybrid": "Hybrid retrieval",
}
TOPIC_ALIASES = {
    "Blockchain": ["blockchain", "блокчейн"],
    "Digital twins": ["digital twin", "digital twins", "цифров"],
    "Metaverse": ["metaverse", "метавселен"],
    "6G": ["6g"],
    "IoT": ["iot", "internet of things", "интернет вещей"],
    "Low-code": ["low-code", "low code", "no-code", "nocode", "малым кодом", "низким кодом"],
}
TOPIC_COLORS = {
    "Blockchain": COLOR_BLUE,
    "Digital twins": COLOR_CYAN,
    "Metaverse": COLOR_PURPLE,
    "6G": COLOR_RED,
    "IoT": COLOR_GREEN,
    "Low-code": COLOR_AMBER,
}
QUERY_SCORE_OVERRIDES = {
    "Как блокчейн используется в цифровых двойниках?": 4.0,
    "Что говорится о low-code platforms?": 4.0,
    "Какие темы связаны с 6G и smart city?": 3.5,
    "Что говорится о цифровых двойниках как основе enterprise metaverse?": 3.5,
    "Как IoT связан с бизнес-моделями и интернетом вещей?": 3.0,
}


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.facecolor"] = COLOR_PANEL
    plt.rcParams["figure.facecolor"] = COLOR_BG
    plt.rcParams["savefig.facecolor"] = COLOR_BG
    plt.rcParams["axes.edgecolor"] = COLOR_GRID
    plt.rcParams["axes.labelcolor"] = COLOR_TEXT
    plt.rcParams["xtick.color"] = COLOR_TEXT
    plt.rcParams["ytick.color"] = COLOR_TEXT
    plt.rcParams["text.color"] = COLOR_TEXT


def local_name(uri: URIRef | str) -> str:
    value = str(uri)
    if value.startswith(BASE_URI):
        return value.removeprefix(BASE_URI)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rsplit("/", 1)[-1]
    return value


def clean_text(value: str, max_len: int = 66) -> str:
    text = " ".join(str(value).replace("\n", " ").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def draw_title(figure: plt.Figure, title: str, subtitle: str, *, title_size: int = 22, subtitle_size: int = 11) -> None:
    figure.text(0.04, 0.965, title, fontsize=title_size, fontweight="bold", color=COLOR_TEXT, ha="left", va="top")
    figure.text(0.04, 0.905, subtitle, fontsize=subtitle_size, color=COLOR_SUBTEXT, ha="left", va="top")


class ProjectVisualizer:
    """Create additional diploma visualizations without rerunning ingestion."""

    def __init__(self) -> None:
        self.settings = get_settings()
        configure_matplotlib()
        self.plot_dir = self.settings.artifacts_plots_dir / "project"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.settings.artifacts_reports_figures_dir / "project_visualizations.md"
        self.manifest_path = self.plot_dir / "project_visualizations_manifest.json"
        self.rdf_graph = Graph()
        self.rdf_graph.parse(self.settings.rdf_dir / "knowledge_graph.ttl", format="turtle")
        self.pipeline_summary = json.loads((self.settings.eval_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
        self.parsed_manifest = json.loads((self.settings.parsed_dir / "manifest.json").read_text(encoding="utf-8"))
        self.rdf_manifest = json.loads((self.settings.rdf_dir / "manifest.json").read_text(encoding="utf-8"))
        self.retrieval_frame = pd.read_csv(
            self.settings.artifacts_metrics_csv_dir / "retrieval_query_results.csv",
            encoding="utf-8-sig",
        )
        self.topk_frame = pd.read_csv(
            self.settings.artifacts_metrics_csv_dir / "topk_summary.csv",
            encoding="utf-8-sig",
        )
        self.title_predicate = URIRef(f"{BASE_URI}title")
        self.label_predicate = URIRef(f"{BASE_URI}label")
        self.topic_predicate = URIRef(f"{BASE_URI}hasTopic")

    def run(self) -> dict[str, str]:
        manifest = {
            "architecture": str(self.build_architecture_diagram()),
            "corpus_composition": str(self.build_corpus_composition()),
            "parser_backends": str(self.build_parser_backend_chart()),
            "topic_focus": str(self.build_topic_focus_chart()),
            "retrieval_overlap": str(self.build_retrieval_overlap_chart()),
            "demo_queries": str(self.build_demo_queries_figure()),
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.report_path.write_text(self._build_report(manifest), encoding="utf-8")
        return manifest

    def build_architecture_diagram(self) -> Path:
        output = self.plot_dir / "architecture_overview.png"
        figure = plt.figure(figsize=(18, 9.8), facecolor=COLOR_BG)
        axis = plt.gca()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

        def draw_link(start: tuple[float, float], end: tuple[float, float], *, arrow: bool = True) -> None:
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>" if arrow else "-",
                    mutation_scale=16 if arrow else 1,
                    linewidth=2.0,
                    color=COLOR_SLATE,
                    connectionstyle="arc3,rad=0.0",
                )
            )

        draw_title(
            figure,
            "Архитектура graph-enhanced RAG-системы",
            "Полный путь от корпуса PDF до ответа в API и Gradio-интерфейсе.",
            title_size=30,
            subtitle_size=16,
        )

        boxes = {
            "pdf": (0.035, 0.66, 0.17, 0.145, "Корпус PDF\n151 документов", COLOR_PANEL_ALT),
            "parse": (0.245, 0.66, 0.18, 0.145, "Парсинг\nGROBID / pypdf", COLOR_PANEL),
            "chunks": (0.465, 0.66, 0.18, 0.145, "Chunking\n5 135 фрагментов", COLOR_PANEL_ALT),
            "extract": (0.685, 0.66, 0.19, 0.145, "Knowledge extraction\nqwen3:8b", COLOR_PANEL),
            "vector": (0.595, 0.365, 0.20, 0.145, "FAISS index\nvector retrieval", COLOR_PANEL_ALT),
            "graph": (0.815, 0.365, 0.17, 0.145, "RDF + Fuseki\ngraph retrieval", COLOR_PANEL_ALT),
            "hybrid": (0.67, 0.105, 0.21, 0.145, "Hybrid retrieval\nfusion + reranking", COLOR_PANEL),
            "answer": (0.405, 0.105, 0.21, 0.145, "Answer generation\nLLM synthesis", COLOR_PANEL_ALT),
            "serve": (0.125, 0.105, 0.22, 0.145, "FastAPI + Gradio\nчат / query endpoint", COLOR_PANEL),
        }

        for _, (x, y, w, h, label, face) in boxes.items():
            axis.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.012,rounding_size=0.025",
                    linewidth=1.4,
                    edgecolor=COLOR_GRID,
                    facecolor=face,
                )
            )
            axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=15, color=COLOR_TEXT)

        top_arrows = [
            ("pdf", "parse"),
            ("parse", "chunks"),
            ("chunks", "extract"),
        ]
        for src, dst in top_arrows:
            sx, sy, sw, sh, *_ = boxes[src]
            dx, dy, dw, dh, *_ = boxes[dst]
            draw_link((sx + sw, sy + sh / 2), (dx, dy + dh / 2))

        extract_x, extract_y, extract_w, extract_h, *_ = boxes["extract"]
        vector_x, vector_y, vector_w, vector_h, *_ = boxes["vector"]
        graph_x, graph_y, graph_w, graph_h, *_ = boxes["graph"]
        hybrid_x, hybrid_y, hybrid_w, hybrid_h, *_ = boxes["hybrid"]
        answer_x, answer_y, answer_w, answer_h, *_ = boxes["answer"]
        serve_x, serve_y, serve_w, serve_h, *_ = boxes["serve"]

        branch_origin = (extract_x + extract_w / 2, extract_y)
        branch_split = (extract_x + extract_w / 2, 0.575)
        left_anchor = (vector_x + vector_w / 2, 0.54)
        right_anchor = (graph_x + graph_w / 2, 0.54)
        vector_target = (vector_x + vector_w / 2, vector_y + vector_h)
        graph_target = (graph_x + graph_w / 2, graph_y + graph_h)

        draw_link(branch_origin, branch_split, arrow=False)
        draw_link(branch_split, left_anchor, arrow=False)
        draw_link(branch_split, right_anchor, arrow=False)
        draw_link(left_anchor, vector_target)
        draw_link(right_anchor, graph_target)

        axis.text(
            branch_split[0],
            0.512,
            "Из extraction формируются\nдве базы: векторная и графовая",
            fontsize=12.5,
            color=COLOR_SUBTEXT,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": COLOR_BG, "edgecolor": "none"},
        )

        vector_start = (vector_x + vector_w / 2, vector_y)
        graph_start = (graph_x + graph_w / 2, graph_y)
        hybrid_top = (hybrid_x + hybrid_w / 2, hybrid_y + hybrid_h)
        elbow_y = hybrid_y + hybrid_h + 0.045
        draw_link(vector_start, (vector_start[0], elbow_y), arrow=False)
        draw_link((vector_start[0], elbow_y), (hybrid_top[0], elbow_y), arrow=False)
        draw_link((hybrid_top[0], elbow_y), hybrid_top)

        draw_link(graph_start, (graph_start[0], elbow_y), arrow=False)
        draw_link((graph_start[0], elbow_y), (hybrid_top[0], elbow_y), arrow=False)
        draw_link((hybrid_top[0], elbow_y), hybrid_top)

        answer_left = (answer_x, answer_y + answer_h / 2)
        answer_right = (answer_x + answer_w, answer_y + answer_h / 2)
        hybrid_left = (hybrid_x, hybrid_y + hybrid_h / 2)
        serve_right = (serve_x + serve_w, serve_y + serve_h / 2)

        draw_link(hybrid_left, answer_right)
        draw_link(answer_left, serve_right)

        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return output

    def build_corpus_composition(self) -> Path:
        output = self.plot_dir / "corpus_composition.png"
        uri_nodes, uri_edges = self._graph_stats()
        labels = [
            "PDF",
            "Parsed docs",
            "Chunks",
            "Knowledge docs",
            "RDF triples",
            "Graph nodes",
            "Graph edges",
        ]
        values = [
            int(self.pipeline_summary["pdf_count"]),
            int(self.pipeline_summary["parsed_count"]),
            int(self.pipeline_summary["chunk_count"]),
            int(self.pipeline_summary["knowledge_document_count"]),
            int(self.rdf_manifest["triple_count"]),
            int(uri_nodes),
            int(uri_edges),
        ]
        colors = [COLOR_BLUE, COLOR_CYAN, COLOR_GREEN, COLOR_AMBER, COLOR_PURPLE, COLOR_RED, COLOR_SLATE]

        figure, axis = plt.subplots(figsize=(14, 7), facecolor=COLOR_BG)
        draw_title(
            figure,
            "Состав корпуса и артефактов пайплайна",
            "Показаны объёмы данных, полученных без дополнительного rerun ingestion.",
        )
        axis.bar(labels, values, color=colors, width=0.65)
        axis.grid(axis="y", color=COLOR_GRID, alpha=0.5, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Количество")
        for idx, value in enumerate(values):
            axis.text(idx, value + max(values) * 0.015, f"{value:,}".replace(",", " "), ha="center", va="bottom", fontsize=10)
        for spine in axis.spines.values():
            spine.set_color(COLOR_GRID)
        figure.subplots_adjust(top=0.83, left=0.07, right=0.98, bottom=0.18)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return output

    def build_parser_backend_chart(self) -> Path:
        output = self.plot_dir / "parser_backend_comparison.png"
        counts = self.parsed_manifest["parser_counts"]
        labels = ["GROBID", "pypdf fallback"]
        values = [int(counts.get("grobid", 0)), int(counts.get("pypdf", 0))]
        colors = [COLOR_GREEN, COLOR_AMBER]

        figure, axis = plt.subplots(figsize=(10, 6), facecolor=COLOR_BG)
        draw_title(
            figure,
            "GROBID vs pypdf",
            "Распределение фактически использованных парсеров в ingestion-пайплайне.",
        )
        wedges, _texts, autotexts = axis.pie(
            values,
            labels=labels,
            colors=colors,
            startangle=110,
            autopct=lambda pct: f"{pct:.1f}%",
            pctdistance=0.78,
            textprops={"color": COLOR_TEXT, "fontsize": 11},
            wedgeprops={"linewidth": 1.2, "edgecolor": COLOR_BG},
        )
        axis.legend(
            wedges,
            [f"{label}: {value}" for label, value in zip(labels, values)],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
            labelcolor=COLOR_TEXT,
        )
        for autotext in autotexts:
            autotext.set_color(COLOR_BG)
            autotext.set_fontweight("bold")
        figure.subplots_adjust(top=0.82, bottom=0.12)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return output

    def build_topic_focus_chart(self) -> Path:
        output = self.plot_dir / "corpus_topic_focus.png"
        topic_counts = self._topic_focus_counts()
        labels = list(TOPIC_ALIASES.keys())
        values = [topic_counts.get(label, 0) for label in labels]
        colors = [TOPIC_COLORS[label] for label in labels]

        figure, axis = plt.subplots(figsize=(12, 7), facecolor=COLOR_BG)
        draw_title(
            figure,
            "Тематики корпуса научных публикаций",
            "Число уникальных статей, связанных с ключевыми направлениями цифровой экономики.",
        )
        axis.barh(labels, values, color=colors, height=0.62)
        axis.invert_yaxis()
        axis.grid(axis="x", color=COLOR_GRID, alpha=0.5, linewidth=0.8)
        axis.set_xlabel("Количество уникальных статей")
        for index, value in enumerate(values):
            axis.text(value + max(values) * 0.015, index, str(value), va="center", fontsize=10)
        for spine in axis.spines.values():
            spine.set_color(COLOR_GRID)
        figure.subplots_adjust(top=0.83, left=0.18, right=0.96, bottom=0.12)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return output

    def build_retrieval_overlap_chart(self) -> Path:
        output = self.plot_dir / "retrieval_mode_overlap.png"
        frame = self.retrieval_frame.copy()
        frame["mode_label"] = frame["mode"].map(MODE_LABELS)

        hit_counts = frame.groupby("mode")["hit"].sum().reindex(["baseline", "tuned", "hybrid"]).fillna(0)
        best_per_question = self._best_mode_per_question(frame)
        win_counts = best_per_question["mode"].value_counts().reindex(["baseline", "tuned", "hybrid"]).fillna(0)

        figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), facecolor=COLOR_BG)
        draw_title(
            figure,
            "Overlap graph vs vector vs hybrid retrieval",
            "Сравнение по hit-count и числу побед на запросах для разных режимов retrieval.",
        )
        figure.subplots_adjust(top=0.78, left=0.06, right=0.97, bottom=0.14, wspace=0.24)

        mode_order = ["baseline", "tuned", "hybrid"]
        mode_colors = [COLOR_SLATE, COLOR_BLUE, COLOR_GREEN]
        mode_labels = [MODE_LABELS[mode] for mode in mode_order]

        axes[0].bar(mode_labels, hit_counts.values, color=mode_colors, width=0.62)
        axes[0].set_title("Количество hit по 10 запросам", fontsize=12, pad=12)
        axes[0].grid(axis="y", color=COLOR_GRID, alpha=0.5)
        axes[0].tick_params(axis="x", rotation=18)
        for idx, value in enumerate(hit_counts.values):
            axes[0].text(idx, value + 0.15, str(int(value)), ha="center", va="bottom")

        axes[1].bar(mode_labels, win_counts.values, color=mode_colors, width=0.62)
        axes[1].set_title("Победы по nDCG / MRR", fontsize=12, pad=12)
        axes[1].grid(axis="y", color=COLOR_GRID, alpha=0.5)
        axes[1].tick_params(axis="x", rotation=18)
        for idx, value in enumerate(win_counts.values):
            axes[1].text(idx, value + 0.15, str(int(value)), ha="center", va="bottom")

        for axis in axes:
            for spine in axis.spines.values():
                spine.set_color(COLOR_GRID)
            axis.set_axisbelow(True)

        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        return output

    def build_demo_queries_figure(self) -> Path:
        output = self.plot_dir / "demo_queries_table.png"
        demo_rows = self._demo_queries_rows()

        figure, axis = plt.subplots(figsize=(18, 7.8), facecolor=COLOR_BG)
        axis.axis("off")
        draw_title(
            figure,
            "Демонстрационные запросы для защиты",
            "Вопросы с наглядным retrieval-эффектом и понятной интерпретацией результата.",
        )
        columns = ["Вопрос", "Лучший режим", "Top source", "Краткая оценка"]
        cell_text = [
            [
                clean_text(row["question"], 62),
                row["best_mode"],
                clean_text(row["top_source"], 56),
                row["assessment"],
            ]
            for row in demo_rows
        ]
        table = axis.table(
            cellText=cell_text,
            colLabels=columns,
            cellLoc="left",
            colLoc="left",
            colColours=[COLOR_PANEL_ALT] * len(columns),
            loc="center",
            colWidths=[0.33, 0.13, 0.33, 0.21],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor(COLOR_GRID)
            cell.set_linewidth(0.8)
            if row_index == 0:
                cell.set_text_props(color=COLOR_TEXT, weight="bold")
                cell.set_facecolor(COLOR_PANEL_ALT)
            else:
                cell.set_facecolor(COLOR_PANEL if row_index % 2 else COLOR_PANEL_ALT)
                cell.get_text().set_color(COLOR_TEXT)
        figure.subplots_adjust(top=0.83, left=0.02, right=0.98, bottom=0.06)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)

        markdown_path = self.settings.artifacts_reports_tables_dir / "demo_queries.md"
        markdown_path.write_text(self._demo_queries_markdown(demo_rows), encoding="utf-8")
        return output

    def _graph_stats(self) -> tuple[int, int]:
        subjects = {node for node in self.rdf_graph.subjects() if isinstance(node, URIRef)}
        objects = {node for node in self.rdf_graph.objects() if isinstance(node, URIRef)}
        uri_nodes = subjects | objects
        uri_edges = sum(
            1
            for subj, predicate, obj in self.rdf_graph
            if isinstance(subj, URIRef) and isinstance(obj, URIRef) and predicate != RDF.type
        )
        return len(uri_nodes), uri_edges

    def _topic_focus_counts(self) -> dict[str, int]:
        topic_labels = {subject: str(obj) for subject, _, obj in self.rdf_graph.triples((None, self.label_predicate, None))}
        article_titles = {subject: str(obj) for subject, _, obj in self.rdf_graph.triples((None, self.title_predicate, None))}

        article_to_topics: dict[str, set[str]] = defaultdict(set)
        for article, _, topic in self.rdf_graph.triples((None, self.topic_predicate, None)):
            title = article_titles.get(article)
            if not title:
                continue
            article_to_topics[title].add(topic_labels.get(topic, ""))

        counts = Counter()
        for title, topics in article_to_topics.items():
            aggregated_text = " ".join([title, *topics]).lower()
            for category, aliases in TOPIC_ALIASES.items():
                if any(alias in aggregated_text for alias in aliases):
                    counts[category] += 1
        return {category: counts.get(category, 0) for category in TOPIC_ALIASES}

    def _best_mode_per_question(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for question, group in frame.groupby("question", sort=False):
            ranked = group.sort_values(
                by=["ndcg", "reciprocal_rank", "recall_at_k", "precision_at_k", "hit"],
                ascending=[False, False, False, False, False],
            )
            rows.append(ranked.iloc[0])
        return pd.DataFrame(rows)

    def _demo_queries_rows(self) -> list[dict[str, str]]:
        best_rows = self._best_mode_per_question(self.retrieval_frame)
        best_rows = best_rows[best_rows["hit"] == True].copy()
        best_rows["priority"] = best_rows["question"].map(lambda question: QUERY_SCORE_OVERRIDES.get(question, 0.0))
        best_rows = best_rows.sort_values(
            by=["priority", "ndcg", "recall_at_k", "precision_at_k"],
            ascending=[False, False, False, False],
        )

        rows = []
        for _, row in best_rows.head(5).iterrows():
            top_source = str(row["retrieved_source_files"]).split(";")[0]
            rows.append(
                {
                    "question": str(row["question"]),
                    "best_mode": MODE_LABELS[str(row["mode"])],
                    "top_source": top_source,
                    "assessment": self._assessment_text(float(row["ndcg"]), float(row["recall_at_k"])),
                }
            )
        return rows

    @staticmethod
    def _assessment_text(ndcg: float, recall: float) -> str:
        if ndcg >= 0.95 and recall >= 0.95:
            return "Высокая релевантность и полное покрытие"
        if ndcg >= 0.75:
            return "Сильный кейс для демонстрации retrieval"
        if recall >= 0.66:
            return "Хорошее покрытие по источникам"
        return "Есть релевантные документы, но ответ стоит комментировать"

    def _build_report(self, manifest: dict[str, str]) -> str:
        labels = {
            "architecture": "Архитектурная схема системы",
            "corpus_composition": "Диаграмма состава корпуса",
            "parser_backends": "GROBID vs pypdf",
            "topic_focus": "График тематик корпуса",
            "retrieval_overlap": "Overlap graph vs vector vs hybrid",
            "demo_queries": "Figure с demo queries",
        }
        lines = ["# Дополнительные визуализации проекта", ""]
        for key, path in manifest.items():
            lines.append(f"- {labels[key]}: `{path}`")
        lines.append("")
        lines.append("Все артефакты построены из уже существующих JSON/CSV/RDF-файлов без нового запуска ingestion.")
        return "\n".join(lines)

    @staticmethod
    def _demo_queries_markdown(rows: list[dict[str, str]]) -> str:
        headers = ["Вопрос", "Лучший режим", "Top source", "Краткая оценка"]
        separator = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows:
            values = [
                str(row["question"]).replace("|", "/"),
                str(row["best_mode"]).replace("|", "/"),
                str(row["top_source"]).replace("|", "/"),
                str(row["assessment"]).replace("|", "/"),
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines) + "\n"


def main() -> None:
    visualizer = ProjectVisualizer()
    manifest = visualizer.run()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
