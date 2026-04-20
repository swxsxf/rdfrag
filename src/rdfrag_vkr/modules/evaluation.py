"""Evaluation helpers and thesis artifact generation."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.modules.hybrid_retriever import HybridRetriever
from rdfrag_vkr.modules.llm_service import LLMService
from rdfrag_vkr.modules.sparql_service import SparqlService
from rdfrag_vkr.modules.vector_retriever import VectorRetriever
from rdfrag_vkr.schemas import RetrievalHit
from rdfrag_vkr.utils.artifacts import (
    write_csv,
    write_html_report,
    write_json_summary,
    write_markdown_table,
)
from rdfrag_vkr.utils.io import read_json, write_json


class Evaluator:
    """Run retrieval experiments and save diploma-ready artifacts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(self.settings)
        self.vector_retriever = VectorRetriever(self.settings)
        self.sparql_service = SparqlService(self.settings)
        self.llm_service = LLMService(self.settings)

    def evaluate_retrieval(self, gold_path: str | None = None, top_k: int = 5) -> dict:
        """Compute baseline, tuned, and hybrid metrics and persist thesis artifacts."""
        gold_file = self.settings.eval_dir / "gold_queries.json"
        if gold_path is not None:
            gold_file = self.settings.project_root / gold_path
        if not gold_file.exists():
            report = {
                "status": "skipped",
                "reason": "No gold_queries.json found. Add evaluation data to data/eval/.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(self.settings.eval_dir / "retrieval_report.json", report)
            return report

        gold_queries = read_json(gold_file)
        per_query_rows: list[dict] = []
        summary_rows: list[dict] = []
        mode_results: dict[str, list[dict]] = {}
        answer_rows: list[dict] = []
        topk_summary_rows: list[dict] = []

        for mode in ("baseline", "tuned", "hybrid"):
            results = self._evaluate_mode(gold_queries, mode=mode, top_k=top_k)
            mode_results[mode] = results
            summary_rows.append(self._summarize_mode(results, mode, top_k))
            per_query_rows.extend(results)

        for comparison_k in (1, 3, 5):
            for mode in ("baseline", "tuned", "hybrid"):
                comparison_results = self._evaluate_mode(gold_queries, mode=mode, top_k=comparison_k)
                topk_summary_rows.append(self._summarize_mode(comparison_results, mode, comparison_k))

        for mode in ("baseline", "tuned", "hybrid"):
            for row in gold_queries[: min(5, len(gold_queries))]:
                hits, graph_context = self._retrieve_by_mode(row["question"], mode, top_k=top_k)
                answer = self.llm_service._fallback_answer(row["question"], hits, graph_context)
                answer_rows.append(
                    {
                        "mode": mode,
                        "question": row["question"],
                        "answer_chars": len(answer),
                        "answer_words": len(answer.split()),
                    }
                )

        report = {
            "status": "completed",
            "queries": len(gold_queries),
            "top_k": top_k,
            "summary": summary_rows,
            "topk_summary": topk_summary_rows,
            "results": per_query_rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(self.settings.eval_dir / "retrieval_report.json", report)
        self._save_artifacts(report, summary_rows, topk_summary_rows, per_query_rows, answer_rows)
        return report

    def _evaluate_mode(self, gold_queries: list[dict], mode: str, top_k: int) -> list[dict]:
        rows: list[dict] = []
        for row in gold_queries:
            retrieved, graph_context = self._retrieve_by_mode(row["question"], mode, top_k=top_k)
            retrieved_ids = [item.doc_id for item in retrieved]
            retrieved_files = [item.source_file for item in retrieved]
            expected_ids = row.get("expected_doc_ids", [])
            expected_files = row.get("expected_source_files", [])
            rank = self._first_relevant_rank(retrieved, expected_ids, expected_files)
            hit = rank is not None
            reciprocal_rank = 1.0 / rank if rank else 0.0
            ndcg = 1.0 / math.log2(rank + 1) if rank else 0.0
            relevant_retrieved = self._count_relevant_retrieved(retrieved, expected_ids, expected_files, top_k=top_k)
            expected_total = self._expected_total(expected_ids, expected_files)
            precision = relevant_retrieved / min(top_k, len(retrieved)) if retrieved else 0.0
            recall = relevant_retrieved / expected_total if expected_total else 0.0
            rows.append(
                {
                    "mode": mode,
                    "question": row["question"],
                    "expected_doc_ids": ";".join(expected_ids),
                    "expected_source_files": ";".join(expected_files),
                    "retrieved_doc_ids": ";".join(retrieved_ids),
                    "retrieved_source_files": ";".join(retrieved_files),
                    "hit": hit,
                    "relevant_retrieved": relevant_retrieved,
                    "expected_total": expected_total,
                    "precision_at_k": round(precision, 6),
                    "recall_at_k": round(recall, 6),
                    "reciprocal_rank": round(reciprocal_rank, 6),
                    "ndcg": round(ndcg, 6),
                    "first_relevant_rank": rank or 0,
                    "graph_hits": len(graph_context),
                }
            )
        return rows

    def _retrieve_by_mode(self, question: str, mode: str, top_k: int) -> tuple[list[RetrievalHit], list[dict]]:
        if mode == "baseline":
            graph_rows = self.sparql_service.search_articles_by_keyword(question, limit=top_k)
            hits = [
                RetrievalHit(
                    doc_id=row["doc_id"],
                    chunk_id=None,
                    title=row["title"],
                    score=row["score"],
                    source_file=row["source_file"],
                    text="Matched through graph entities.",
                    source="graph",
                    metadata={"matched_entities": row.get("matched_entities", [])},
                )
                for row in graph_rows
            ]
            return hits, graph_rows
        if mode == "tuned":
            vector_hits = self.vector_retriever.search(question, top_k=top_k)
            return vector_hits, []
        return self.retriever.search(question, top_k=top_k)

    @staticmethod
    def _first_relevant_rank(
        retrieved: list[RetrievalHit],
        expected_ids: list[str],
        expected_files: list[str],
    ) -> int | None:
        for index, item in enumerate(retrieved, start=1):
            if Evaluator._is_relevant(item, expected_ids, expected_files):
                return index
        return None

    @staticmethod
    def _is_relevant(item: RetrievalHit, expected_ids: list[str], expected_files: list[str]) -> bool:
        return item.doc_id in expected_ids or item.source_file in expected_files

    @staticmethod
    def _count_relevant_retrieved(
        retrieved: list[RetrievalHit],
        expected_ids: list[str],
        expected_files: list[str],
        top_k: int,
    ) -> int:
        seen: set[str] = set()
        count = 0
        for item in retrieved[:top_k]:
            key = item.doc_id or item.source_file
            if key in seen:
                continue
            if Evaluator._is_relevant(item, expected_ids, expected_files):
                count += 1
                seen.add(key)
        return count

    @staticmethod
    def _expected_total(expected_ids: list[str], expected_files: list[str]) -> int:
        if expected_ids and expected_files:
            return max(len(set(expected_ids)), len(set(expected_files)))
        if expected_ids:
            return len(set(expected_ids))
        return len(set(expected_files))

    @staticmethod
    def _summarize_mode(rows: list[dict], mode: str, top_k: int) -> dict:
        total = len(rows)
        hit_rate = sum(1 for row in rows if row["hit"]) / total if total else 0.0
        precision = sum(row["precision_at_k"] for row in rows) / total if total else 0.0
        recall = sum(row["recall_at_k"] for row in rows) / total if total else 0.0
        mrr = sum(row["reciprocal_rank"] for row in rows) / total if total else 0.0
        ndcg = sum(row["ndcg"] for row in rows) / total if total else 0.0
        return {
            "mode": mode,
            "queries": total,
            "top_k": top_k,
            "hit_rate_at_k": round(hit_rate, 6),
            "precision_at_k": round(precision, 6),
            "recall_at_k": round(recall, 6),
            "mrr_at_k": round(mrr, 6),
            "ndcg_at_k": round(ndcg, 6),
        }

    def _save_artifacts(
        self,
        report: dict,
        summary_rows: list[dict],
        topk_summary_rows: list[dict],
        per_query_rows: list[dict],
        answer_rows: list[dict],
    ) -> None:
        knowledge_rows, knowledge_backend_rows, knowledge_summary = self._collect_knowledge_coverage()
        write_csv(self.settings.artifacts_metrics_csv_dir / "retrieval_query_results.csv", per_query_rows)
        write_csv(self.settings.artifacts_metrics_csv_dir / "answer_length_results.csv", answer_rows)
        write_csv(self.settings.artifacts_metrics_csv_dir / "topk_summary.csv", topk_summary_rows)
        write_csv(self.settings.artifacts_metrics_csv_dir / "knowledge_coverage.csv", knowledge_rows)
        write_csv(self.settings.artifacts_metrics_csv_dir / "knowledge_backend_summary.csv", knowledge_backend_rows)
        write_json_summary(self.settings.artifacts_metrics_json_dir / "retrieval_summary.json", report)
        write_json_summary(self.settings.artifacts_metrics_json_dir / "knowledge_coverage_summary.json", knowledge_summary)
        write_markdown_table(self.settings.artifacts_reports_tables_dir / "retrieval_summary.md", summary_rows)
        write_csv(self.settings.artifacts_reports_tables_dir / "retrieval_summary.csv", summary_rows)
        write_markdown_table(self.settings.artifacts_reports_tables_dir / "topk_summary.md", topk_summary_rows)
        write_markdown_table(self.settings.artifacts_reports_tables_dir / "retrieval_per_query.md", per_query_rows)
        write_markdown_table(self.settings.artifacts_reports_tables_dir / "answer_summary.md", answer_rows)
        write_markdown_table(self.settings.artifacts_reports_tables_dir / "knowledge_backend_summary.md", knowledge_backend_rows)
        figure_index = (
            "- retrieval_quality_comparison.png\n"
            "- baseline_vs_tuned_vs_hybrid.png\n"
            "- precision_recall_comparison.png\n"
            "- per_query_hit_heatmap.png\n"
            "- first_relevant_rank_comparison.png\n"
            "- topk_hit_rate_comparison.png\n"
            "- topk_recall_comparison.png\n"
            "- answer_length_comparison.png\n"
            "- knowledge_backend_distribution.png\n"
            "- knowledge_entities_relations_distribution.png\n"
            "- knowledge_entities_relations_scatter.png\n"
        )
        (self.settings.artifacts_reports_figures_dir / "figure_index.md").write_text(figure_index, encoding="utf-8")

        sections = [
            ("Retrieval Summary", "<pre>" + str(summary_rows) + "</pre>"),
            ("Top-K Summary", "<pre>" + str(topk_summary_rows) + "</pre>"),
            ("Per-query Results", "<pre>" + str(per_query_rows[:10]) + "</pre>"),
            ("Knowledge Coverage", "<pre>" + str(knowledge_summary) + "</pre>"),
        ]
        write_html_report(
            self.settings.artifacts_reports_dir / "retrieval_experiment_summary.html",
            "Retrieval Experiment Summary",
            sections,
        )

        self._save_plots(summary_rows, topk_summary_rows, per_query_rows, answer_rows, knowledge_rows, knowledge_backend_rows)

    def _save_plots(
        self,
        summary_rows: list[dict],
        topk_summary_rows: list[dict],
        per_query_rows: list[dict],
        answer_rows: list[dict],
        knowledge_rows: list[dict],
        knowledge_backend_rows: list[dict],
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except Exception:
            return

        summary_frame = pd.DataFrame(summary_rows)
        if not summary_frame.empty:
            figure = plt.figure(figsize=(10, 5))
            positions = range(len(summary_frame))
            width = 0.25
            plt.bar([pos - width for pos in positions], summary_frame["hit_rate_at_k"], width=width, label="HitRate")
            plt.bar(positions, summary_frame["mrr_at_k"], width=width, label="MRR")
            plt.bar([pos + width for pos in positions], summary_frame["ndcg_at_k"], width=width, label="nDCG")
            plt.xticks(list(positions), summary_frame["mode"])
            plt.ylim(0, 1.05)
            plt.title("Retrieval Quality Comparison")
            plt.legend()
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_retrieval_dir / "retrieval_quality_comparison.png", dpi=200)
            plt.close(figure)

            figure = plt.figure(figsize=(8, 5))
            plt.bar(summary_frame["mode"], summary_frame["hit_rate_at_k"], color=["#9ca3af", "#2563eb", "#059669"])
            plt.ylim(0, 1.05)
            plt.title("Baseline vs Tuned vs Hybrid")
            plt.ylabel("HitRate@K")
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_retrieval_dir / "baseline_vs_tuned_vs_hybrid.png", dpi=200)
            plt.close(figure)

            figure = plt.figure(figsize=(8, 5))
            positions = range(len(summary_frame))
            width = 0.35
            plt.bar([pos - width / 2 for pos in positions], summary_frame["precision_at_k"], width=width, label="Precision@K")
            plt.bar([pos + width / 2 for pos in positions], summary_frame["recall_at_k"], width=width, label="Recall@K")
            plt.xticks(list(positions), summary_frame["mode"])
            plt.ylim(0, 1.05)
            plt.title("Precision@K vs Recall@K")
            plt.legend()
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_retrieval_dir / "precision_recall_comparison.png", dpi=200)
            plt.close(figure)

        topk_frame = pd.DataFrame(topk_summary_rows)
        if not topk_frame.empty:
            figure = plt.figure(figsize=(9, 5))
            for mode, color in (("baseline", "#9ca3af"), ("tuned", "#2563eb"), ("hybrid", "#059669")):
                mode_frame = topk_frame[topk_frame["mode"] == mode].sort_values("top_k")
                if not mode_frame.empty:
                    plt.plot(mode_frame["top_k"], mode_frame["hit_rate_at_k"], marker="o", label=mode, color=color)
            plt.xticks([1, 3, 5])
            plt.ylim(0, 1.05)
            plt.title("Top-1 / Top-3 / Top-5 HitRate Comparison")
            plt.xlabel("K")
            plt.ylabel("HitRate@K")
            plt.legend()
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_retrieval_dir / "topk_hit_rate_comparison.png", dpi=200)
            plt.close(figure)

            figure = plt.figure(figsize=(9, 5))
            for mode, color in (("baseline", "#9ca3af"), ("tuned", "#2563eb"), ("hybrid", "#059669")):
                mode_frame = topk_frame[topk_frame["mode"] == mode].sort_values("top_k")
                if not mode_frame.empty:
                    plt.plot(mode_frame["top_k"], mode_frame["recall_at_k"], marker="o", label=mode, color=color)
            plt.xticks([1, 3, 5])
            plt.ylim(0, 1.05)
            plt.title("Top-1 / Top-3 / Top-5 Recall Comparison")
            plt.xlabel("K")
            plt.ylabel("Recall@K")
            plt.legend()
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_retrieval_dir / "topk_recall_comparison.png", dpi=200)
            plt.close(figure)

        per_query_frame = pd.DataFrame(per_query_rows)
        if not per_query_frame.empty:
            hit_pivot = per_query_frame.pivot(index="question", columns="mode", values="hit").fillna(0).astype(float)
            if not hit_pivot.empty:
                figure = plt.figure(figsize=(10, max(5, len(hit_pivot) * 0.6)))
                plt.imshow(hit_pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
                plt.xticks(range(len(hit_pivot.columns)), hit_pivot.columns)
                plt.yticks(range(len(hit_pivot.index)), hit_pivot.index)
                plt.title("Per-query Retrieval Hit Heatmap")
                plt.colorbar(label="Hit")
                plt.tight_layout()
                figure.savefig(self.settings.artifacts_plots_retrieval_dir / "per_query_hit_heatmap.png", dpi=200)
                plt.close(figure)

            rank_frame = per_query_frame.copy()
            rank_frame["first_relevant_rank"] = rank_frame["first_relevant_rank"].replace(0, float("nan"))
            grouped_rank = rank_frame.groupby("mode", as_index=False)["first_relevant_rank"].mean()
            if not grouped_rank.empty:
                figure = plt.figure(figsize=(8, 5))
                plt.bar(grouped_rank["mode"], grouped_rank["first_relevant_rank"], color=["#64748b", "#0ea5e9", "#22c55e"])
                plt.title("Average First Relevant Rank")
                plt.ylabel("Rank")
                plt.tight_layout()
                figure.savefig(self.settings.artifacts_plots_retrieval_dir / "first_relevant_rank_comparison.png", dpi=200)
                plt.close(figure)

        answer_frame = pd.DataFrame(answer_rows)
        if not answer_frame.empty:
            grouped = answer_frame.groupby("mode", as_index=False)["answer_words"].mean()
            figure = plt.figure(figsize=(8, 5))
            plt.bar(grouped["mode"], grouped["answer_words"], color=["#64748b", "#0ea5e9", "#22c55e"])
            plt.title("Average Answer Length by Retrieval Mode")
            plt.ylabel("Words")
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_answers_dir / "answer_length_comparison.png", dpi=200)
            plt.close(figure)

        knowledge_frame = pd.DataFrame(knowledge_rows)
        backend_frame = pd.DataFrame(knowledge_backend_rows)
        if not backend_frame.empty:
            figure = plt.figure(figsize=(8, 5))
            plt.bar(backend_frame["backend"], backend_frame["documents"], color=["#0ea5e9", "#f59e0b", "#64748b", "#ef4444"][: len(backend_frame)])
            plt.title("Knowledge Extraction Backend Distribution")
            plt.ylabel("Documents")
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_training_dir / "knowledge_backend_distribution.png", dpi=200)
            plt.close(figure)

        if not knowledge_frame.empty:
            figure = plt.figure(figsize=(8, 5))
            plt.hist(knowledge_frame["entity_count"], bins=15, alpha=0.7, label="Entities")
            plt.hist(knowledge_frame["relation_count"], bins=15, alpha=0.7, label="Relations")
            plt.title("Knowledge Extraction Coverage Distribution")
            plt.xlabel("Count per document")
            plt.ylabel("Documents")
            plt.legend()
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_training_dir / "knowledge_entities_relations_distribution.png", dpi=200)
            plt.close(figure)

            figure = plt.figure(figsize=(8, 5))
            plt.scatter(knowledge_frame["entity_count"], knowledge_frame["relation_count"], alpha=0.7, color="#2563eb")
            plt.title("Entities vs Relations per Document")
            plt.xlabel("Entity count")
            plt.ylabel("Relation count")
            plt.tight_layout()
            figure.savefig(self.settings.artifacts_plots_training_dir / "knowledge_entities_relations_scatter.png", dpi=200)
            plt.close(figure)

    def _collect_knowledge_coverage(self) -> tuple[list[dict], list[dict], dict]:
        cache_dir = self.settings.artifacts_models_tuned_dir / "knowledge_cache"
        rows: list[dict] = []
        backend_counts: dict[str, int] = {}

        if not cache_dir.exists():
            return [], [], {"status": "missing", "cache_dir": str(cache_dir)}

        for path in sorted(cache_dir.glob("*.json")):
            payload = self._safe_read_json(path)
            if not payload:
                continue
            backend = str(payload.get("backend", "unknown"))
            entity_count = len(payload.get("entities", []))
            relation_count = len(payload.get("relations", []))
            rows.append(
                {
                    "doc_cache_file": path.name,
                    "backend": backend,
                    "entity_count": entity_count,
                    "relation_count": relation_count,
                }
            )
            backend_counts[backend] = backend_counts.get(backend, 0) + 1

        backend_rows = [{"backend": key, "documents": value} for key, value in sorted(backend_counts.items())]
        entity_mean = sum(row["entity_count"] for row in rows) / len(rows) if rows else 0.0
        relation_mean = sum(row["relation_count"] for row in rows) / len(rows) if rows else 0.0
        summary = {
            "status": "completed",
            "documents": len(rows),
            "average_entities_per_doc": round(entity_mean, 4),
            "average_relations_per_doc": round(relation_mean, 4),
            "backend_distribution": backend_rows,
        }
        return rows, backend_rows, summary

    @staticmethod
    def _safe_read_json(path: Path) -> dict | None:
        try:
            return read_json(path)
        except Exception:
            return None
