"""Project settings and path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class Settings:
    """Central project settings for the RDFRAG diploma pipeline."""

    project_root: Path = Path(os.getenv("RDFRAG_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT)))
    raw_pdf_dir: Path = Path()
    parsed_dir: Path = Path()
    chunks_dir: Path = Path()
    rdf_dir: Path = Path()
    embeddings_dir: Path = Path()
    eval_dir: Path = Path()
    artifacts_dir: Path = Path()
    artifacts_metrics_dir: Path = Path()
    artifacts_metrics_csv_dir: Path = Path()
    artifacts_metrics_json_dir: Path = Path()
    artifacts_plots_dir: Path = Path()
    artifacts_plots_training_dir: Path = Path()
    artifacts_plots_retrieval_dir: Path = Path()
    artifacts_plots_answers_dir: Path = Path()
    artifacts_models_dir: Path = Path()
    artifacts_models_baseline_dir: Path = Path()
    artifacts_models_tuned_dir: Path = Path()
    artifacts_reports_dir: Path = Path()
    artifacts_reports_tables_dir: Path = Path()
    artifacts_reports_figures_dir: Path = Path()
    chunk_size: int = 1200
    chunk_overlap: int = 200
    vector_dim: int = 256
    vector_backend: str = os.getenv("RDFRAG_VECTOR_BACKEND", "faiss")
    embedding_model_name: str = os.getenv("RDFRAG_EMBEDDING_MODEL", "deepvk/USER-base")
    fuseki_dataset_url: str = os.getenv("RDFRAG_FUSEKI_URL", "http://localhost:3030/rdfrag")
    fuseki_mode: str = os.getenv("RDFRAG_FUSEKI_MODE", "required")
    fuseki_admin_user: str = os.getenv("RDFRAG_FUSEKI_ADMIN_USER", "admin")
    fuseki_admin_password: str = os.getenv("RDFRAG_FUSEKI_ADMIN_PASSWORD", "admin")
    grobid_url: str = os.getenv("RDFRAG_GROBID_URL", "http://localhost:8070")
    grobid_timeout_seconds: int = int(os.getenv("RDFRAG_GROBID_TIMEOUT", "120"))
    grobid_mode: str = os.getenv("RDFRAG_GROBID_MODE", "required")
    llm_provider: str = os.getenv("RDFRAG_LLM_PROVIDER", "ollama")
    ollama_url: str = os.getenv("RDFRAG_OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("RDFRAG_OLLAMA_MODEL", "qwen3:8b")
    ollama_timeout_seconds: int = int(os.getenv("RDFRAG_OLLAMA_TIMEOUT", "90"))
    ollama_temperature: float = float(os.getenv("RDFRAG_OLLAMA_TEMPERATURE", "0.2"))
    ollama_num_predict: int = int(os.getenv("RDFRAG_OLLAMA_NUM_PREDICT", "220"))
    ollama_unavailable_ttl_seconds: int = int(os.getenv("RDFRAG_OLLAMA_UNAVAILABLE_TTL", "600"))
    knowledge_llm_failures_before_disable: int = int(os.getenv("RDFRAG_KNOWLEDGE_LLM_FAILURES_BEFORE_DISABLE", "3"))
    knowledge_failed_cache_retry_seconds: int = int(os.getenv("RDFRAG_KNOWLEDGE_FAILED_CACHE_RETRY_SECONDS", "1800"))
    knowledge_backend: str = os.getenv("RDFRAG_KNOWLEDGE_BACKEND", "ollama_hybrid")
    knowledge_llm_max_chars: int = int(os.getenv("RDFRAG_KNOWLEDGE_LLM_MAX_CHARS", "900"))
    retrieval_top_k_eval: int = int(os.getenv("RDFRAG_RETRIEVAL_TOP_K_EVAL", "5"))

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.raw_pdf_dir = self.project_root / "data" / "raw_pdfs"
        self.parsed_dir = self.project_root / "data" / "parsed"
        self.chunks_dir = self.project_root / "data" / "chunks"
        self.rdf_dir = self.project_root / "data" / "rdf"
        self.embeddings_dir = self.project_root / "data" / "embeddings"
        self.eval_dir = self.project_root / "data" / "eval"

        self.artifacts_dir = self.project_root / "artifacts"
        self.artifacts_metrics_dir = self.artifacts_dir / "metrics"
        self.artifacts_metrics_csv_dir = self.artifacts_metrics_dir / "csv"
        self.artifacts_metrics_json_dir = self.artifacts_metrics_dir / "json"
        self.artifacts_plots_dir = self.artifacts_dir / "plots"
        self.artifacts_plots_training_dir = self.artifacts_plots_dir / "training"
        self.artifacts_plots_retrieval_dir = self.artifacts_plots_dir / "retrieval"
        self.artifacts_plots_answers_dir = self.artifacts_plots_dir / "answers"
        self.artifacts_models_dir = self.artifacts_dir / "models"
        self.artifacts_models_baseline_dir = self.artifacts_models_dir / "baseline"
        self.artifacts_models_tuned_dir = self.artifacts_models_dir / "tuned"
        self.artifacts_reports_dir = self.artifacts_dir / "reports"
        self.artifacts_reports_tables_dir = self.artifacts_reports_dir / "tables"
        self.artifacts_reports_figures_dir = self.artifacts_reports_dir / "figures"

    def ensure_directories(self) -> None:
        """Create expected project directories when they are missing."""
        for path in (
            self.raw_pdf_dir,
            self.parsed_dir,
            self.chunks_dir,
            self.rdf_dir,
            self.embeddings_dir,
            self.eval_dir,
            self.artifacts_dir,
            self.artifacts_metrics_dir,
            self.artifacts_metrics_csv_dir,
            self.artifacts_metrics_json_dir,
            self.artifacts_plots_dir,
            self.artifacts_plots_training_dir,
            self.artifacts_plots_retrieval_dir,
            self.artifacts_plots_answers_dir,
            self.artifacts_models_dir,
            self.artifacts_models_baseline_dir,
            self.artifacts_models_tuned_dir,
            self.artifacts_reports_dir,
            self.artifacts_reports_tables_dir,
            self.artifacts_reports_figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return project settings with ensured directories."""
    settings = Settings()
    settings.ensure_directories()
    return settings
