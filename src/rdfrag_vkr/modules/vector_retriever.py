"""Vector retrieval with sentence-transformers, FAISS and a safe hash fallback."""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.schemas import ChunkRecord, RetrievalHit
from rdfrag_vkr.utils.io import read_json, read_jsonl, write_json, write_jsonl


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]


class VectorRetriever:
    """Build and query a local vector index for scientific chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self.logger = logging.getLogger(__name__)

    def build_index(self, chunks: list[ChunkRecord] | None = None) -> dict:
        """Build a persisted vector index using sentence-transformers and FAISS."""
        if chunks is None:
            rows = read_jsonl(self.settings.chunks_dir / "all_chunks.jsonl")
            chunks = [ChunkRecord(**row) for row in rows]
        self.logger.info("Vector indexing stage started for %s chunks.", len(chunks))
        if not chunks:
            manifest = {"backend": "empty", "count": 0}
            write_json(self.settings.embeddings_dir / "vector_index.json", manifest)
            return manifest

        embeddings, backend = self._build_embeddings(chunks)
        manifest = {
            "backend": backend,
            "embedding_model": self.settings.embedding_model_name if backend != "hash" else None,
            "vector_dim": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
        }

        if backend == "faiss":
            index = faiss.IndexFlatIP(int(embeddings.shape[1]))
            index.add(embeddings)
            faiss.write_index(index, str(self._faiss_index_path()))
        else:
            np.save(self._numpy_embeddings_path(), embeddings)

        write_jsonl(self._metadata_path(), [chunk.model_dump() for chunk in chunks])
        write_json(self.settings.embeddings_dir / "vector_index.json", manifest)
        return manifest

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Search the persisted vector index and return top hits."""
        manifest_path = self.settings.embeddings_dir / "vector_index.json"
        if not manifest_path.exists():
            return []
        manifest = read_json(manifest_path)
        rows = read_jsonl(self._metadata_path())
        if not rows:
            return []
        chunks = [ChunkRecord(**row) for row in rows]
        try:
            query_embedding, backend = self._embed_queries([query], manifest)[0], manifest.get("backend", "hash")
        except Exception as exc:
            self.logger.warning(
                "Vector query embedding failed for backend=%s, falling back to lexical search: %s",
                manifest.get("backend", "unknown"),
                exc,
            )
            return self._lexical_search(query, chunks, top_k=top_k)

        scored: list[tuple[int, float]] = []
        if backend == "faiss" and self._faiss_index_path().exists():
            index = faiss.read_index(str(self._faiss_index_path()))
            scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
            for idx, score in zip(indices[0], scores[0], strict=False):
                if idx >= 0:
                    scored.append((int(idx), float(score)))
        else:
            matrix = np.load(self._numpy_embeddings_path())
            similarities = matrix @ query_embedding
            best_indices = np.argsort(similarities)[::-1][:top_k]
            scored = [(int(idx), float(similarities[idx])) for idx in best_indices]

        return [
            RetrievalHit(
                doc_id=chunks[idx].doc_id,
                chunk_id=chunks[idx].chunk_id,
                title=chunks[idx].title,
                score=score,
                source_file=chunks[idx].source_file,
                text=chunks[idx].text[:1000],
                source="vector",
                metadata={"chunk_index": chunks[idx].chunk_index, "backend": backend},
            )
            for idx, score in scored
        ]

    def _lexical_search(self, query: str, chunks: list[ChunkRecord], top_k: int) -> list[RetrievalHit]:
        """Fallback retrieval based on token overlap when vector inference is unavailable."""
        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, ChunkRecord]] = []
        for chunk in chunks:
            chunk_tokens = Counter(_tokenize(chunk.title + " " + chunk.text[:1500]))
            overlap = sum(min(query_tokens[token], chunk_tokens[token]) for token in query_tokens)
            if overlap == 0:
                continue
            score = float(overlap) + (0.25 if query.lower() in chunk.title.lower() else 0.0)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                score=score,
                source_file=chunk.source_file,
                text=chunk.text[:1000],
                source="vector",
                metadata={"chunk_index": chunk.chunk_index, "backend": "lexical-fallback"},
            )
            for score, chunk in scored[:top_k]
        ]

    def _build_embeddings(self, chunks: list[ChunkRecord]) -> tuple[np.ndarray, str]:
        texts = [chunk.text for chunk in chunks]
        try:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype("float32"), "faiss"
        except Exception:
            vectors = np.vstack([self._hash_embed_text(text) for text in texts]).astype("float32")
            return vectors, "hash"

    def _embed_queries(self, queries: list[str], manifest: dict) -> np.ndarray:
        if manifest.get("backend") == "faiss":
            model = self._load_model()
            return model.encode(
                queries,
                batch_size=min(16, len(queries)),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
        return np.vstack([self._hash_embed_text(query) for query in queries]).astype("float32")

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.embedding_model_name, local_files_only=True)
        return self._model

    def _hash_embed_text(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        counts = Counter(tokens)
        vector = np.zeros(self.settings.vector_dim, dtype="float32")
        for token, count in counts.items():
            position = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.settings.vector_dim
            vector[position] += float(count)
        norm = math.sqrt(float(np.dot(vector, vector))) or 1.0
        return vector / norm

    def _faiss_index_path(self) -> Path:
        return self.settings.embeddings_dir / "vector.index"

    def _numpy_embeddings_path(self) -> Path:
        return self.settings.embeddings_dir / "vector.npy"

    def _metadata_path(self) -> Path:
        return self.settings.embeddings_dir / "chunks.jsonl"
