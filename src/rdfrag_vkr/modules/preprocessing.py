"""Text cleaning and chunking for parsed scientific articles."""

from __future__ import annotations

import logging
import re
import time

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.schemas import ChunkRecord, ParsedDocument
from rdfrag_vkr.utils.io import write_json, write_jsonl


class Preprocessor:
    """Clean parsed text and split it into retrieval-friendly chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)

    def process_corpus(self, documents: list[ParsedDocument]) -> list[ChunkRecord]:
        """Preprocess parsed documents and persist chunk files."""
        all_chunks: list[ChunkRecord] = []
        total = len(documents)
        started_at = time.monotonic()
        self.logger.info("Preprocessing stage started for %s parsed documents.", total)
        for index, document in enumerate(documents, start=1):
            cleaned_text = self.clean_text(document.text)
            chunks = self.chunk_text(
                text=cleaned_text,
                doc_id=document.metadata.doc_id,
                title=document.metadata.title,
                source_file=document.metadata.source_file,
            )
            if chunks:
                self.save_chunks(document.metadata.doc_id, chunks)
                all_chunks.extend(chunks)
            if index == 1 or index % 10 == 0 or index == total:
                self.logger.info(
                    "[PREPROCESS %s/%s | remaining=%s] %s | new_chunks=%s | total_chunks=%s | total_elapsed=%.1fs",
                    index,
                    total,
                    total - index,
                    document.metadata.source_file,
                    len(chunks),
                    len(all_chunks),
                    time.monotonic() - started_at,
                )
        write_jsonl(self.settings.chunks_dir / "all_chunks.jsonl", [chunk.model_dump() for chunk in all_chunks])
        write_json(
            self.settings.chunks_dir / "manifest.json",
            {
                "chunk_count": len(all_chunks),
                "document_count": len(documents),
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            },
        )
        return all_chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize noisy PDF text with lightweight heuristics."""
        cleaned = text.replace("\u00ad", "")
        cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)
        cleaned = re.sub(r"\r", "\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        return cleaned.strip()

    def chunk_text(self, text: str, doc_id: str, title: str, source_file: str) -> list[ChunkRecord]:
        """Split cleaned text into overlapping chunks."""
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        chunks: list[ChunkRecord] = []
        current = ""
        chunk_index = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tentative = f"{current} {sentence}".strip() if current else sentence
            if current and len(tentative) > self.settings.chunk_size:
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc_id}-chunk-{chunk_index}",
                        doc_id=doc_id,
                        source_file=source_file,
                        title=title,
                        text=current.strip(),
                        chunk_index=chunk_index,
                        token_estimate=max(1, len(current.split())),
                    )
                )
                overlap_words = current.split()[-self.settings.chunk_overlap // 6 :]
                current = " ".join(overlap_words + [sentence]).strip()
                chunk_index += 1
            else:
                current = tentative
        if current:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}-chunk-{chunk_index}",
                    doc_id=doc_id,
                    source_file=source_file,
                    title=title,
                    text=current.strip(),
                    chunk_index=chunk_index,
                    token_estimate=max(1, len(current.split())),
                )
            )
        return chunks

    def save_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        """Persist document-level chunks to JSONL."""
        write_jsonl(self.settings.chunks_dir / f"{doc_id}.jsonl", [chunk.model_dump() for chunk in chunks])
