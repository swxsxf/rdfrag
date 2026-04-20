"""PDF parsing module with a local fallback parser for MVP usage."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from pypdf import PdfReader

from rdfrag_vkr.config import Settings, get_settings
from rdfrag_vkr.schemas import ArticleMetadata, ParsedDocument
from rdfrag_vkr.utils.io import make_document_id, write_json


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class PDFParser:
    """Parse PDFs from the corpus and persist structured results."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        self._grobid_available: bool | None = None

    def list_pdfs(self) -> list[Path]:
        """Return all PDF files from the raw corpus directory."""
        return sorted(self.settings.raw_pdf_dir.glob("*.pdf"))

    def parse_corpus(self) -> list[ParsedDocument]:
        """Parse the full raw PDF corpus and persist JSON outputs."""
        documents: list[ParsedDocument] = []
        pdf_files = self.list_pdfs()
        for pdf_path in pdf_files:
            try:
                document = self.parse_pdf(pdf_path)
                self.save_document(document)
                documents.append(document)
            except Exception as exc:  # pragma: no cover - best effort for noisy PDFs
                failed_path = self.settings.parsed_dir / f"{make_document_id(pdf_path.name)}.error.txt"
                failed_path.write_text(str(exc), encoding="utf-8")
        manifest = {
            "pdf_count": len(pdf_files),
            "parsed_count": len(documents),
            "parser_mode": "local_pypdf_fallback",
            "todo": "Integrate GROBID for richer scientific metadata and section parsing.",
        }
        write_json(self.settings.parsed_dir / "manifest.json", manifest)
        return documents

    def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        """Parse a single PDF using pypdf as a local fallback."""
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        full_text = "\n\n".join(pages).strip()

        metadata_map = reader.metadata or {}
        title = self._pick_title(pdf_path, metadata_map)
        authors = self._pick_authors(metadata_map)
        year = self._pick_year(pdf_path, metadata_map, full_text)
        abstract = self._pick_abstract(full_text)

        metadata = ArticleMetadata(
            doc_id=make_document_id(pdf_path.name),
            source_file=pdf_path.name,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            page_count=len(reader.pages),
            parser="pypdf",
        )
        return ParsedDocument(metadata=metadata, text=full_text, pages=pages)

    def save_document(self, document: ParsedDocument) -> Path:
        """Save parsed document to data/parsed."""
        output_path = self.settings.parsed_dir / f"{document.metadata.doc_id}.json"
        write_json(output_path, document.model_dump())
        return output_path

    @staticmethod
    def _pick_title(pdf_path: Path, metadata_map: object) -> str:
        title = getattr(metadata_map, "title", None)
        if title and str(title).strip():
            return str(title).strip()
        return pdf_path.stem.replace("_", " ").strip()

    @staticmethod
    def _pick_authors(metadata_map: object) -> list[str]:
        author_value = getattr(metadata_map, "author", None)
        if not author_value:
            return []
        authors = re.split(r"[;,/]| and ", str(author_value))
        return [author.strip() for author in authors if author.strip()]

    @staticmethod
    def _pick_year(pdf_path: Path, metadata_map: object, text: str) -> int | None:
        for candidate in (
            str(getattr(metadata_map, "creation_date", "")),
            pdf_path.name,
            text[:1500],
        ):
            match = re.search(r"(19|20)\d{2}", candidate)
            if match:
                return int(match.group(0))
        return None

    @staticmethod
    def _pick_abstract(text: str) -> str | None:
        match = re.search(
            r"(abstract|аннотация)\s*[:.\n]\s*(.{120,1200})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        abstract = re.split(
            r"\n\s*\n|keywords|ключевые слова",
            match.group(2),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return re.sub(r"\s+", " ", abstract).strip()[:1000]

    def parse_corpus(self) -> list[ParsedDocument]:
        """Parse the full raw PDF corpus and persist JSON outputs."""
        documents: list[ParsedDocument] = []
        parser_counts = {"grobid": 0, "pypdf": 0}
        pdf_files = self.list_pdfs()
        total = len(pdf_files)
        grobid_available = self.is_grobid_available()
        self.logger.info(
            "Parsing stage started: %s PDF files detected. GROBID available: %s",
            total,
            grobid_available,
        )
        if not grobid_available:
            self.logger.info("GROBID is unavailable, parser will use pypdf fallback for this run.")
        started_at = time.monotonic()
        for index, pdf_path in enumerate(pdf_files, start=1):
            item_started_at = time.monotonic()
            try:
                document = self.parse_pdf(pdf_path)
                self.save_document(document)
                parser_counts[document.metadata.parser] = parser_counts.get(document.metadata.parser, 0) + 1
                documents.append(document)
                self.logger.info(
                    "[PARSE %s/%s | remaining=%s] %s | parser=%s | pages=%s | item_elapsed=%.1fs | total_elapsed=%.1fs",
                    index,
                    total,
                    total - index,
                    pdf_path.name,
                    document.metadata.parser,
                    document.metadata.page_count,
                    time.monotonic() - item_started_at,
                    time.monotonic() - started_at,
                )
            except Exception as exc:  # pragma: no cover - best effort for noisy PDFs
                failed_path = self.settings.parsed_dir / f"{make_document_id(pdf_path.name)}.error.txt"
                failed_path.write_text(str(exc), encoding="utf-8")
                self.logger.exception(
                    "[PARSE %s/%s | remaining=%s] %s failed after %.1fs",
                    index,
                    total,
                    total - index,
                    pdf_path.name,
                    time.monotonic() - item_started_at,
                )
        manifest = {
            "pdf_count": len(pdf_files),
            "parsed_count": len(documents),
            "parser_mode": "grobid_with_pypdf_fallback",
            "parser_counts": parser_counts,
            "todo": "Improve section-level parsing and references extraction for the final thesis version.",
        }
        write_json(self.settings.parsed_dir / "manifest.json", manifest)
        return documents

    def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        """Parse a single PDF, preferring GROBID and falling back to pypdf."""
        grobid_document = self._parse_with_grobid(pdf_path)
        if grobid_document is not None:
            return grobid_document
        return self._parse_with_pypdf(pdf_path)

    def is_grobid_available(self) -> bool:
        """Check whether the GROBID service is reachable."""
        if self._grobid_available is not None:
            return self._grobid_available
        try:
            response = httpx.get(
                f"{self.settings.grobid_url}/api/isalive",
                timeout=min(self.settings.grobid_timeout_seconds, 10),
            )
            self._grobid_available = response.status_code == 200 and "true" in response.text.lower()
        except Exception:
            self._grobid_available = False
        return self._grobid_available

    def _parse_with_grobid(self, pdf_path: Path) -> ParsedDocument | None:
        """Parse one PDF with GROBID if the service is available."""
        if not self.is_grobid_available():
            return None
        try:
            with pdf_path.open("rb") as handle:
                response = httpx.post(
                    f"{self.settings.grobid_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, handle, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "includeRawCitations": "0",
                        "segmentSentences": "1",
                    },
                    timeout=self.settings.grobid_timeout_seconds,
                )
            response.raise_for_status()
            if not response.text.strip():
                return None
            return self._tei_to_document(pdf_path, response.text)
        except Exception:
            return None

    def _parse_with_pypdf(self, pdf_path: Path) -> ParsedDocument:
        """Parse a single PDF using pypdf as a reliable fallback."""
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        full_text = "\n\n".join(pages).strip()

        metadata_map = reader.metadata or {}
        metadata = ArticleMetadata(
            doc_id=make_document_id(pdf_path.name),
            source_file=pdf_path.name,
            title=self._pick_title(pdf_path, metadata_map),
            authors=self._pick_authors(metadata_map),
            year=self._pick_year(pdf_path, metadata_map, full_text),
            abstract=self._pick_abstract(full_text),
            page_count=len(reader.pages),
            parser="pypdf",
        )
        return ParsedDocument(metadata=metadata, text=full_text, pages=pages)

    def _tei_to_document(self, pdf_path: Path, xml_text: str) -> ParsedDocument:
        """Convert GROBID TEI XML into the common parsed-document schema."""
        root = ET.fromstring(xml_text)
        title = self._tei_text(root.find(".//tei:titleStmt/tei:title", TEI_NS)) or pdf_path.stem
        authors = [
            author
            for author in (
                self._tei_author_name(node) for node in root.findall(".//tei:sourceDesc//tei:author", TEI_NS)
            )
            if author
        ]
        abstract = self._tei_join(root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS))
        body_paragraphs = [self._tei_text(node) for node in root.findall(".//tei:text/tei:body//tei:p", TEI_NS)]
        body_text = "\n\n".join(paragraph for paragraph in body_paragraphs if paragraph).strip()
        page_count = len(root.findall(".//tei:pb", TEI_NS))
        year = self._pick_year(pdf_path, self._tei_publication_dates(root), body_text)

        metadata = ArticleMetadata(
            doc_id=make_document_id(pdf_path.name),
            source_file=pdf_path.name,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract or None,
            page_count=page_count,
            parser="grobid",
        )
        return ParsedDocument(
            metadata=metadata,
            text="\n\n".join(part for part in [abstract, body_text] if part).strip(),
            pages=[paragraph for paragraph in body_paragraphs if paragraph],
        )

    @staticmethod
    def _pick_abstract(text: str) -> str | None:
        match = re.search(
            r"(abstract|аннотация)\s*[:.\n]\s*(.{120,1200})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        abstract = re.split(
            r"\n\s*\n|keywords|ключевые слова",
            match.group(2),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return re.sub(r"\s+", " ", abstract).strip()[:1000]

    @staticmethod
    def _pick_abstract(text: str) -> str | None:
        match = re.search(
            r"(abstract|аннотация)\s*[:.\n]\s*(.{120,1200})",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        abstract = re.split(
            r"\n\s*\n|keywords|ключевые слова",
            match.group(2),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return re.sub(r"\s+", " ", abstract).strip()[:1000]

    @staticmethod
    def _tei_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return re.sub(r"\s+", " ", "".join(node.itertext())).strip()

    def _tei_join(self, nodes: list[ET.Element]) -> str:
        parts = [self._tei_text(node) for node in nodes]
        return "\n\n".join(part for part in parts if part)

    def _tei_author_name(self, node: ET.Element) -> str:
        forename = self._tei_join(node.findall(".//tei:forename", TEI_NS))
        surname = self._tei_text(node.find(".//tei:surname", TEI_NS))
        combined = f"{forename} {surname}".strip()
        return combined or self._tei_text(node)

    @staticmethod
    def _tei_publication_dates(root: ET.Element) -> str:
        values = []
        for node in root.findall(".//tei:publicationStmt//tei:date", TEI_NS):
            value = node.attrib.get("when", "") or "".join(node.itertext()).strip()
            if value:
                values.append(value)
        return " ".join(values)


class FinalPDFParser:
    """Final parser implementation with mandatory GROBID-first behavior."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(__name__)
        self._grobid_available: bool | None = None

    def list_pdfs(self) -> list[Path]:
        return sorted(self.settings.raw_pdf_dir.glob("*.pdf"))

    def parse_corpus(self) -> list[ParsedDocument]:
        documents: list[ParsedDocument] = []
        pdf_files = self.list_pdfs()
        total = len(pdf_files)
        grobid_available = self.is_grobid_available()
        if self.settings.grobid_mode == "required" and not grobid_available:
            raise RuntimeError(
                f"GROBID is required for this run but unavailable at {self.settings.grobid_url}. "
                "Start docker compose services before running ingestion."
            )

        parser_counts = {"grobid": 0, "pypdf": 0}
        started_at = time.monotonic()
        self.logger.info(
            "Parsing stage started: %s PDF files detected. GROBID available: %s. Mode: %s",
            total,
            grobid_available,
            self.settings.grobid_mode,
        )

        for index, pdf_path in enumerate(pdf_files, start=1):
            item_started_at = time.monotonic()
            try:
                document = self.parse_pdf(pdf_path)
                self.save_document(document)
                parser_counts[document.metadata.parser] = parser_counts.get(document.metadata.parser, 0) + 1
                documents.append(document)
                self.logger.info(
                    "[PARSE %s/%s | remaining=%s] %s | parser=%s | pages=%s | item_elapsed=%.1fs | total_elapsed=%.1fs",
                    index,
                    total,
                    total - index,
                    pdf_path.name,
                    document.metadata.parser,
                    document.metadata.page_count,
                    time.monotonic() - item_started_at,
                    time.monotonic() - started_at,
                )
            except Exception as exc:
                failed_path = self.settings.parsed_dir / f"{make_document_id(pdf_path.name)}.error.txt"
                failed_path.write_text(str(exc), encoding="utf-8")
                self.logger.exception(
                    "[PARSE %s/%s | remaining=%s] %s failed after %.1fs",
                    index,
                    total,
                    total - index,
                    pdf_path.name,
                    time.monotonic() - item_started_at,
                )
                if self.settings.grobid_mode == "required":
                    raise

        manifest = {
            "pdf_count": len(pdf_files),
            "parsed_count": len(documents),
            "parser_mode": "grobid_required" if self.settings.grobid_mode == "required" else "grobid_with_fallback",
            "parser_counts": parser_counts,
        }
        write_json(self.settings.parsed_dir / "manifest.json", manifest)
        return documents

    def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        grobid_document = self._parse_with_grobid(pdf_path)
        if grobid_document is not None:
            return grobid_document
        if self.settings.grobid_mode == "required":
            self.logger.warning(
                "GROBID parsing failed for %s; using controlled pypdf fallback while keeping GROBID as the primary parser.",
                pdf_path.name,
            )
        return self._parse_with_pypdf(pdf_path)

    def save_document(self, document: ParsedDocument) -> Path:
        output_path = self.settings.parsed_dir / f"{document.metadata.doc_id}.json"
        try:
            write_json(output_path, document.model_dump())
            return output_path
        except PermissionError:
            fallback_path = self.settings.parsed_dir / f"{document.metadata.doc_id}.rebuilt.json"
            self.logger.warning(
                "Primary parsed output is locked for %s, writing fallback artifact to %s",
                document.metadata.source_file,
                fallback_path.name,
            )
            write_json(fallback_path, document.model_dump())
            return fallback_path

    def is_grobid_available(self) -> bool:
        if self._grobid_available is not None:
            return self._grobid_available
        try:
            response = httpx.get(
                f"{self.settings.grobid_url}/api/isalive",
                timeout=min(self.settings.grobid_timeout_seconds, 10),
            )
            self._grobid_available = response.status_code == 200 and "true" in response.text.lower()
        except Exception:
            self._grobid_available = False
        return self._grobid_available

    def _parse_with_grobid(self, pdf_path: Path) -> ParsedDocument | None:
        if not self.is_grobid_available():
            return None
        try:
            with pdf_path.open("rb") as handle:
                response = httpx.post(
                    f"{self.settings.grobid_url}/api/processFulltextDocument",
                    files={"input": (pdf_path.name, handle, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "includeRawCitations": "0",
                        "segmentSentences": "1",
                    },
                    timeout=self.settings.grobid_timeout_seconds,
                )
            response.raise_for_status()
            return self._tei_to_document(pdf_path, response.text)
        except Exception:
            return None

    def _parse_with_pypdf(self, pdf_path: Path) -> ParsedDocument:
        reader = PdfReader(str(pdf_path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        full_text = "\n\n".join(pages).strip()
        metadata_map = reader.metadata or {}
        return ParsedDocument(
            metadata=ArticleMetadata(
                doc_id=make_document_id(pdf_path.name),
                source_file=pdf_path.name,
                title=self._pick_title(pdf_path, metadata_map),
                authors=self._pick_authors(metadata_map),
                year=self._pick_year(pdf_path, metadata_map, full_text),
                abstract=self._pick_abstract(full_text),
                page_count=len(reader.pages),
                parser="pypdf",
            ),
            text=full_text,
            pages=pages,
        )

    def _tei_to_document(self, pdf_path: Path, xml_text: str) -> ParsedDocument:
        root = ET.fromstring(xml_text)
        title = self._tei_text(root.find(".//tei:titleStmt/tei:title", TEI_NS)) or pdf_path.stem
        authors = [
            author
            for author in (
                self._tei_author_name(node) for node in root.findall(".//tei:sourceDesc//tei:author", TEI_NS)
            )
            if author
        ]
        abstract = self._tei_join(root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS))
        body_paragraphs = [self._tei_text(node) for node in root.findall(".//tei:text/tei:body//tei:p", TEI_NS)]
        body_text = "\n\n".join(paragraph for paragraph in body_paragraphs if paragraph).strip()
        page_count = len(root.findall(".//tei:pb", TEI_NS))
        year = self._pick_year(pdf_path, self._tei_publication_dates(root), body_text)
        metadata = ArticleMetadata(
            doc_id=make_document_id(pdf_path.name),
            source_file=pdf_path.name,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract or None,
            page_count=page_count,
            parser="grobid",
        )
        return ParsedDocument(
            metadata=metadata,
            text="\n\n".join(part for part in [abstract, body_text] if part).strip(),
            pages=[paragraph for paragraph in body_paragraphs if paragraph],
        )

    @staticmethod
    def _pick_title(pdf_path: Path, metadata_map: object) -> str:
        title = getattr(metadata_map, "title", None)
        if title and str(title).strip():
            return str(title).strip()
        return pdf_path.stem.replace("_", " ").strip()

    @staticmethod
    def _pick_authors(metadata_map: object) -> list[str]:
        author_value = getattr(metadata_map, "author", None)
        if not author_value:
            return []
        authors = re.split(r"[;,/]| and ", str(author_value))
        return [author.strip() for author in authors if author.strip()]

    @staticmethod
    def _pick_year(pdf_path: Path, metadata_map: object, text: str) -> int | None:
        for candidate in (str(getattr(metadata_map, "creation_date", "")), str(metadata_map), pdf_path.name, text[:2000]):
            match = re.search(r"(19|20)\d{2}", candidate)
            if match:
                return int(match.group(0))
        return None

    @staticmethod
    def _pick_abstract(text: str) -> str | None:
        match = re.search(r"(abstract|аннотация)\s*[:.\n]\s*(.{120,1800})", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        abstract = re.split(r"\n\s*\n|keywords|ключевые слова", match.group(2), maxsplit=1, flags=re.IGNORECASE)[0]
        return re.sub(r"\s+", " ", abstract).strip()[:1200]

    @staticmethod
    def _tei_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return re.sub(r"\s+", " ", "".join(node.itertext())).strip()

    def _tei_join(self, nodes: list[ET.Element]) -> str:
        parts = [self._tei_text(node) for node in nodes]
        return "\n\n".join(part for part in parts if part)

    def _tei_author_name(self, node: ET.Element) -> str:
        forename = self._tei_join(node.findall(".//tei:forename", TEI_NS))
        surname = self._tei_text(node.find(".//tei:surname", TEI_NS))
        combined = f"{forename} {surname}".strip()
        return combined or self._tei_text(node)

    @staticmethod
    def _tei_publication_dates(root: ET.Element) -> str:
        values = []
        for node in root.findall(".//tei:publicationStmt//tei:date", TEI_NS):
            value = node.attrib.get("when", "") or "".join(node.itertext()).strip()
            if value:
                values.append(value)
        return " ".join(values)


PDFParser = FinalPDFParser
