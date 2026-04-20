"""I/O and identifier helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


def make_document_id(source_name: str) -> str:
    """Build a stable ASCII document id from the original file name."""
    stem = Path(source_name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem.encode("ascii", "ignore").decode("ascii")).strip("-")
    digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:8]
    slug = slug[:48] if slug else "document"
    return f"{slug}-{digest}"


def make_entity_id(entity_type: str, label: str) -> str:
    """Build a stable entity id."""
    base = f"{entity_type}:{label.strip().lower()}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{entity_type.lower()}-{digest}"


def write_json(path: Path | str, payload: object) -> None:
    """Write JSON with UTF-8 and pretty formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    """Write JSONL rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path | str) -> object:
    """Read JSON content."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict]:
    """Read JSONL rows."""
    path = Path(path)
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
