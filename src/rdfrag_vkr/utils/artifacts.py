"""Helpers for saving experiment artifacts for the thesis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    """Write rows into a UTF-8 CSV file."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: Iterable[dict]) -> None:
    """Persist a markdown table for direct insertion into the thesis."""
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("| metric | value |\n|---|---|\n", encoding="utf-8")
        return
    path.write_text(frame.to_markdown(index=False), encoding="utf-8")


def write_html_report(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    """Write a simple self-contained HTML report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"<h2>{heading}</h2>\n<div>{content}</div>" for heading, content in sections)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
    h1, h2 {{ color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {body}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_json_summary(path: Path, payload: object) -> None:
    """Write a thesis artifact JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
