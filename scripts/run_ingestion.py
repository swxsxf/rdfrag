"""Run the full ingestion pipeline."""

from __future__ import annotations

import json
import logging

from rdfrag_vkr.pipelines.ingestion import run_ingestion


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = run_ingestion()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
