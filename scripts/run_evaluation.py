"""Run retrieval evaluation over the sample gold set."""

from __future__ import annotations

import json

from rdfrag_vkr.modules.evaluation import Evaluator


if __name__ == "__main__":
    report = Evaluator().evaluate_retrieval()
    print(json.dumps(report, ensure_ascii=False, indent=2))
