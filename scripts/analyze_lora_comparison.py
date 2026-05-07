"""Compute simple generation-quality indicators for base Qwen vs LoRA.

These are not retrieval metrics. They summarize generated answers from the
comparison CSV produced in Kaggle.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


STRUCTURE_MARKERS = [
    "ответ",
    "ключевые",
    "преимущества",
    "ограничения",
    "вывод",
    "обоснование",
]

REFUSAL_PATTERNS = [
    "нельзя сделать вывод",
    "нельзя точно ответить",
    "не указано",
    "недостаточно",
    "нет информации",
]


def has_structure(text: object) -> int:
    lowered = str(text).lower()
    return int(any(marker in lowered for marker in STRUCTURE_MARKERS))


def has_refusal(text: object) -> int:
    lowered = str(text).lower()
    return int(any(pattern in lowered for pattern in REFUSAL_PATTERNS))


def cyrillic_ratio(text: object) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", str(text))
    if not letters:
        return 0.0
    cyrillic = [char for char in letters if re.match(r"[А-Яа-яЁё]", char)]
    return len(cyrillic) / len(letters)


def word_count(text: object) -> int:
    return len(str(text).split())


def answer_context_overlap(answer: object, context: object) -> float:
    answer_words = {
        word
        for word in re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", str(answer).lower())
        if word not in {"ответ", "вывод", "контекст", "основе", "предоставленного"}
    }
    context_words = set(re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", str(context).lower()))
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("kaggle/working/qwen3_lora_comparison_real_context.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/csv/qwen3_lora_generation_metrics.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    rows = []
    for column, name in [
        ("base_qwen3_8b", "Base Qwen3:8B"),
        ("qwen3_8b_lora", "Qwen3:8B + LoRA"),
    ]:
        rows.append(
            {
                "model": name,
                "avg_answer_words": round(df[column].apply(word_count).mean(), 2),
                "format_adherence": round(df[column].apply(has_structure).mean(), 3),
                "refusal_rate": round(df[column].apply(has_refusal).mean(), 3),
                "russian_language_rate": round(
                    df[column].apply(lambda text: cyrillic_ratio(text) >= 0.75).mean(),
                    3,
                ),
                "avg_context_overlap": round(
                    df.apply(lambda row: answer_context_overlap(row[column], row["context_short"]), axis=1).mean(),
                    3,
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False, encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
