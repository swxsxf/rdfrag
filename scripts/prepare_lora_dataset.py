"""Prepare a small QLoRA/SFT dataset from existing RAG artifacts.

The script does not run ingestion and does not rebuild indexes. It only reads
already prepared chunks and gold queries, then writes chat-style JSONL files
that can be uploaded to Kaggle for LoRA fine-tuning experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "chunks" / "all_chunks.jsonl"
DEFAULT_GOLD = ROOT / "data" / "eval" / "gold_queries.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fine_tuning"

SYSTEM_PROMPT = (
    "Ты — эксперт-аналитик по цифровой экономике. Отвечай на русском языке, "
    "кратко, структурированно и только по предоставленному контексту. "
    "Не упоминай score, chunk, API и внутреннюю диагностику."
)

TOPICS: dict[str, list[str]] = {
    "блокчейн": ["блокчейн", "blockchain", "distributed ledger", "dlt"],
    "цифровые двойники": ["цифровой двойник", "цифровые двойники", "digital twin", "digital twins"],
    "IoT": ["iot", "интернет вещей", "internet of things", "internet of vehicles"],
    "6G": ["6g", "5g", "b5g"],
    "метавселенная": ["метавселен", "metaverse"],
    "low-code": ["low-code", "low code", "низким кодом", "малым кодом"],
    "умный город": ["smart city", "smart cities", "умный город", "умного города"],
    "умный транспорт": ["smart transportation", "smart mobility", "умный транспорт", "транспорт"],
}

QUESTION_TEMPLATES: dict[str, list[str]] = {
    "блокчейн": [
        "Что говорится о блокчейне в предоставленном контексте?",
        "Какие функции блокчейна описаны в документах?",
    ],
    "цифровые двойники": [
        "Что говорится о цифровых двойниках?",
        "Как цифровые двойники используются в рассматриваемой области?",
    ],
    "IoT": [
        "Что такое IoT и как он применяется в цифровой экономике?",
        "Какие задачи решает интернет вещей в найденных материалах?",
    ],
    "6G": [
        "Какая роль 6G описана в материалах?",
        "С какими технологиями связывается развитие 6G?",
    ],
    "метавселенная": [
        "Какие технологии используются в исследованиях метавселенной?",
        "Как метавселенная связана с цифровыми двойниками и IoT?",
    ],
    "low-code": [
        "Что говорится о low-code платформах?",
        "Какие преимущества и ограничения low-code разработки описаны в материалах?",
    ],
    "умный город": [
        "Какие темы связаны с умными городами?",
        "Как цифровые технологии применяются в smart city?",
    ],
    "умный транспорт": [
        "Какие технологии используются в умном транспорте?",
        "Как данные применяются в интеллектуальных транспортных системах?",
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"(?<=[.!?])(?=[A-ZА-ЯЁ0-9])", " ", text)
    text = re.sub(r"(?<=[а-яёa-z])(?=[А-ЯЁ])", " ", text)
    text = text.replace(" .", ".").replace(" ,", ",")
    return text


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [part.strip(" -•") for part in parts if 45 <= len(part.strip()) <= 420]
    return sentences


def normalize_filename(text: str) -> str:
    text = text.lower().replace(".pdf", "")
    text = re.sub(r"[^a-zа-яё0-9]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def cyrillic_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text or "")
    if not letters:
        return 0.0
    cyrillic = [char for char in letters if re.match(r"[А-Яа-яЁё]", char)]
    return len(cyrillic) / len(letters)


def topic_for_chunk(chunk: dict[str, Any]) -> str | None:
    haystack = f"{chunk.get('title', '')} {chunk.get('text', '')}".lower()
    for topic, aliases in TOPICS.items():
        if any(alias.lower() in haystack for alias in aliases):
            return topic
    return None


def score_sentence(sentence: str, question: str, topic: str | None) -> int:
    lowered = sentence.lower()
    words = [word for word in re.findall(r"[a-zа-яё0-9]{3,}", question.lower()) if len(word) > 3]
    score = sum(2 for word in words if word in lowered)
    if topic:
        score += sum(3 for alias in TOPICS.get(topic, []) if alias.lower() in lowered)
    score += min(len(sentence) // 180, 3)
    return score


def compact_sentence(sentence: str, limit: int = 230) -> str:
    sentence = clean_text(sentence).strip(" -•,;")
    if len(sentence) <= limit:
        return sentence
    shortened = sentence[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened + "."


def select_points(chunks: list[dict[str, Any]], question: str, topic: str | None, max_points: int = 4) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    order = 0
    for chunk in chunks:
        for sentence in split_sentences(chunk.get("text", "")):
            if cyrillic_ratio(sentence) < 0.35:
                continue
            candidates.append((score_sentence(sentence, question, topic), order, sentence))
            order += 1
    candidates.sort(key=lambda item: (-item[0], item[1]))

    points: list[str] = []
    seen: set[str] = set()
    for score, _, sentence in candidates:
        if score <= 0 and len(points) >= 2:
            continue
        sentence = compact_sentence(sentence)
        key = sentence.lower()[:90]
        if key not in seen:
            seen.add(key)
            points.append(sentence)
        if len(points) >= max_points:
            break
    return points


def build_context(chunks: list[dict[str, Any]], max_chars: int = 2800) -> str:
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        text = clean_text(chunk.get("text", ""))[:950]
        block = (
            f"[Фрагмент {index}]\n"
            f"Документ: {chunk.get('source_file', '')}\n"
            f"Название: {chunk.get('title', '')}\n"
            f"Текст: {text}"
        )
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_answer(question: str, chunks: list[dict[str, Any]], topic: str | None) -> str:
    points = select_points(chunks, question, topic)
    if not points:
        points = ["В предоставленном контексте тема раскрыта ограниченно, поэтому вывод следует считать предварительным."]

    topic_part = f" по теме «{topic}»" if topic else ""
    lead = f"Ответ: На основе предоставленного контекста{topic_part} можно сделать следующий вывод: {points[0]}"
    if len(points) > 1:
        continuation = points[1][0].lower() + points[1][1:] if points[1] else points[1]
        lead += f" Также важно, что {continuation}"

    bullet_points = points[:4]
    bullets = "\n".join(f"- {point}" for point in bullet_points)
    if len(bullet_points) < 3:
        bullets += "\n- Вывод ограничен предоставленными фрагментами и не должен расширяться за их пределы."

    return f"{lead}\n\nКлючевые пункты:\n{bullets}"


def make_example(question: str, chunks: list[dict[str, Any]], topic: str | None) -> dict[str, Any]:
    context = build_context(chunks)
    answer = build_answer(question, chunks, topic)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос: {question}\n\nКонтекст:\n{context}"},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "topic": topic or "gold",
            "source_files": sorted({chunk.get("source_file", "") for chunk in chunks if chunk.get("source_file")}),
            "chunk_ids": [chunk.get("chunk_id", "") for chunk in chunks],
        },
    }


def match_gold_chunks(gold_item: dict[str, Any], chunks: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    expected_files = [normalize_filename(item) for item in gold_item.get("expected_source_files", [])]
    if not expected_files:
        return []

    matched: list[dict[str, Any]] = []
    for chunk in chunks:
        source = normalize_filename(chunk.get("source_file", ""))
        title = normalize_filename(chunk.get("title", ""))
        if any(expected in source or source in expected or expected in title for expected in expected_files):
            matched.append(chunk)
        if len(matched) >= limit:
            break
    return matched


def build_examples(chunks: list[dict[str, Any]], gold: list[dict[str, Any]], max_synthetic: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    examples: list[dict[str, Any]] = []

    for item in gold:
        question = item.get("question", "").strip()
        matched = match_gold_chunks(item, chunks)
        if question and matched:
            examples.append(make_example(question, matched, topic_for_chunk(matched[0])))

    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        text = clean_text(chunk.get("text", ""))
        if len(text) < 350 or cyrillic_ratio(text) < 0.35:
            continue
        topic = topic_for_chunk(chunk)
        if topic:
            by_topic[topic].append(chunk)

    per_topic = max(1, max_synthetic // max(1, len(TOPICS)))
    seen_pairs: set[tuple[str, str]] = set()
    for topic, topic_chunks in by_topic.items():
        random.shuffle(topic_chunks)
        questions = QUESTION_TEMPLATES.get(topic, [f"Что говорится о теме {topic}?"])
        count = 0
        for chunk in topic_chunks:
            question = questions[count % len(questions)]
            key = (chunk.get("chunk_id", ""), question)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            examples.append(make_example(question, [chunk], topic))
            count += 1
            if count >= per_topic:
                break

    random.shuffle(examples)
    return examples


def write_alpaca_copy(path: Path, examples: list[dict[str, Any]]) -> None:
    rows: list[dict[str, str]] = []
    for example in examples:
        messages = example["messages"]
        rows.append(
            {
                "instruction": messages[0]["content"],
                "input": messages[1]["content"],
                "output": messages[2]["content"],
            }
        )
    write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen LoRA dataset from existing RAG chunks.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-synthetic", type=int, default=160)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)
    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold.exists() else []
    examples = build_examples(chunks, gold, max_synthetic=args.max_synthetic, seed=args.seed)

    valid_size = max(1, round(len(examples) * args.valid_ratio))
    valid = examples[:valid_size]
    train = examples[valid_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "qwen3_rag_lora_train.jsonl", train)
    write_jsonl(args.output_dir / "qwen3_rag_lora_valid.jsonl", valid)
    write_jsonl(args.output_dir / "qwen3_rag_lora_all_messages.jsonl", examples)
    write_alpaca_copy(args.output_dir / "qwen3_rag_lora_all_alpaca.jsonl", examples)

    summary = {
        "total_examples": len(examples),
        "train_examples": len(train),
        "valid_examples": len(valid),
        "topics": dict(sorted({topic: sum(1 for row in examples if row["metadata"]["topic"] == topic) for topic in TOPICS}.items())),
        "source": {
            "chunks": str(args.chunks),
            "gold": str(args.gold),
            "note": "Prepared from existing artifacts only; ingestion was not executed.",
        },
    }
    (args.output_dir / "qwen3_rag_lora_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
