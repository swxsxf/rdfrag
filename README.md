# RDFRAG VKR

Гибридная `graph + vector RAG`-система для корпуса научных PDF-статей по цифровой экономике и смежным цифровым технологиям.

Проект реализует полный pipeline:
- парсинг PDF (`GROBID -> pypdf fallback`)
- preprocessing и chunking
- knowledge extraction
- построение RDF-графа знаний
- графовый retrieval через `Fuseki / SPARQL`
- векторный retrieval через `FAISS`
- hybrid fusion + reranking
- answer generation через `Ollama + qwen3:8b`
- API на `FastAPI` и UI на `Gradio`

## Технологии

- Python
- FastAPI
- Gradio
- Apache Jena Fuseki
- rdflib
- FAISS
- Ollama
- qwen3:8b
- deepvk/USER-base

## Структура

```text
data/
  eval/            # evaluation inputs/results
  rdf/             # RDF graph and knowledge artifacts
artifacts/
  metrics/         # CSV/JSON metrics
  plots/           # generated plots and visualizations
  reports/         # markdown/html reports
scripts/
src/rdfrag_vkr/
tests/
```

## Основные команды

Установка:

```bash
pip install -e .
```

Запуск ingestion:

```bash
python scripts/run_ingestion.py
```

Загрузка RDF в Fuseki:

```bash
python scripts/upload_rdf.py
```

Запуск evaluation:

```bash
python scripts/run_evaluation.py
```

Запуск API:

```bash
python main.py --mode api --host 0.0.0.0 --port 8000
```

Запуск Gradio UI:

```bash
python main.py
```

## Что есть в репозитории

- исходный код пайплайна и API
- тесты
- evaluation metrics
- готовые графики и визуализации для ВКР
- RDF-артефакты и evaluation-данные

## Что не хранится в репозитории

Из репозитория исключены тяжёлые локальные данные:
- исходные PDF
- parsed outputs
- chunks
- embeddings

Это сделано, чтобы репозиторий оставался компактным и пригодным для GitHub.
