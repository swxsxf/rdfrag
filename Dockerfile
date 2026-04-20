FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN mkdir -p /app/data/raw_pdfs /app/data/parsed /app/data/chunks /app/data/rdf /app/data/embeddings /app/data/eval

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -e ".[full]"

CMD ["uvicorn", "rdfrag_vkr.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
