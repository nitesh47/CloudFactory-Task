FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
 && poetry config virtualenvs.create false

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Ensure CPU-only onnxruntime
RUN pip uninstall -y onnxruntime-gpu 2>/dev/null || true \
 && pip install --no-cache-dir onnxruntime

# Copy application code
COPY src /app/src
COPY extract.py evaluate_all.py /app/
COPY scripts /app/scripts

# Pre-download DocTR models during build for offline use
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(det_arch='fast_base', reco_arch='parseq', pretrained=True)"

EXPOSE 8000

CMD ["python", "extract.py", "--help"]
