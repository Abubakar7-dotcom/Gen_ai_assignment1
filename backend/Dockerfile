# Build context = repository root (see docker-compose.yml) so the shared corruption module can be copied in.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/srv:/srv/backend \
    MODELS_DIR=/models SAMPLES_DIR=/srv/backend/samples
WORKDIR /srv/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# single source of truth for corruptions (same file the models were trained with)
COPY src/__init__.py /srv/src/__init__.py
COPY src/data/__init__.py src/data/corruptions.py /srv/src/data/
COPY backend/app ./app
COPY backend/samples ./samples
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
