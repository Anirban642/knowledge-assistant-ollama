FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user; /code/data holds uploaded files and the vector database
RUN useradd --create-home appuser \
    && mkdir -p /code/data \
    && chown -R appuser /code/data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=3)"

# One worker on purpose: chat history is kept in memory (see README, "Known limitations")
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]