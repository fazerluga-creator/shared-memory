FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SHARED_MEMORY_HOME=/data

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install . && mkdir -p /data

# No credentials needed: SQLite backend + local fastembed (the model is
# downloaded lazily on the first search/add, not at startup).
VOLUME ["/data"]
CMD ["shared-memory-mcp"]
