FROM python:3.11-slim

WORKDIR /app

# System deps needed by pypdf/bs4/chromadb's onnx runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Render/Railway inject $PORT at runtime and route traffic to it — this
# must NOT be a hardcoded port for those platforms to work. Defaults to
# 8000 for plain local `docker compose up`.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
