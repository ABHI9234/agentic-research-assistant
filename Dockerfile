FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first — much smaller than default
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
COPY render-requirements.txt .
RUN pip install --no-cache-dir -r render-requirements.txt

COPY . .

EXPOSE 8000

CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
