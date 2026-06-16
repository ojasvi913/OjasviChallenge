# Lightweight Dockerfile for reproduction
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts and data
COPY . .

# Run the full pipeline
CMD ["sh", "-c", "python scripts/resumescore.py && python scripts/bm25_score.py && python scripts/feature_engineering.py && python scripts/ranking.py"]
