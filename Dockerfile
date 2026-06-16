# Lightweight Dockerfile for reproduction
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts and data
COPY . .

# Run the full pipeline
CMD ["sh", "-c", "python run_pipeline.py --candidates data/candidates.jsonl --out final_rankings.csv"]
