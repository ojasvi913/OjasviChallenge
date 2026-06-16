# Intelligent Candidate Discovery & Ranking Submission
Team Antigravity

## Overview
Our pipeline processes 100K+ raw candidate resumes into a highly targeted top-100 ranking optimized for production retrieval, rigorous evaluation methodologies, and strong software engineering fundamentals. 

The pipeline filters out "model-tweakers" and keyword-stuffers in favor of engineers who have actually deployed robust, large-scale systems at product companies.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Our pipeline uses standard data science libraries (pandas, scikit-learn, rank-bm25) and python-docx. No GPUs or external APIs are required.*

2. **Data Prerequisites:**
   Ensure the competition data files are located in a `data/` directory relative to the repository root.

## Reproducing the Ranking

You can reproduce our entire ranking pipeline from the raw `candidates.jsonl` to the `final_rankings.csv` in less than 2 minutes using our single orchestrated command:

```bash
python scripts/resumescore.py --input data/candidates.jsonl
python scripts/bm25_score.py
python scripts/feature_engineering.py
python scripts/ranking.py
```

### Pipeline Stages

1. **`resumescore.py` (The Fast Filter):** 
   A fast streaming filter that discards ~85% of candidates based on hard constraints: < 4.0 Years of Experience, or failing to meet a minimum technical vocabulary threshold (meaning they lack basic engineering/data/ML keywords).
2. **`bm25_score.py` (The Relevance Engine):** 
   Parses the JD using `python-docx`, extracts critical search/ranking keywords, and runs an Okapi BM25 index over the remaining 15K candidates to extract the top 600 most semantically relevant profiles.
3. **`feature_engineering.py` (The Depth Analyzer):** 
   Extracts 30+ sophisticated features from the top 600 candidates. This includes:
   - **Depth Markers**: Counts of production/scale, evaluation (NDCG/MAP), and retrieval (FAISS/Pinecone) evidence.
   - **Company Ratios**: Distinguishes product company tenure from consulting.
   - **Honeypot Guards**: Validates chronological timelines and keyword stuffing.
4. **`ranking.py` (The Final Ranker):** 
   Combines the BM25 base score with aggressive depth multipliers and honeypot penalties to produce the final `final_rankings.csv`. It explicitly rewards candidates with domain experience in HR-Tech/Marketplaces and generates human-readable reasoning strings.

## Environment Constraints
- **Hardware:** CPU only. Requires < 4GB RAM.
- **Runtime:** < 2 minutes end-to-end for 100K candidates.
- **External APIs:** None. All scoring is deterministic and local.
