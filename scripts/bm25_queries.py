"""
BM25 query definitions for job-relevance scoring.

Each query targets a distinct dimension from the job description.
Searched against career descriptions + profile summary — where
candidates describe what they actually built, not just keyword lists.
"""

BM25_QUERIES = {
    "production_ml": (
    "production machine learning deployed productionized "
    "model serving inference training evaluation "
    "feature engineering feature store "
    "embeddings embedding drift retrieval quality "
    "offline evaluation online evaluation "
    "ab testing experimentation "
    "pipeline scalable latency throughput"
    ),
    "retrieval_ranking": (
    "search retrieval ranking recommendation relevance matching "
    "personalization candidate matching information retrieval "
    "learning to rank ltr search relevance feed ranking ads ranking "
    "recommendation system recommendation engine "
    "ndcg mrr map ab testing experimentation "
    "elasticsearch opensearch lucene faiss "
    "hybrid retrieval semantic search"
    )
}

BM25_QUERY_WEIGHTS = {
    "production_ml": 1.0,
    "retrieval_ranking": 1.0,
}
