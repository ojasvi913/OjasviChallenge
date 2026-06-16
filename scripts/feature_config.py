"""
Configuration constants for feature engineering.

Term lists, company classifications, title scoring tiers,
and honeypot detection thresholds.
"""

# ---------------------------------------------------------------------------
# Depth feature term lists (searched in career_history descriptions)
# ---------------------------------------------------------------------------

RETRIEVAL_DEPTH_TERMS = [
    "search", "retrieval", "ranking", "recommendation", "relevance",
    "matching", "personalization", "learning to rank", "ltr",
    "re-ranking", "reranking", "information retrieval",
    "semantic search", "hybrid retrieval", "vector search",
    "dense retrieval", "sparse retrieval", "bm25", "tf-idf",
    "query understanding", "search quality", "candidate matching",
    "feed ranking", "ads ranking", "content ranking",
    "elasticsearch", "opensearch", "solr", "lucene",
    "faiss", "annoy", "hnsw",
    "pinecone", "weaviate", "qdrant", "milvus",
    "embedding", "embeddings", "similarity",
]

EVALUATION_DEPTH_TERMS = [
    "ndcg", "mrr", "map", "mean average precision",
    "precision", "recall", "f1",
    "offline evaluation", "online evaluation",
    "a/b test", "ab test", "a/b testing", "ab testing",
    "experimentation", "experiment", "evaluation framework",
    "benchmark", "test set", "holdout",
    "cross-validation", "cross validation",
    "metrics", "ground truth",
    "annotation", "labeling", "human evaluation",
]

PRODUCTION_DEPTH_TERMS = [
    "deployed", "shipped", "production", "productionized",
    "serving", "model serving", "inference",
    "latency", "throughput", "scale", "scaled",
    "real users", "live traffic", "real-time",
    "sla", "uptime", "monitoring", "alerting",
    "ci/cd", "cicd", "continuous integration",
    "kubernetes", "k8s", "docker", "container",
    "microservice", "api", "endpoint", "rest api",
    "load balancing", "auto-scaling", "autoscaling",
    "mlops", "ml pipeline", "feature store",
    "model registry", "model monitoring",
    "data pipeline", "etl", "airflow",
    "batch processing", "stream processing", "kafka",
]

# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini",
    "hcl technologies", "hcl tech",
    "tech mahindra", "mindtree", "ltimindtree",
    "mphasis", "l&t infotech", "lti",
    "persistent systems", "hexaware",
    "birlasoft", "zensar", "niit technologies",
    "cyient", "coforge", "sonata software",
    "deloitte", "ey", "ernst & young", "kpmg", "pwc",
    "mckinsey", "boston consulting", "bcg", "bain",
]

CONSULTING_INDUSTRY_KEYWORDS = [
    "consulting", "it services", "information technology and services",
    "staffing", "outsourcing", "professional services",
    "managed services", "bpo", "business process",
]

PRODUCT_COMPANIES = [
    # Indian product companies
    "swiggy", "zomato", "flipkart", "myntra", "meesho",
    "paytm", "phonpe", "razorpay", "cred", "groww",
    "ola", "rapido", "nykaa", "bigbasket", "dunzo",
    "dream11", "sharechat", "moj", "lenskart", "urban company",
    "zerodha", "freshworks", "zoho", "postman", "browserstack",
    "hasura", "chargebee", "druva", "icertis",
    "byju", "unacademy", "upgrad", "vedantu",
    # Global product companies
    "google", "alphabet", "meta", "facebook", "amazon", "apple",
    "microsoft", "netflix", "uber", "lyft", "airbnb",
    "spotify", "stripe", "databricks", "snowflake",
    "linkedin", "twitter", "x corp", "pinterest", "snap",
    "salesforce", "adobe", "atlassian", "slack",
    "openai", "anthropic", "cohere", "hugging face",
    "doordash", "instacart", "grubhub",
    "nvidia", "intel", "amd",
    "shopify", "square", "block",
    "dropbox", "notion", "figma", "canva",
    "redrob",
]

PRODUCT_INDUSTRY_KEYWORDS = [
    "internet", "computer software", "e-commerce",
    "fintech", "financial technology",
    "consumer electronics", "social media",
    "online media", "gaming", "edtech",
]

# ---------------------------------------------------------------------------
# Penalty detection terms
# ---------------------------------------------------------------------------

WRAPPER_TERMS = [
    "langchain", "llamaindex", "llama index",
    "openai api", "chatgpt", "gpt-4", "gpt-3",
    "prompt engineering", "prompt design",
    "rag", "retrieval augmented generation",
    "chain of thought", "few-shot",
    "auto-gpt", "autogpt", "agent framework",
]

DEEPER_ML_TERMS = [
    "pytorch", "tensorflow", "keras", "jax",
    "opencv", "yolo", "spacy", "nltk", "gensim",
]

DOMAIN_KEYWORDS = [
    "hr tech", "hr-tech", "recruiting", "recruitment", "job board", "applicant tracking",
    "ats", "marketplace", "two-sided market", "e-commerce", "ecommerce",
    "matching engine", "talent acquisition"
]

DEEPER_ML_TERMS_SECONDARY = [
    "custom model", "fine-tuning", "fine tuning", "finetuning",
    "training loop", "loss function", "backpropagation",
    "embedding model", "sentence-transformers", "sentence transformers",
    "bert", "transformer", "attention mechanism",
    "xgboost", "lightgbm", "gradient boosting",
    "feature engineering", "feature store",
    "vector database", "faiss", "approximate nearest neighbor",
    "distributed training", "model optimization",
    "quantization", "distillation", "pruning",
    "scikit-learn", "sklearn",
]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection",
    "image segmentation", "opencv", "yolo", "resnet", "vgg",
    "convolutional neural network", "cnn",
    "speech recognition", "speech synthesis", "tts", "asr",
    "voice", "audio processing", "speech to text",
    "robotics", "ros", "autonomous", "self-driving",
    "slam", "lidar", "sensor fusion", "control systems",
    "point cloud", "depth estimation",
]

NLP_IR_TERMS = [
    "nlp", "natural language processing",
    "information retrieval", "search", "retrieval",
    "text classification", "ner", "named entity",
    "sentiment analysis", "topic modeling",
    "language model", "transformer", "bert",
    "tokenization", "embedding", "word2vec",
    "ranking", "relevance", "query",
]

RESEARCH_TERMS = [
    "research", "paper", "publication", "published",
    "conference", "journal", "arxiv",
    "phd", "doctoral", "thesis", "dissertation",
    "academic", "university", "professor",
    "lab", "laboratory", "research scientist",
    "novel approach", "state of the art", "sota",
]

PRODUCTION_EVIDENCE_TERMS = [
    "production", "deployed", "shipped", "launched",
    "real users", "live", "serving",
    "scale", "million", "billion",
    "api", "service", "microservice",
    "monitoring", "alerting", "on-call",
]

# ---------------------------------------------------------------------------
# Title scoring tiers (checked in order — first match wins)
# ---------------------------------------------------------------------------

TITLE_SCORE_MAP = [
    # Tier 1: Search/NLP/ML/AI specialists
    (5, [
        "machine learning", "ml engineer", "ai engineer",
        "search engineer", "retrieval engineer", "ranking engineer",
        "relevance engineer", "recommendation engineer",
        "nlp engineer", "nlu engineer",
        "data scientist", "research engineer", "applied scientist",
    ]),
    # Tier 2: Backend / platform / data
    (4, [
        "backend engineer", "backend developer",
        "platform engineer", "infrastructure engineer",
        "data engineer", "distributed systems",
        "site reliability", "sre","devops",
    ]),
    # Tier 3: General software engineering
    (3, [
        "software engineer", "software developer",
        "full stack", "full-stack",
        "developer", "swe", "sde",
    ]),
    # Tier 4: Frontend / mobile
    (2, [
        "frontend", "front-end", "front end",
        "mobile developer", "ios developer", "android developer",
        "react developer", "angular developer",
        "ui engineer", "ux engineer",
    ]),
    # Tier 5: QA / testing / DevOps
    (1, [
        "qa", "quality assurance", "test engineer", "tester",
        "release engineer",
        "support engineer", "technical support",
    ]),
]

# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

BUZZWORD_TERMS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "data science", "big data", "blockchain",
    "cloud", "agile", "devops", "digital transformation",
    "innovative", "cutting-edge", "state-of-the-art",
    "passionate", "dynamic", "results-driven",
    "leverage", "synergy", "paradigm", "disruptive",
    "thought leader", "visionary",
    "generative ai", "chatgpt", "llm",
]

# Seniority keyword → level mapping for title progression detection
SENIORITY_LEVELS = {
    "intern": 0, "trainee": 0, "apprentice": 0,
    "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3, "sr": 3, "lead": 3,
    "staff": 4, "principal": 4, "architect": 4,
    "director": 5, "vp": 5, "vice president": 5,
    "manager": 3, "senior manager": 4,
    "head": 5, "cto": 6, "ceo": 6, "chief": 6,
}
