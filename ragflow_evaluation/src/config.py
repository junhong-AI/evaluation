"""
Configuration for multi-embedding RAG evaluation
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAGFlow Configuration
RAGFLOW_CONFIG = {
    "base_url": os.getenv("RAGFLOW_BASE_URL", "http://172.25.184.41:80"),
    "api_key": os.getenv("RAGFLOW_API_KEY", "ragflow-NlZTM0MWZjOGRmYzExZjA5NTU1NjZlNz"),
    "chat_id": os.getenv("RAGFLOW_CHAT_ID", "c78456e48e0c11f0be8956e79b878cc6"),
}

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ragas Evaluator Configuration
RAGAS_EVALUATOR_MODEL = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini")

# Embedding Models to Test
# Note: RAGFlow requires these models to be configured in separate knowledge bases
EMBEDDING_MODELS: List[Dict] = [
    {
        "name": "openai_small",
        "display_name": "OpenAI text-embedding-3-small",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "ragflow_model_name": "text-embedding-3-small",  # Model name in RAGFlow
        "description": "Fast, cost-effective OpenAI embedding"
    },
    {
        "name": "mistral",
        "display_name": "Mistral Embed",
        "provider": "mistral",
        "model": "mistral-embed",
        "dimensions": 1024,
        "ragflow_model_name": "mistral-embed",
        "description": "Mistral's embedding model"
    },
    {
        "name": "snowflake",
        "display_name": "Snowflake Arctic Embed Large",
        "provider": "snowflake",
        "model": "snowflake-arctic-embed-l",
        "dimensions": 1024,
        "ragflow_model_name": "snowflake-arctic-embed-l",
        "description": "Snowflake's Arctic embedding model"
    },
    {
        "name": "baai",
        "display_name": "BAAI BGE Large EN v1.5",
        "provider": "baai",
        "model": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "ragflow_model_name": "BAAI/bge-large-en-v1.5",
        "description": "BAAI's BGE large English embedding"
    }
]

# Ragas Metrics Configuration
RAGAS_METRICS = {
    "context_precision": True,
    "context_recall": True,
    "faithfulness": True,
    "answer_relevancy": True,
}

# Data Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TEST_DATASET_PATH = os.path.join(DATA_DIR, "test_qa_pairs.jsonl")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Evaluation Settings
EVAL_SETTINGS = {
    "stream": False,  # Don't stream responses
    "temperature": 0.1,  # Low temperature for consistent answers
    "max_retries": 3,  # Retry failed queries
    "timeout": 60,  # Request timeout in seconds
}
