"""
Configuration for Custom RAG with Embedding and Reranking Models
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ragas Evaluator Configuration
RAGAS_EVALUATOR_MODEL = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini")

# LLM Configuration for Answer Generation
GENERATION_LLM = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_tokens": 500
}

# Embedding Models Configuration
EMBEDDING_MODELS: List[Dict] = [
    {
        "name": "openai_small",
        "display_name": "OpenAI text-embedding-3-small",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "collection_name": "uso_openai_small",
        "description": "Fast, cost-effective OpenAI embedding",
        "cost_per_1k_tokens": 0.00002
    },
    {
        "name": "mistral",
        "display_name": "Mistral Embed",
        "provider": "huggingface",
        "model": "mistralai/Mistral-7B-v0.1",  # Using Mistral via HF
        "dimensions": 1024,
        "collection_name": "uso_mistral",
        "description": "Mistral's embedding model via HuggingFace",
        "cost_per_1k_tokens": 0  # Free if local
    },
    {
        "name": "snowflake",
        "display_name": "Snowflake Arctic Embed Large",
        "provider": "huggingface",
        "model": "Snowflake/snowflake-arctic-embed-l",
        "dimensions": 1024,
        "collection_name": "uso_snowflake",
        "description": "Snowflake's Arctic embedding model",
        "cost_per_1k_tokens": 0  # Free if local
    },
    {
        "name": "baai",
        "display_name": "BAAI BGE Large EN v1.5",
        "provider": "huggingface",
        "model": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "collection_name": "uso_baai",
        "description": "BAAI's BGE large English embedding",
        "cost_per_1k_tokens": 0  # Free if local
    }
]

# Reranking Models Configuration
RERANKING_MODELS: List[Dict] = [
    {
        "name": "flashrank_default",
        "display_name": "FlashRank Default",
        "provider": "flashrank",
        "model": "ms-marco-MiniLM-L-12-v2",
        "description": "Fast reranking model for production",
        "enabled": True
    },
    {
        "name": "bge_reranker",
        "display_name": "BGE Reranker",
        "provider": "huggingface",
        "model": "BAAI/bge-reranker-large",
        "description": "High-quality reranking model",
        "enabled": False  # Optional, slower but more accurate
    }
]

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "top_k": 10,  # Number of chunks to retrieve initially
    "top_n_after_rerank": 5,  # Number of chunks after reranking
    "use_reranking": True,  # Enable reranking
    "similarity_threshold": 0.5,  # Minimum similarity score
}

# Document Chunking Configuration
CHUNKING_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", " ", ""]
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "provider": "chromadb",  # Options: "chromadb", "faiss"
    "persist_directory": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "vector_stores"
    ),
    "distance_metric": "cosine"  # cosine, euclidean, dot_product
}

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
SOURCE_DOCS_PATH = os.path.join(DATA_DIR, "source_docs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SOURCE_DOCS_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_CONFIG["persist_directory"], exist_ok=True)

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a technical support assistant for i-sprint's USO (Unified Security Operations) product.

Use the following retrieved context to answer the user's question. If you don't know the answer based on the context, say so. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

# Evaluation Settings
EVAL_SETTINGS = {
    "run_evaluation_per_query": True,  # Evaluate each query immediately
    "save_intermediate_results": True,  # Save after each model
    "verbose": True,  # Print progress
}

# Model Selection Helper
def get_embedding_model_config(model_name: str) -> Dict:
    """Get configuration for a specific embedding model"""
    for model in EMBEDDING_MODELS:
        if model["name"] == model_name:
            return model
    raise ValueError(f"Model {model_name} not found in configuration")

def get_reranker_config(reranker_name: str = None) -> Dict:
    """Get configuration for reranker model"""
    if reranker_name is None:
        # Return first enabled reranker
        for reranker in RERANKING_MODELS:
            if reranker.get("enabled", False):
                return reranker
        return RERANKING_MODELS[0]  # Default to first

    for reranker in RERANKING_MODELS:
        if reranker["name"] == reranker_name:
            return reranker
    raise ValueError(f"Reranker {reranker_name} not found in configuration")
