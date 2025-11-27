# RAGFlow Multi-Embedding Evaluation Pipeline

## Overview
This project evaluates RAG (Retrieval-Augmented Generation) performance across multiple embedding models using RAGFlow API and Ragas evaluation framework.

## Embedding Models Tested
1. **OpenAI**: text-embedding-3-small (1536 dims)
2. **Mistral**: mistral-embed (1024 dims)
3. **Snowflake**: snowflake-arctic-embed-l (1024 dims)
4. **BAAI**: bge-large-en-v1.5 (1024 dims)

## Test Dataset
- **Source**: USO Administration Guide 6.0.1-GA
- **Test Questions**: 10 Q&A pairs
- **Format**: CSV with columns: query, ground_truth, contexts_joined

## Project Structure
```
ragflow_evaluation/
├── data/
│   ├── test_qa_pairs.jsonl       # Standardized test dataset
│   └── source_docs/              # Reference documents
├── notebooks/
│   ├── 00_setup_check.ipynb      # Verify connections and setup
│   ├── 01_single_model_test.ipynb # Baseline with one model
│   └── 02_multi_model_eval.ipynb  # Full comparison
├── src/
│   ├── config.py                 # Model configurations
│   ├── ragflow_client.py         # RAGFlow API wrapper
│   ├── evaluation_pipeline.py    # Core evaluation logic
│   └── utils.py                  # Helper functions
├── results/                      # Evaluation outputs
├── requirements.txt
└── .env
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
OPENAI_API_KEY=your_key
RAGFLOW_API_KEY=your_key
RAGFLOW_BASE_URL=http://172.25.184.41:80
```

3. Run notebooks in order:
   - `00_setup_check.ipynb` - Verify setup
   - `01_single_model_test.ipynb` - Baseline test
   - `02_multi_model_eval.ipynb` - Full evaluation

## Evaluation Metrics (Ragas)
- **Context Precision**: Relevance of retrieved chunks
- **Context Recall**: Coverage of reference contexts
- **Faithfulness**: Answer grounded in retrieved context
- **Answer Relevancy**: Answer relevance to question

## RAGFlow Configuration
- **Chat ID**: c78456e48e0c11f0be8956e79b878cc6
- **Endpoint**: /api/v1/chats_openai/{chat_id}/chat/completions
- **Model Selection**: Via payload parameter
