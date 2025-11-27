# Changes Summary: Custom RAG Implementation

## Overview

Added a customized RAG system that uses **Ollama models** for generation and **OpenAI embeddings** for retrieval, as an alternative to the RAGFlow-based approach.

## What Was Added

### 1. Core Components

#### [custom_rag_client.py](src/custom_rag_client.py)
A standalone RAG client that:
- Uses OpenAI embeddings (text-embedding-3-small or others) for document encoding
- Performs in-memory vector similarity search (cosine similarity)
- Calls Ollama API for text generation
- Supports flexible document loading (text files, JSON, directories)
- Implements basic RAG pipeline: retrieve → format → generate

**Key Features:**
- `add_documents()` - Add documents manually
- `load_documents()` - Load from files/directories
- `query()` - Query the RAG system
- `test_connection()` - Verify Ollama is running

#### [custom_evaluation_pipeline.py](src/custom_evaluation_pipeline.py)
Evaluation pipeline adapted for the custom RAG:
- Uses Ragas metrics (context precision, recall, faithfulness, answer relevancy)
- Supports single model and multi-model evaluation
- Compatible with existing test datasets
- Same evaluation framework as RAGFlow pipeline for fair comparison

**Key Features:**
- `run_evaluation()` - Evaluate a single model
- `run_multi_model_evaluation()` - Compare multiple Ollama models

### 2. Documentation

#### [CUSTOM_RAG_README.md](CUSTOM_RAG_README.md)
Comprehensive documentation covering:
- Architecture and design
- Prerequisites and installation
- Configuration options
- Document loading strategies
- Evaluation workflow
- Troubleshooting guide
- Advantages/disadvantages vs RAGFlow

#### [QUICKSTART_CUSTOM_RAG.md](QUICKSTART_CUSTOM_RAG.md)
Quick start guide with:
- Step-by-step setup (5 minutes)
- Common issues and solutions
- Recommended settings for different use cases
- Example workflows
- Performance expectations

### 3. Jupyter Notebook

#### [01_single_model_test_custom.ipynb](notebooks/01_single_model_test_custom.ipynb)
Interactive notebook demonstrating:
- Setup and configuration
- Ollama model selection
- Document loading
- RAG system testing
- Full evaluation workflow
- Results visualization
- Multi-model comparison (optional)

## Key Differences: RAGFlow vs Custom RAG

| Aspect | RAGFlow | Custom RAG |
|--------|---------|------------|
| **Architecture** | Server-based, external service | Pure Python, local |
| **Setup** | Requires RAGFlow server | Just Python + Ollama |
| **Generation** | Uses configured LLM API | Uses local Ollama models |
| **Embeddings** | RAGFlow-managed | OpenAI API (configurable) |
| **Retrieval** | RAGFlow's indexing | In-memory cosine similarity |
| **Control** | Limited to RAGFlow APIs | Full pipeline control |
| **Cost** | Depends on APIs used | Only embedding API costs |
| **Scalability** | Production-ready | Good for testing/research |

## How to Use

### Quick Start

1. **Install Ollama:**
   ```bash
   # Download from https://ollama.ai
   ollama pull llama3.1:8b
   ```

2. **Set up environment:**
   ```bash
   # Create .env with:
   OPENAI_API_KEY=your_key_here
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook
   # Navigate to: notebooks/01_single_model_test_custom.ipynb
   ```

4. **Follow the cells** to test and evaluate

### Python Script Usage

```python
from custom_rag_client import CustomRAGClient
from custom_evaluation_pipeline import CustomEvaluationPipeline

# Initialize
client = CustomRAGClient(
    ollama_model='llama3.1:8b',
    embedding_model='text-embedding-3-small'
)

# Add documents
client.add_documents(["Doc 1", "Doc 2", "Doc 3"])

# Query
result = client.query("What is this about?")
print(result['answer'])

# Evaluate
pipeline = CustomEvaluationPipeline(rag_client=client)
results = pipeline.run_evaluation(test_dataset)
```

## Benefits

1. **Full Control**: Customize every part of the pipeline
2. **Cost-Effective**: Local models = no API costs for generation
3. **Easy Experimentation**: Switch models with one line
4. **Transparent**: See exactly what's happening
5. **Educational**: Learn how RAG works under the hood
6. **Flexible**: Easy to extend or modify

## Use Cases

### When to Use Custom RAG

- **Research & Development**: Testing different approaches
- **Cost Optimization**: Minimize API costs
- **Learning**: Understanding RAG internals
- **Prototyping**: Quick experiments with different models
- **Privacy**: Keep generation local

### When to Use RAGFlow

- **Production Deployment**: Ready-made infrastructure
- **Complex Pipelines**: Advanced retrieval strategies
- **UI Required**: Need a web interface
- **Team Collaboration**: Shared RAG system

## Integration with Existing Code

The custom RAG integrates seamlessly:

- Uses same **Ragas evaluation** framework
- Compatible with existing **test datasets** (JSONL format)
- Same **metrics** (context precision, recall, faithfulness, answer relevancy)
- Same **output format** for results
- Can **compare directly** with RAGFlow results

## Available Ollama Models

Popular models you can test:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| phi3:mini | ~2GB | ⚡⚡⚡ | ⭐⭐ | Quick tests |
| mistral:7b | ~4GB | ⚡⚡ | ⭐⭐⭐ | Balanced |
| llama3.1:8b | ~4.7GB | ⚡⚡ | ⭐⭐⭐⭐ | Recommended |
| gemma2:9b | ~5GB | ⚡⚡ | ⭐⭐⭐⭐ | High quality |
| llama3.1:70b | ~40GB | ⚡ | ⭐⭐⭐⭐⭐ | Best quality |

## Next Steps

1. **Try the notebook**: Run `01_single_model_test_custom.ipynb`
2. **Load your documents**: Replace sample data with your corpus
3. **Compare models**: Test multiple Ollama models
4. **Tune parameters**: Optimize for your use case
5. **Compare with RAGFlow**: Run both and compare results

## Technical Notes

### Embedding Strategy
- Uses OpenAI's embedding API (high quality, cost-effective)
- Embeddings created once, cached in memory
- Cosine similarity for retrieval

### Generation Strategy
- Ollama API via HTTP requests
- Non-streaming for evaluation consistency
- Customizable prompt templates

### Evaluation Strategy
- Same Ragas metrics as RAGFlow evaluation
- OpenAI GPT-4o-mini as evaluator by default
- Per-question and aggregate metrics

## Files Added

```
ragflow_evaluation/
├── src/
│   ├── custom_rag_client.py              # Core RAG client
│   └── custom_evaluation_pipeline.py     # Evaluation pipeline
├── notebooks/
│   └── 01_single_model_test_custom.ipynb # Demo notebook
├── CUSTOM_RAG_README.md                  # Full documentation
├── QUICKSTART_CUSTOM_RAG.md              # Quick start guide
└── CHANGES_SUMMARY.md                    # This file
```

## Dependencies

All required packages are already in `requirements.txt`:
- `openai` - For embeddings
- `requests` - For Ollama API
- `numpy` - For vector operations
- `ragas` - For evaluation
- `langchain-openai` - For Ragas evaluator

No new dependencies needed!

## Troubleshooting

See [QUICKSTART_CUSTOM_RAG.md](QUICKSTART_CUSTOM_RAG.md) for common issues and solutions.

## Support

For questions:
1. Check the documentation files
2. Review the notebook examples
3. Consult Ollama docs: https://ollama.ai/docs
4. Consult Ragas docs: https://docs.ragas.io