# Custom RAG with Ollama Models

This setup provides a customized RAG (Retrieval-Augmented Generation) system using:
- **OpenAI Embeddings** for document retrieval (text-embedding-3-small or other models)
- **Ollama Models** for local LLM generation
- **Ragas** for evaluation metrics

## Why Use Custom RAG?

1. **Full Control**: Complete control over the retrieval and generation pipeline
2. **Cost-Effective**: Use local Ollama models instead of expensive API calls
3. **Flexible**: Easy to swap embedding models, generators, or add custom logic
4. **Privacy**: Documents and generation stay local (except for embeddings)
5. **Experimentation**: Test different model combinations easily

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

```bash
# After installation, pull models you want to use:
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:mini
ollama pull gemma2:9b

# Verify Ollama is running:
ollama list
```

### 2. Install Python Dependencies

```bash
pip install numpy openai requests python-dotenv
pip install ragas langchain-openai pandas plotly
```

### 3. Set Up Environment Variables

Create a `.env` file with your OpenAI API key (for embeddings and evaluation):

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

### Option 1: Use the Jupyter Notebook

1. Open the notebook:
   ```
   notebooks/01_single_model_test_custom.ipynb
   ```

2. Follow the cells step-by-step:
   - Configure Ollama and embedding models
   - Load your documents
   - Test the RAG system
   - Run evaluation
   - Visualize results

### Option 2: Use Python Scripts

```python
from custom_rag_client import CustomRAGClient
from custom_evaluation_pipeline import CustomEvaluationPipeline

# Initialize RAG client
client = CustomRAGClient(
    ollama_base_url='http://localhost:11434',
    ollama_model='llama3.1:8b',
    embedding_model='text-embedding-3-small',
    top_k=5
)

# Add documents
documents = [
    "Your document text here...",
    "Another document...",
]
client.add_documents(documents)

# Query the system
result = client.query("What is this about?")
print(result['answer'])
```

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         v
┌─────────────────────────────────┐
│  OpenAI Embedding API           │
│  (text-embedding-3-small)       │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  Vector Similarity Search       │
│  (Cosine Similarity)            │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  Top-K Document Retrieval       │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  Prompt Construction            │
│  (Context + Question)           │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  Ollama Local LLM               │
│  (llama3.1, mistral, etc.)      │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  Generated Answer               │
└─────────────────────────────────┘
```

## Features

### CustomRAGClient

- **Flexible Document Loading**: Load from text files, JSON, or directories
- **OpenAI Embeddings**: High-quality embeddings for retrieval
- **Cosine Similarity Search**: Fast and accurate retrieval
- **Local LLM Generation**: Use any Ollama model
- **Customizable Prompts**: Define your own prompt templates

### CustomEvaluationPipeline

- **Ragas Metrics**: Context precision, context recall, faithfulness, answer relevancy
- **Multi-Model Testing**: Easy comparison across different Ollama models
- **Detailed Results**: Per-question metrics and aggregate statistics
- **Visualization**: Built-in plotting for results analysis

## Configuration Options

### Ollama Models

Popular models you can use:
- `llama3.1:8b` - Meta's Llama 3.1 (8B parameters) - Good balance
- `llama3.1:70b` - Larger model for better quality (requires more RAM)
- `mistral:7b` - Mistral 7B - Fast and efficient
- `mixtral:8x7b` - Mixture of Experts - High quality
- `phi3:mini` - Microsoft's small model - Very fast
- `gemma2:9b` - Google's Gemma 2 - Good performance

### Embedding Models

OpenAI embedding models:
- `text-embedding-3-small` - Fast, cost-effective (1536 dimensions)
- `text-embedding-3-large` - Higher quality (3072 dimensions)
- `text-embedding-ada-002` - Previous generation

### RAG Parameters

```python
CONFIG = {
    'ollama_base_url': 'http://localhost:11434',
    'ollama_model': 'llama3.1:8b',
    'embedding_model': 'text-embedding-3-small',
    'top_k': 5,              # Number of documents to retrieve
    'temperature': 0.1,      # Lower = more deterministic
}
```

## Loading Documents

### From Files

```python
# Load all documents from a directory
client.load_documents('/path/to/documents')

# Load a single text file
client.load_documents('/path/to/document.txt')

# Load a JSON file
client.load_documents('/path/to/documents.json')
```

### Manually

```python
documents = [
    "Document 1 content...",
    "Document 2 content...",
]
client.add_documents(documents)
```

### JSON Format

If using JSON, format should be:
```json
{
  "documents": [
    "Document 1 text...",
    "Document 2 text..."
  ]
}
```

Or simply an array:
```json
[
  "Document 1 text...",
  "Document 2 text..."
]
```

## Evaluation

The system uses Ragas metrics for evaluation:

1. **Context Precision**: How relevant are the retrieved documents?
2. **Context Recall**: Did we retrieve all necessary information?
3. **Faithfulness**: Is the answer grounded in the retrieved context?
4. **Answer Relevancy**: How relevant is the answer to the question?

### Running Evaluation

```python
# Load test dataset
from utils import load_dataset_from_jsonl
dataset = load_dataset_from_jsonl('data/test_qa_pairs.jsonl')

# Initialize pipeline
pipeline = CustomEvaluationPipeline(
    rag_client=client,
    evaluator_model='gpt-4o-mini'
)

# Run evaluation
result = pipeline.run_evaluation(
    test_dataset=dataset,
    model_name='custom_rag',
    verbose=True
)
```

## Comparing Multiple Models

```python
# Test multiple Ollama models
models_to_test = [
    'llama3.1:8b',
    'mistral:7b',
    'phi3:mini',
]

results = pipeline.run_multi_model_evaluation(
    test_dataset=dataset,
    ollama_models=models_to_test
)
```

## Troubleshooting

### Ollama Not Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed (it should auto-start on installation)
# On macOS/Linux, it runs as a service
# On Windows, check if Ollama.exe is running
```

### Model Not Found

```bash
# Pull the model first
ollama pull llama3.1:8b

# List available models
ollama list
```

### Out of Memory

If you get OOM errors:
1. Use smaller models (phi3:mini, mistral:7b)
2. Reduce `top_k` to retrieve fewer documents
3. Close other applications
4. Consider using quantized models

### Slow Generation

1. Use smaller models
2. Reduce `max_tokens` parameter
3. Use a GPU if available (Ollama auto-detects)

## Advantages Over RAGFlow

1. **No External Dependencies**: No need to run RAGFlow server
2. **Simpler Setup**: Just Python + Ollama
3. **Full Control**: Customize every aspect of the pipeline
4. **Cost-Effective**: Local models = no API costs for generation
5. **Easy Model Switching**: Change models with one line of code
6. **Transparent**: You see exactly what's happening

## Disadvantages

1. **Manual Indexing**: You handle document chunking and embedding
2. **No UI**: Command-line/notebook only (vs RAGFlow's web UI)
3. **Basic Retrieval**: Simple cosine similarity (no advanced ranking)
4. **Limited Scale**: In-memory storage (fine for testing, not for production)

## Next Steps

1. **Add Better Chunking**: Implement semantic chunking for better retrieval
2. **Use Vector Databases**: Integrate with ChromaDB, Pinecone, or Weaviate
3. **Add Reranking**: Implement cross-encoder reranking for better results
4. **Experiment with Prompts**: Create custom prompt templates
5. **Try Different Embeddings**: Test with other embedding models
6. **Optimize for Your Use Case**: Tune parameters based on your data

## Example: Complete Workflow

```python
# 1. Initialize
from custom_rag_client import CustomRAGClient
from custom_evaluation_pipeline import CustomEvaluationPipeline

client = CustomRAGClient(
    ollama_model='llama3.1:8b',
    embedding_model='text-embedding-3-small'
)

# 2. Load documents
client.load_documents('./my_documents')

# 3. Test a query
result = client.query("What is AccessMatrix?")
print(result['answer'])

# 4. Run evaluation
pipeline = CustomEvaluationPipeline(rag_client=client)
eval_result = pipeline.run_evaluation(test_dataset=my_dataset)

# 5. Analyze results
print(eval_result['ragas_results'].to_pandas())
```

## Support

For issues or questions:
1. Check Ollama documentation: https://ollama.ai/docs
2. Check Ragas documentation: https://docs.ragas.io
3. Review the notebook for examples
