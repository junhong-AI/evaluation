# Quick Start: Custom RAG with Ollama

Get started with the custom RAG system in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Ollama installed ([download here](https://ollama.ai))
- [ ] OpenAI API key (for embeddings)

## Step 1: Install Ollama

### macOS/Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
Download from [https://ollama.ai/download](https://ollama.ai/download)

### Verify Installation
```bash
ollama --version
```

## Step 2: Pull Ollama Models

```bash
# Start with a small, fast model
ollama pull llama3.1:8b

# Optional: Pull other models to compare
ollama pull mistral:7b
ollama pull phi3:mini
```

## Step 3: Install Python Dependencies

```bash
# From the ragflow_evaluation directory
pip install -r requirements.txt
```

## Step 4: Set Up Environment Variables

Create a `.env` file in the `ragflow_evaluation` directory:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

## Step 5: Open the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/01_single_model_test_custom.ipynb
```

## Step 6: Run the Notebook

Follow the cells in order:

1. **Setup** - Imports and configuration ✓
2. **Configuration** - Set your Ollama model preference
3. **Initialize Client** - Create the RAG client
4. **Load Documents** - Add your knowledge base
5. **Test** - Try a query to make sure it works
6. **Load Dataset** - Load test questions
7. **Run Evaluation** - Evaluate performance with Ragas
8. **Analyze** - View metrics and results

## Quick Test (Python Script)

Want to test without Jupyter? Run this:

```python
# test_custom_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

from custom_rag_client import CustomRAGClient

# Initialize
client = CustomRAGClient(
    ollama_model='llama3.1:8b',
    embedding_model='text-embedding-3-small'
)

# Add sample documents
documents = [
    "AccessMatrix is an identity and access management solution.",
    "It supports SAML, OAuth, and OpenID Connect protocols.",
    "The system provides multi-factor authentication capabilities.",
]
client.add_documents(documents)

# Test query
result = client.query("What is AccessMatrix?")
print("Answer:", result['answer'])
print("Retrieved", len(result['contexts']), "contexts")
```

Run it:
```bash
python test_custom_rag.py
```

## Common Issues

### Issue: "Connection refused" when querying Ollama

**Solution**: Make sure Ollama is running
```bash
# Check if running
curl http://localhost:11434/api/tags

# If not running, start it (should auto-start on install)
ollama serve
```

### Issue: "Model not found"

**Solution**: Pull the model first
```bash
ollama pull llama3.1:8b
ollama list  # Verify it's there
```

### Issue: "OpenAI API key not found"

**Solution**: Check your .env file
```bash
# Make sure .env exists and contains:
OPENAI_API_KEY=sk-...
```

### Issue: Slow generation

**Solution**: Use a smaller model
```python
# Instead of llama3.1:70b, use:
ollama_model='phi3:mini'  # Much faster!
```

## What's Next?

1. **Load Your Documents**: Replace sample documents with your actual corpus
   ```python
   client.load_documents('./path/to/your/documents')
   ```

2. **Try Different Models**: Compare model performance
   ```python
   models = ['llama3.1:8b', 'mistral:7b', 'phi3:mini']
   results = pipeline.run_multi_model_evaluation(dataset, models)
   ```

3. **Tune Parameters**: Adjust for your use case
   ```python
   CONFIG = {
       'top_k': 10,        # Retrieve more documents
       'temperature': 0.3,  # More creative answers
   }
   ```

4. **Run Full Evaluation**: Test with your QA dataset
   ```python
   dataset = load_dataset_from_jsonl('your_test_data.jsonl')
   result = pipeline.run_evaluation(dataset)
   ```

## Recommended Settings

### For Speed (Testing/Development)
```python
CONFIG = {
    'ollama_model': 'phi3:mini',
    'embedding_model': 'text-embedding-3-small',
    'top_k': 3,
    'temperature': 0.1,
}
```

### For Quality (Production)
```python
CONFIG = {
    'ollama_model': 'llama3.1:70b',
    'embedding_model': 'text-embedding-3-large',
    'top_k': 5,
    'temperature': 0.1,
}
```

### For Balance
```python
CONFIG = {
    'ollama_model': 'llama3.1:8b',
    'embedding_model': 'text-embedding-3-small',
    'top_k': 5,
    'temperature': 0.1,
}
```

## Performance Expectations

| Model | Speed | Quality | RAM Required |
|-------|-------|---------|--------------|
| phi3:mini | ⚡⚡⚡ | ⭐⭐ | ~4GB |
| mistral:7b | ⚡⚡ | ⭐⭐⭐ | ~8GB |
| llama3.1:8b | ⚡⚡ | ⭐⭐⭐⭐ | ~8GB |
| llama3.1:70b | ⚡ | ⭐⭐⭐⭐⭐ | ~40GB |

## Example Workflow

```python
from custom_rag_client import CustomRAGClient
from custom_evaluation_pipeline import CustomEvaluationPipeline
from utils import load_dataset_from_jsonl

# 1. Setup
client = CustomRAGClient(
    ollama_model='llama3.1:8b',
    embedding_model='text-embedding-3-small',
    top_k=5
)

# 2. Load your documents
client.load_documents('./documents')
print(f"Loaded {len(client.documents)} documents")

# 3. Quick test
test = client.query("What is this about?")
print("Answer:", test['answer'])

# 4. Full evaluation
pipeline = CustomEvaluationPipeline(rag_client=client)
dataset = load_dataset_from_jsonl('test_data.jsonl')
results = pipeline.run_evaluation(dataset, verbose=True)

# 5. View metrics
print(results['ragas_results'].to_pandas().mean())
```

## Getting Help

- **Ollama Issues**: Check [Ollama docs](https://ollama.ai/docs)
- **Ragas Issues**: Check [Ragas docs](https://docs.ragas.io)
- **Code Issues**: Review `CUSTOM_RAG_README.md` for detailed docs

## Comparison: RAGFlow vs Custom RAG

| Feature | RAGFlow | Custom RAG |
|---------|---------|------------|
| Setup | Complex (server required) | Simple (Python only) |
| Control | Limited | Full control |
| UI | Web interface | Notebook/CLI |
| Cost | API calls | Local (except embeddings) |
| Flexibility | Fixed pipeline | Fully customizable |
| Best For | Production deployment | Research, testing, learning |

Both are valid approaches - use what fits your needs!