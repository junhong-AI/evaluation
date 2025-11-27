# Quick Start Guide

## 📋 What Was Built

A complete RAG evaluation pipeline that:
- Tests 4 embedding models (OpenAI, Mistral, Snowflake, BAAI)
- Uses your 20 Q&A pairs from USO Administration Guide
- Queries RAGFlow API for each test
- Calculates Ragas metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy)
- Generates comparison reports and visualizations

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
cd C:\Users\junhongs\Desktop\capstone\evaluation\ragflow_evaluation
pip install -r requirements.txt
```

### Step 2: Verify Setup

Open and run: `notebooks/00_setup_check.ipynb`

This will verify:
- ✅ All dependencies installed
- ✅ Environment variables configured
- ✅ RAGFlow connection works
- ✅ OpenAI API accessible
- ✅ Test dataset loaded (20 questions)

### Step 3: Run Single Model Test (Baseline)

Open and run: `notebooks/01_single_model_test.ipynb`

This tests with **text-embedding-3-small** to:
- Validate the evaluation pipeline
- Identify any issues before scaling
- Establish baseline metrics

**Expected Runtime:** ~2-3 minutes (20 queries)

### Step 4: Run Multi-Model Evaluation

Open and run: `notebooks/02_multi_model_eval.ipynb`

This evaluates all 4 models:
1. OpenAI text-embedding-3-small
2. Mistral mistral-embed
3. Snowflake snowflake-arctic-embed-l
4. BAAI bge-large-en-v1.5

**Expected Runtime:** ~10-15 minutes (80 total queries)

**Outputs:**
- `results/multi_model_{timestamp}/comparison_table.csv`
- `results/multi_model_{timestamp}/comparison_chart.html`
- `results/multi_model_{timestamp}/radar_chart.html`
- `results/multi_model_{timestamp}/EVALUATION_REPORT.md`
- Individual model results CSVs

## 📊 Understanding Results

### Metrics Explained

1. **Context Precision** (0-1)
   - Are retrieved chunks relevant to the question?
   - Higher = better retrieval quality

2. **Context Recall** (0-1)
   - Did we retrieve the reference contexts?
   - Higher = fewer missed relevant documents

3. **Faithfulness** (0-1)
   - Is the answer grounded in retrieved context?
   - Higher = less hallucination

4. **Answer Relevancy** (0-1)
   - Is the answer relevant to the question?
   - Higher = more on-topic responses

### Example Interpretation

```
Model: OpenAI text-embedding-3-small
├── Context Precision: 0.85  → 85% of retrieved chunks are relevant
├── Context Recall: 0.72     → Found 72% of ideal contexts
├── Faithfulness: 0.91       → 91% of answer is supported by context
└── Answer Relevancy: 0.88   → 88% relevance to question
```

## 🔧 Troubleshooting

### Issue: RAGFlow Connection Failed

**Symptoms:**
- `00_setup_check.ipynb` shows connection error
- "Request failed" messages

**Solutions:**
1. Verify RAGFlow is running: `http://172.25.184.41:80`
2. Check API key in `.env` file
3. Verify chat_id is correct: `c78456e48e0c11f0be8956e79b878cc6`
4. Test with curl:
   ```bash
   curl -X POST http://172.25.184.41:80/api/v1/chats_openai/c78456e48e0c11f0be8956e79b878cc6/chat/completions \
     -H "Authorization: Bearer ragflow-NlZTM0MWZjOGRmYzExZjA5NTU1NjZlNz" \
     -H "Content-Type: application/json" \
     -d '{"model":"default","messages":[{"role":"user","content":"test"}],"stream":false}'
   ```

### Issue: OpenAI API Error

**Symptoms:**
- Ragas evaluation fails
- "OpenAI API key invalid" error

**Solutions:**
1. Check API key in `.env` file
2. Verify key has credits: https://platform.openai.com/usage
3. Test with simple query in `00_setup_check.ipynb`

### Issue: Model Not Found in RAGFlow

**Symptoms:**
- Queries succeed but return empty contexts
- "Model not configured" error

**Solutions:**
1. Verify embedding models are configured in RAGFlow
2. Check model names match RAGFlow's configuration
3. Update `ragflow_model_name` in `src/config.py` to match RAGFlow

### Issue: Ragas Evaluation Slow

**Expected:**
- Ragas uses GPT-4o-mini for evaluation
- Each question requires 4 metric calculations
- 20 questions × 4 metrics = 80 LLM calls per model
- ~10-15 seconds per question

**To Speed Up:**
- Use fewer test questions (edit dataset)
- Disable some metrics in `src/evaluation_pipeline.py`

## 🎯 Next Steps

### After First Run

1. **Review Results:**
   - Open `EVALUATION_REPORT.md` for summary
   - Check interactive charts in HTML files
   - Review per-model CSV files

2. **Identify Best Model:**
   - Look at "Best Overall Model" in report
   - Consider trade-offs (cost vs performance)

3. **Analyze Failures:**
   - Check which questions failed
   - Review error messages
   - Adjust prompts or retrieval if needed

### Adapt to Custom RAG

To use with your own RAG system (not RAGFlow):

1. **Update `src/ragflow_client.py`:**
   - Replace `query()` method with your RAG API
   - Ensure it returns: `{"answer": str, "contexts": [str]}`

2. **Update `src/config.py`:**
   - Add your RAG endpoint configuration
   - Update model names to match your system

3. **Run evaluation:**
   - Same notebooks will work
   - Pipeline is provider-agnostic

## 📁 Project Structure

```
ragflow_evaluation/
├── data/
│   └── test_qa_pairs.jsonl          ← Your 20 Q&A pairs (READY)
├── notebooks/
│   ├── 00_setup_check.ipynb         ← Run this first
│   ├── 01_single_model_test.ipynb   ← Then this (baseline)
│   └── 02_multi_model_eval.ipynb    ← Finally this (full eval)
├── src/
│   ├── config.py                    ← Model configurations
│   ├── ragflow_client.py            ← RAGFlow API wrapper
│   ├── evaluation_pipeline.py       ← Core evaluation logic
│   └── utils.py                     ← Helper functions
├── results/                         ← Evaluation outputs go here
├── .env                             ← API keys (CONFIGURED)
└── requirements.txt                 ← Dependencies
```

## 💡 Tips

1. **Start Small:**
   - Test with 5 questions first (edit dataset)
   - Verify everything works
   - Then run full 20

2. **Monitor Costs:**
   - Each evaluation run costs ~$0.10-0.50 (depends on answer length)
   - 4 models × 20 questions × $0.01 ≈ $0.80 per full run

3. **Save Results:**
   - Results are timestamped automatically
   - Compare multiple runs over time
   - Track improvements

4. **Customize:**
   - Add more embedding models in `src/config.py`
   - Adjust Ragas metrics in `src/evaluation_pipeline.py`
   - Modify test dataset as needed

## ❓ Questions?

Common questions:

**Q: Can I use different test questions?**
A: Yes! Update `data/test_qa_pairs.jsonl` with format:
```json
{"user_input": "question", "reference": "answer", "reference_contexts": ["context1"]}
```

**Q: How do I add more embedding models?**
A: Edit `src/config.py` → `EMBEDDING_MODELS` list. Add new dict with model config.

**Q: Can I use local LLMs?**
A: Yes! Update `src/evaluation_pipeline.py` to use Ollama instead of OpenAI for Ragas evaluator.

**Q: Results directory is huge?**
A: Results are timestamped per run. Delete old runs to save space.

## 🎉 You're Ready!

Run `notebooks/00_setup_check.ipynb` to begin!
