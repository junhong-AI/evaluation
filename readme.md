## Getting started 


 Objective 

List of RAG evaluation frameworks 

RAGAS : opensoure framework measures Faithfullness, relevance, context precision, can synthetically generate QA pairs from documents

Deepeval : open source build for CI/CD. use pytest integration. measures RAG triad plus toxicity, Bias, also has synthetic data generation 

Trulens : open source framework, focus on RAG triad. 

Llamaindex : RAG framework with built in evaluation. 


# For platforms (monitoring and production)
LangSmith
Arize AI (Phoenix)
Braintrust


## Generate a RAGAS synthetic testset from the USO Admin Guide

This repo includes a notebook `dataLoader_ragas.ipynb` that generates a synthetic RAG evaluation dataset from the PDF at `material/USO-Administration-Guide-6.0.1-GA (AI).pdf`.

Steps on Windows (PowerShell):

1. Open the notebook `dataLoader_ragas.ipynb` and select a Python kernel.
2. Install required packages in the notebook kernel:

```powershell
# In a notebook cell, run:
%pip install ragas langchain-community langchain-openai openai pypdf pandas tqdm
```

3. Set your OpenAI key for generation (choose a cost-effective model like gpt-4o-mini):

```powershell
$env:OPENAI_API_KEY = "sk-..."
# Optional: override the default model and testset size
$env:RAGAS_GEN_MODEL = "gpt-4o-mini"
$env:RAGAS_TESTSET_SIZE = "24"
```

4. Run the notebook cells top to bottom. The script will:
	 - Load and chunk the PDF into LangChain Documents
	 - Use Ragas TestsetGenerator to synthesize queries, ground-truth answers, and reference contexts
	 - Save outputs to:
		 - `dataset/WikiEval/synthetic_usoadmin_ragas.jsonl`
		 - `dataset/WikiEval/synthetic_usoadmin_ragas.csv` (contexts flattened)

If you prefer Azure OpenAI or other providers, wrap your provider with LangChain and pass it via `LangchainLLMWrapper` per Ragas docs.

Notes:
- Generation uses LLM calls and may incur cost. Start with a small `RAGAS_TESTSET_SIZE` (e.g., 10–20) to validate.
- You can re-run with a larger size once satisfied.




# todo
Step	Default Ragas behavior	Customizable?
Question generation	LLM creates QA pairs from docs	✅ prompt, model, #questions
Answer generation	LLM answers using the doc	✅ prompt, model
Context retrieval	Uses your retriever or defaults	✅ custom retriever, top-k
Output dataset	JSON / Hugging Face Dataset	✅ can filter, edit, or merge


For synthetic data generation:
- customized prompts
- model choice (open ai models)
- num of questions
- context retrieval strategy
- custom retriever (e.g., vector db)

1. Prompt



domain_prompt = """
You are generating synthetic questions to evaluate a Retrieval-Augmented Generation (RAG) chatbot
for a cybersecurity company that builds Access Management (AM).

Each question should reflect realistic user inquiries or troubleshooting requests.

Write 3–5 diverse, specific questions per document that:
- Focus on authentication, identity federation, authorization, or security configuration.
- Mention relevant standards when appropriate (SAML, OAuth2, OpenID Connect, etc.).
- Include context like “internal admin,” “audit,” or “security policy.”
- Avoid trivia or generic textbook questions.
- Use professional, concise phrasing typical of IT/security engineers.

Only use information from the provided document to ensure the question is grounded in it.
"""

dataset = generator.generate(
    documents=Dataset.from_list(docs),
    num_questions=4,
    question_prompt_template=domain_prompt,
)

answer_prompt = """
Answer the question using only the content in the provided document.
Provide a concise, accurate, and technically detailed response, as if explaining to a
cybersecurity engineer configuring Access Manager or SSO.
Do not include general definitions or external knowledge.
"""

dataset = generator.generate(
    documents=Dataset.from_list(docs),
    num_questions=4,
    question_prompt_template=domain_prompt,
    answer_prompt_template=answer_prompt,
)

