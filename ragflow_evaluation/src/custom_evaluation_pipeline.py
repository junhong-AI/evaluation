"""
Evaluation pipeline for custom RAG with Ollama models
"""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

from custom_rag_client import CustomRAGClient
from config import RAGAS_EVALUATOR_MODEL, RESULTS_DIR
from utils import save_results


class CustomEvaluationPipeline:
    """Pipeline for evaluating custom RAG systems with Ollama models"""

    def __init__(
        self,
        rag_client: CustomRAGClient,
        evaluator_model: str = None,
        evaluator_api_base: Optional[str] = None
    ):
        """
        Initialize evaluation pipeline

        Args:
            rag_client: Initialized CustomRAGClient
            evaluator_model: LLM model for Ragas evaluation (default: from config)
            evaluator_api_base: Optional API base URL for evaluator (for using Ollama for evaluation too)
        """
        self.client = rag_client
        self.evaluator_model = evaluator_model or RAGAS_EVALUATOR_MODEL

        # Initialize Ragas evaluator LLM
        # Use OpenAI by default, but can be configured to use Ollama via openai_api_base
        if evaluator_api_base:
            # Use Ollama or custom endpoint for evaluation
            evaluator_llm = ChatOpenAI(
                model=self.evaluator_model,
                temperature=0,
                openai_api_base=evaluator_api_base,
                openai_api_key="ollama"  # Dummy key for Ollama
            )
        else:
            # Use OpenAI for evaluation (default)
            evaluator_llm = ChatOpenAI(model=self.evaluator_model, temperature=0)

        self.evaluator_llm_wrapper = LangchainLLMWrapper(evaluator_llm)

        # Initialize metrics
        self.metrics = [
            ContextPrecision(llm=self.evaluator_llm_wrapper),
            ContextRecall(llm=self.evaluator_llm_wrapper),
            Faithfulness(llm=self.evaluator_llm_wrapper),
            AnswerRelevancy(llm=self.evaluator_llm_wrapper)
        ]

    def run_evaluation(
        self,
        test_dataset: List[Dict],
        model_name: str = "custom_rag",
        verbose: bool = True,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run evaluation on test dataset

        Args:
            test_dataset: List of test cases with user_input, reference, reference_contexts
            model_name: Name identifier for this evaluation run
            verbose: Whether to print progress
            temperature: Temperature for generation

        Returns:
            Dict containing:
                - model_name: Model identifier
                - ragas_results: Ragas evaluation results
                - eval_data: Collected evaluation data
                - timestamp: Evaluation timestamp
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating: Custom RAG with Ollama")
            print(f"Generator: {self.client.ollama_model}")
            print(f"Embeddings: {self.client.embedding_model}")
            print(f"Evaluator: {self.evaluator_model}")
            print(f"{'='*70}\n")

        # Collect evaluation data
        eval_data = []
        failed_queries = []

        for i, test_case in enumerate(test_dataset):
            if verbose:
                print(f"[{i+1}/{len(test_dataset)}] Querying: {test_case['user_input'][:60]}...")

            # Query custom RAG
            result = self.client.query(
                question=test_case['user_input'],
                temperature=temperature
            )

            if result['success']:
                # Collect data for Ragas
                eval_data.append({
                    'user_input': test_case['user_input'],
                    'retrieved_contexts': result['contexts'],
                    'response': result['answer'],
                    'reference': test_case['reference'],
                    'reference_contexts': test_case['reference_contexts']
                })

                if verbose:
                    answer_preview = result['answer'][:80] if result['answer'] else "No answer"
                    print(f"  ✓ Answer: {answer_preview}...")
                    print(f"  ✓ Retrieved {len(result['contexts'])} contexts")
            else:
                failed_queries.append({
                    'question': test_case['user_input'],
                    'error': result['error']
                })
                if verbose:
                    print(f"  ✗ Failed: {result['error']}")

            # Small delay to avoid overwhelming the system
            time.sleep(0.5)

        # Check if we have enough successful queries
        if len(eval_data) == 0:
            print(f"\n⚠️  All queries failed for {model_name}")
            return {
                'model_name': model_name,
                'ragas_results': None,
                'eval_data': eval_data,
                'failed_queries': failed_queries,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }

        if verbose:
            print(f"\n✓ Collected {len(eval_data)} successful queries")
            if failed_queries:
                print(f"⚠️  {len(failed_queries)} queries failed")

        # Run Ragas evaluation
        if verbose:
            print(f"\nRunning Ragas evaluation with {self.evaluator_model}...")

        try:
            dataset = EvaluationDataset.from_list(eval_data)
            ragas_results = evaluate(dataset, metrics=self.metrics)

            if verbose:
                print(f"\n{'='*70}")
                print(f"RESULTS for {model_name}")
                print(f"{'='*70}")
                print(ragas_results.to_pandas())
                print(f"{'='*70}\n")

            return {
                'model_name': model_name,
                'ragas_results': ragas_results,
                'eval_data': eval_data,
                'failed_queries': failed_queries,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }

        except Exception as e:
            print(f"\n✗ Ragas evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'ragas_results': None,
                'eval_data': eval_data,
                'failed_queries': failed_queries,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }

    def run_multi_model_evaluation(
        self,
        test_dataset: List[Dict],
        ollama_models: List[str],
        save_dir: str = None,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run evaluation across multiple Ollama models

        Args:
            test_dataset: List of test cases
            ollama_models: List of Ollama model names to test
            save_dir: Directory to save results (default: auto-generated)
            temperature: Temperature for generation

        Returns:
            Dict with results for each model
        """
        # Create results directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(RESULTS_DIR) / f"custom_eval_{timestamp}"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Starting multi-model custom RAG evaluation")
        print(f"📊 Ollama models to evaluate: {len(ollama_models)}")
        print(f"📝 Test questions: {len(test_dataset)}")
        print(f"💾 Results directory: {save_dir}\n")

        all_results = {}

        for model_name in ollama_models:
            print(f"\n{'='*70}")
            print(f"Switching to model: {model_name}")
            print(f"{'='*70}")

            # Update client's model
            self.client.ollama_model = model_name

            # Test connection
            if not self.client.test_connection():
                print(f"⚠️  Warning: Could not verify model {model_name}")

            # Run evaluation
            result = self.run_evaluation(
                test_dataset=test_dataset,
                model_name=model_name,
                verbose=True,
                temperature=temperature
            )

            all_results[model_name] = result

            # Save individual model results
            if result['success'] and result['ragas_results'] is not None:
                save_results(
                    results=result['ragas_results'],
                    model_name=model_name,
                    output_dir=str(save_dir)
                )

        print(f"\n✅ Evaluation complete! Results saved to: {save_dir}")
        return all_results