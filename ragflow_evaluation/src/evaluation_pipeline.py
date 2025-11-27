"""
Core evaluation pipeline for running Ragas evaluation across models
"""
import time
from typing import List, Dict, Any
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

from ragflow_client import RAGFlowClient
from config import RAGAS_EVALUATOR_MODEL, RESULTS_DIR
from utils import save_results


class EvaluationPipeline:
    """Pipeline for evaluating RAG systems with multiple embedding models"""

    def __init__(self, ragflow_client: RAGFlowClient, evaluator_model: str = None):
        """
        Initialize evaluation pipeline

        Args:
            ragflow_client: Initialized RAGFlow client
            evaluator_model: LLM model for Ragas evaluation (default: from config)
        """
        self.client = ragflow_client
        self.evaluator_model = evaluator_model or RAGAS_EVALUATOR_MODEL

        # Initialize Ragas evaluator LLM
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
        model_config: Dict,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation for a single embedding model

        Args:
            test_dataset: List of test cases with user_input, reference, reference_contexts
            model_config: Model configuration dict with 'name', 'ragflow_model_name', etc.
            verbose: Whether to print progress

        Returns:
            Dict containing:
                - model_name: Model identifier
                - ragas_results: Ragas evaluation results
                - eval_data: Collected evaluation data
                - timestamp: Evaluation timestamp
        """
        model_name = model_config['name']
        ragflow_model = model_config['ragflow_model_name']

        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_config['display_name']}")
            print(f"Model: {ragflow_model}")
            print(f"{'='*70}\n")

        # Collect evaluation data
        eval_data = []
        failed_queries = []

        for i, test_case in enumerate(test_dataset):
            if verbose:
                print(f"[{i+1}/{len(test_dataset)}] Querying: {test_case['user_input'][:60]}...")

            # Query RAGFlow
            result = self.client.query(
                question=test_case['user_input'],
                model_name=ragflow_model
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
                    print(f"  ✓ Answer: {result['answer'][:80]}...")
                    print(f"  ✓ Retrieved {len(result['contexts'])} contexts")
            else:
                failed_queries.append({
                    'question': test_case['user_input'],
                    'error': result['error']
                })
                if verbose:
                    print(f"  ✗ Failed: {result['error']}")

            # Small delay to avoid rate limits
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
                print(f"RESULTS for {model_config['display_name']}")
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
        model_configs: List[Dict],
        save_dir: str = None
    ) -> Dict[str, Any]:
        """
        Run evaluation across multiple embedding models

        Args:
            test_dataset: List of test cases
            model_configs: List of model configuration dicts
            save_dir: Directory to save results (default: auto-generated)

        Returns:
            Dict with results for each model
        """
        # Create results directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(RESULTS_DIR) / f"evaluation_{timestamp}"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Starting multi-model evaluation")
        print(f"📊 Models to evaluate: {len(model_configs)}")
        print(f"📝 Test questions: {len(test_dataset)}")
        print(f"💾 Results directory: {save_dir}\n")

        all_results = {}

        for model_config in model_configs:
            result = self.run_evaluation(
                test_dataset=test_dataset,
                model_config=model_config,
                verbose=True
            )

            all_results[model_config['name']] = result

            # Save individual model results
            if result['success'] and result['ragas_results'] is not None:
                save_results(
                    results=result['ragas_results'],
                    model_name=model_config['name'],
                    output_dir=str(save_dir)
                )

        print(f"\n✅ Evaluation complete! Results saved to: {save_dir}")
        return all_results
