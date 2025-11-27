"""
Utility functions for data loading and processing
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import csv


def load_test_dataset_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load test dataset from CSV file and convert to Ragas format

    Args:
        csv_path: Path to CSV file with columns: query, ground_truth, contexts_joined

    Returns:
        List of dicts with keys: user_input, reference, reference_contexts
    """
    df = pd.read_csv(csv_path, encoding='utf-8')

    dataset = []
    for _, row in df.iterrows():
        # Split contexts_joined back into list
        contexts_text = row['contexts_joined']
        # Try to parse as list, otherwise split by double newline
        if isinstance(contexts_text, str):
            # Check if it looks like JSON array
            if contexts_text.strip().startswith('['):
                try:
                    contexts = json.loads(contexts_text)
                except:
                    contexts = [contexts_text]
            else:
                # Split by double newline or keep as single context
                contexts = [ctx.strip() for ctx in contexts_text.split('\n\n') if ctx.strip()]
                if not contexts:
                    contexts = [contexts_text]
        else:
            contexts = [str(contexts_text)]

        dataset.append({
            "user_input": row['query'],
            "reference": row['ground_truth'],
            "reference_contexts": contexts
        })

    return dataset


def save_dataset_to_jsonl(dataset: List[Dict], output_path: str):
    """
    Save dataset to JSONL format

    Args:
        dataset: List of test cases
        output_path: Path to save JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load dataset from JSONL file

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of test cases
    """
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def validate_dataset(dataset: List[Dict]) -> bool:
    """
    Validate that dataset has required fields

    Args:
        dataset: List of test cases

    Returns:
        True if valid, False otherwise
    """
    required_fields = {'user_input', 'reference', 'reference_contexts'}

    for i, item in enumerate(dataset):
        missing = required_fields - set(item.keys())
        if missing:
            print(f"Item {i} missing fields: {missing}")
            return False

        if not isinstance(item['reference_contexts'], list):
            print(f"Item {i}: reference_contexts must be a list")
            return False

    return True


def format_results_table(results_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Format Ragas evaluation results as a pandas DataFrame

    Args:
        results_dict: Dictionary of model_name -> ragas results

    Returns:
        DataFrame with comparison table
    """
    rows = []
    for model_name, results in results_dict.items():
        # Extract metrics from Ragas results
        # results is typically a pandas DataFrame or dict
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            metrics = df.mean().to_dict()
        elif isinstance(results, dict):
            metrics = results
        else:
            continue

        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def save_results(
    results: Any,
    model_name: str,
    output_dir: str,
    include_raw: bool = True
):
    """
    Save evaluation results to disk

    Args:
        results: Ragas evaluation results
        model_name: Name of the model
        output_dir: Directory to save results
        include_raw: Whether to include raw response data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        csv_path = output_path / f"{model_name}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    # Save summary statistics
    if hasattr(results, 'to_pandas'):
        summary = df.mean().to_frame(name='mean')
        summary['std'] = df.std()
        summary_path = output_path / f"{model_name}_summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")


def print_dataset_summary(dataset: List[Dict]):
    """Print summary statistics of the dataset"""
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total Questions: {len(dataset)}")

    # Question length stats
    q_lengths = [len(item['user_input']) for item in dataset]
    print(f"\nQuestion Length:")
    print(f"  Min: {min(q_lengths)} chars")
    print(f"  Max: {max(q_lengths)} chars")
    print(f"  Avg: {sum(q_lengths)/len(q_lengths):.1f} chars")

    # Ground truth length stats
    gt_lengths = [len(item['reference']) for item in dataset]
    print(f"\nGround Truth Length:")
    print(f"  Min: {min(gt_lengths)} chars")
    print(f"  Max: {max(gt_lengths)} chars")
    print(f"  Avg: {sum(gt_lengths)/len(gt_lengths):.1f} chars")

    # Reference contexts stats
    ctx_counts = [len(item['reference_contexts']) for item in dataset]
    print(f"\nReference Contexts:")
    print(f"  Min: {min(ctx_counts)} contexts")
    print(f"  Max: {max(ctx_counts)} contexts")
    print(f"  Avg: {sum(ctx_counts)/len(ctx_counts):.1f} contexts")

    print(f"{'='*60}\n")
