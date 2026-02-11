"""
RAG Starter Kit: Retrieval Benchmark

Measures Recall@K and MRR@K on a synthetic eval set. Run AFTER generate_eval_set.py
and ingestion. Use before/after any retrieval change to quantify impact.

Based on concepts from the "Systematically Improving RAG" course (Maven).

Usage:
    python src/benchmark.py
    python src/benchmark.py --eval-file data/eval/questions.jsonl --top-k 5 10 15
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client

from config import MINIMUM_SIMILARITY_THRESHOLD
from retrieval import build_chunk_id, retrieve_documents

load_dotenv()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_recall(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Recall = (relevant items retrieved) / (total relevant items)
    For a single expected chunk: 1 if found in predictions, else 0.
    """
    if not ground_truth:
        return 0.0
    return len([g for g in ground_truth if g in predictions]) / len(ground_truth)


def calculate_mrr(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant item.
    If the correct chunk is at position 1, MRR=1.0; at position 3, MRR=1/3.
    """
    mrr = 0.0
    for label in ground_truth:
        if label in predictions:
            rank = predictions.index(label) + 1
            mrr = max(mrr, 1.0 / rank)
    return mrr


# ---------------------------------------------------------------------------
# Retrieval (shared with app.py via retrieval.py)
# ---------------------------------------------------------------------------


async def retrieve(
    supabase,
    openai_client: AsyncOpenAI,
    question: str,
    top_k: int,
    similarity_threshold: float,
) -> list[str]:
    """
    Run retrieval and return list of chunk_ids (url|chunk_number) in rank order.
    Uses shared retrieval logic so benchmark results reflect production behavior.
    """
    docs = await retrieve_documents(
        supabase,
        openai_client,
        question,
        max_results=top_k,
        similarity_threshold=similarity_threshold,
        fallback_limit=2,
        verbose=False,
    )
    return [build_chunk_id(row) for row in docs]


# ---------------------------------------------------------------------------
# Benchmark Run
# ---------------------------------------------------------------------------


async def run_benchmark(
    eval_path: Path,
    top_k_values: list[int],
    similarity_threshold: float,
) -> dict:
    """Run benchmark and return metrics."""
    if not eval_path.exists():
        print(f"❌ Eval file not found: {eval_path}")
        print("   Run: python src/generate_eval_set.py")
        sys.exit(1)

    eval_items = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    if not eval_items:
        print(f"⚠️ No eval items in {eval_path}")
        sys.exit(1)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not all([supabase_url, supabase_key, openai_key]):
        print("❌ Set SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY in .env")
        sys.exit(1)

    supabase = create_client(supabase_url, supabase_key)
    openai_client = AsyncOpenAI(api_key=openai_key)

    print(f"📊 Benchmarking {len(eval_items)} questions (top_k: {top_k_values})")
    print("-" * 50)

    # Collect predictions for max(top_k) to avoid redundant retrieval
    max_k = max(top_k_values)
    all_predictions: list[list[str]] = []

    for i, item in enumerate(eval_items):
        preds = await retrieve(
            supabase,
            openai_client,
            item["question"],
            top_k=max_k,
            similarity_threshold=similarity_threshold,
        )
        all_predictions.append(preds)
        if (i + 1) % 10 == 0:
            print(f"   Retrieved {i + 1}/{len(eval_items)}...")
        await asyncio.sleep(0.05)

    # Compute per-question scores for each k
    ground_truths = [[item["chunk_id"]] for item in eval_items]
    metrics = {}
    bootstrap_data: dict[str, list[float]] = {}

    for k in top_k_values:
        recalls = []
        mrrs = []
        for preds, gt in zip(all_predictions, ground_truths):
            preds_k = preds[:k]
            recalls.append(calculate_recall(preds_k, gt))
            mrrs.append(calculate_mrr(preds_k, gt))
        mean_recall = sum(recalls) / len(recalls)
        mean_mrr = sum(mrrs) / len(mrrs)
        metrics[f"Recall@{k}"] = mean_recall
        metrics[f"MRR@{k}"] = mean_mrr
        bootstrap_data[f"Recall@{k}"] = recalls
        bootstrap_data[f"MRR@{k}"] = mrrs

    metrics["_bootstrap_data"] = bootstrap_data

    return metrics


def bootstrap_confidence_intervals(
    per_question_scores: list[float],
    n_samples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """
    Resample with replacement, compute mean for each sample, return (mean, ci_low, ci_high).
    """
    n = len(per_question_scores)
    if n < 2:
        return per_question_scores[0] if per_question_scores else 0.0, 0.0, 0.0

    bootstrap_means = []
    for _ in range(n_samples):
        indices = random.choices(range(n), k=n)
        sample = [per_question_scores[i] for i in indices]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = 1 - ci
    lo_idx = int(n_samples * (alpha / 2))
    hi_idx = int(n_samples * (1 - alpha / 2))
    mean = sum(per_question_scores) / n
    return mean, bootstrap_means[lo_idx], bootstrap_means[hi_idx]


def print_results(metrics: dict, bootstrap: bool = False, n_samples: int = 1000):
    """Print metrics in a readable table. With bootstrap, show 95% CI."""
    bootstrap_data = metrics.pop("_bootstrap_data", None)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS" + (" (with 95% bootstrap CI)" if bootstrap else ""))
    print("=" * 60)

    for name in sorted(metrics.keys()):
        value = metrics[name]
        if bootstrap and bootstrap_data and name in bootstrap_data:
            mean, ci_lo, ci_hi = bootstrap_confidence_intervals(
                bootstrap_data[name], n_samples=n_samples
            )
            print(f"  {name:12} {mean:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            print(f"  {name:12} {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval/questions.jsonl"),
        help="Path to eval JSONL from generate_eval_set.py",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="K values for Recall@K and MRR@K",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=MINIMUM_SIMILARITY_THRESHOLD,
        help="Minimum similarity for retrieval",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute 95%% confidence intervals via bootstrap resampling",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    args = parser.parse_args()

    metrics = asyncio.run(
        run_benchmark(
            args.eval_file,
            args.top_k,
            args.similarity_threshold,
        )
    )
    print_results(metrics, bootstrap=args.bootstrap, n_samples=args.bootstrap_samples)
