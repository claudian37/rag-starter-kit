"""
RAG Starter Kit: Retrieval Benchmark

Measures Recall@K and MRR@K on a synthetic eval set. Run AFTER generate_eval_set.py
and ingestion. Use before/after any retrieval change to quantify impact.

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
from tqdm import tqdm

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
    Retrieval includes fallback behavior (returning top results when none meet
    threshold), so metrics reflect production retrieval rather than strict similarity.
    """
    docs = await retrieve_documents(
        supabase,
        openai_client,
        question,
        max_results=top_k,
        similarity_threshold=similarity_threshold,
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
    concurrency: int = 5,
) -> dict:
    """Run benchmark and return metrics."""
    if not eval_path.exists():
        print(f"❌ Eval file not found: {eval_path}")
        print("   Run: python src/generate_eval_set.py")
        sys.exit(1)

    eval_items = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("expected JSON object")
                if "question" not in item or "chunk_id" not in item:
                    raise ValueError("missing required keys: question, chunk_id")
                eval_items.append(item)
            except (json.JSONDecodeError, ValueError) as e:
                print(
                    f"⚠️ Skipping malformed item on line {line_number} in {eval_path}: {e}"
                )
                continue

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

    max_k = max(top_k_values)
    max_retries = 3
    retry_delays = [1.0, 2.0, 4.0]

    async def fetch_one(item: dict, index: int) -> tuple[int, list[str] | None, str | None]:
        """Retrieve for one item with retry; returns (index, preds, chunk_id) or (index, None, None) on skip."""
        for attempt in range(max_retries):
            try:
                preds = await retrieve(
                    supabase,
                    openai_client,
                    item["question"],
                    top_k=max_k,
                    similarity_threshold=similarity_threshold,
                )
                return (index, preds, item["chunk_id"])
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delays[attempt])
                else:
                    print(f"   ⚠️ Skipped item {index + 1} after {max_retries} attempts: {e}")
                    return (index, None, None)

    print(f"📊 Benchmarking {len(eval_items)} questions (top_k: {top_k_values}, {concurrency} concurrent)")
    print("-" * 50)

    results: list[tuple[list[str] | None, str | None] | None] = [None] * len(eval_items)
    queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()
    for i, item in enumerate(eval_items):
        await queue.put((i, item))
    for _ in range(concurrency):
        await queue.put((None, None))

    async def worker(pbar: tqdm) -> None:
        while True:
            index, item = await queue.get()
            try:
                if index is None:
                    return
                _, preds, chunk_id = await fetch_one(item, index)
                results[index] = (preds, chunk_id)
                pbar.update(1)
            finally:
                queue.task_done()

    with tqdm(total=len(eval_items), desc="Retrieving", unit="q") as pbar:
        workers = [asyncio.create_task(worker(pbar)) for _ in range(concurrency)]
        await queue.join()
        await asyncio.gather(*workers)

    all_predictions: list[list[str]] = []
    ground_truths: list[list[str]] = []
    skipped = 0
    for (preds, chunk_id) in results:
        if preds is not None:
            all_predictions.append(preds)
            ground_truths.append([chunk_id])
        else:
            skipped += 1

    if skipped:
        print(f"   ⚠️ {skipped} item(s) skipped due to errors (metrics exclude them)")

    if not all_predictions:
        print("❌ No successful retrievals; cannot compute metrics")
        sys.exit(1)

    # Compute per-question scores for each k
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

    When n < 2, bootstrap cannot produce meaningful intervals. Returns (mean, 0.0, 0.0)
    for a single score, or (0.0, 0.0, 0.0) for empty input.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not (0 < ci < 1):
        raise ValueError("ci must be in (0, 1)")

    n = len(per_question_scores)
    if n < 2:
        return per_question_scores[0] if per_question_scores else 0.0, 0.0, 0.0

    bootstrap_means = []
    for _ in range(n_samples):
        indices = random.choices(range(n), k=n)
        sample = [per_question_scores[i] for i in indices]
        bootstrap_means.append(sum(sample) / len(sample))

    bootstrap_means.sort()
    alpha = 1 - ci
    # Use (n_samples - 1) * p for percentile indexing so 97.5% -> index 974 for n=1000
    lo_idx = int((n_samples - 1) * (alpha / 2))
    hi_idx = int((n_samples - 1) * (1 - alpha / 2))
    lo_idx = max(0, min(lo_idx, n_samples - 1))
    hi_idx = max(0, min(hi_idx, n_samples - 1))
    mean = sum(per_question_scores) / n
    return mean, bootstrap_means[lo_idx], bootstrap_means[hi_idx]


def print_results(metrics: dict, bootstrap: bool = False, n_samples: int = 1000):
    """Print metrics in a readable table. With bootstrap, show 95% CI."""
    bootstrap_data = metrics.get("_bootstrap_data")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS" + (" (with 95% bootstrap CI)" if bootstrap else ""))
    print("=" * 60)

    for name in sorted(metrics.keys()):
        if name == "_bootstrap_data":
            continue
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent retrieval requests (default: 5)",
    )
    args = parser.parse_args()

    if args.bootstrap and args.bootstrap_samples < 1:
        print("❌ --bootstrap-samples must be >= 1")
        sys.exit(1)

    metrics = asyncio.run(
        run_benchmark(
            args.eval_file,
            args.top_k,
            args.similarity_threshold,
            args.concurrency,
        )
    )
    print_results(metrics, bootstrap=args.bootstrap, n_samples=args.bootstrap_samples)
