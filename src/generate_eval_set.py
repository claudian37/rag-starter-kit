"""
RAG Starter Kit: Synthetic Evaluation Set Generator

Generates question–chunk pairs for benchmarking retrieval. Uses LLM to create
questions that can be answered by each chunk. Run this AFTER ingesting your data.

Usage:
    python src/generate_eval_set.py
    python src/generate_eval_set.py --num-per-chunk 2 --output eval/questions.jsonl

Output: JSONL file with {question, chunk_id, chunk} (chunk truncated to 500 characters) for benchmark.py
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import LLM_MODEL
from ingest import chunk_text

load_dotenv()

# Constraints to encourage diverse questions (variations in style, scope, and framing)
DIVERSITY_CONSTRAINTS = [
    "Shift timeframes when applicable (e.g. 'last quarter' vs 'this year', 'past 3 months' vs 'annually').",
    "Weave in a tangential detail (e.g. a hypothetical scenario or unrelated fact) that does not appear in the chunk.",
    "Swap concrete entities (e.g. a named company for another, a city for a different one, or a number for another).",
    "Use colloquial or shorthand phrasing (e.g. abbreviations, casual tone) as a real user might type.",
    "Reframe the question (e.g. from 'what is X' to 'why does X happen', or from beginner to expert lens).",
]


def _extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    content = content.strip()
    # Strip ```json or ``` code block wrappers
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", content)
    if match:
        return match.group(1).strip()
    return content


async def generate_question_for_chunk(
    client: AsyncOpenAI,
    chunk_content: str,
    chunk_id: str,
    constraint: str,
) -> dict | None:
    """Generate a single question that can be answered by the chunk."""
    prompt = f"""Generate a hypothetical question that can be answered using the following text chunk.

Text chunk:
{chunk_content[:2000]}

Rules:
- The question should be at most 2 sentences.
- Do not copy phrases verbatim from the chunk; paraphrase.
- Make the question realistic (how a reader might actually ask).
- The question must be answerable using this chunk or with a small tweak.
- Apply this constraint: {constraint}

Respond with a JSON object: {{"question": "your question here"}}"""

    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()
        content = _extract_json_from_response(content)
        data = json.loads(content)
        return {
            "question": data["question"],
            "chunk_id": chunk_id,
            "chunk": chunk_content[:500],  # Store truncated for reference
        }
    except Exception as e:
        print(f"⚠️ Error generating question for chunk {chunk_id[:20]}...: {e}")
        return None


def load_chunks_from_data_dir(data_dir: Path) -> list[tuple[str, str]]:
    """
    Load and chunk markdown files the same way ingest.py does.
    Returns list of (chunk_id, chunk_text).
    chunk_id format: file://{filename}|{chunk_index} (url|chunk_number) to match
    Supabase ingest and benchmark ground-truth construction.
    """
    chunks = []
    for path in sorted(data_dir.glob("*.md")) + sorted(data_dir.glob("*.markdown")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            continue
        text_chunks = chunk_text(content)
        url = f"file://{path.name}"
        for i, text in enumerate(text_chunks):
            chunk_id = f"{url}|{i}"
            chunks.append((chunk_id, text))
    return chunks


async def main(
    output_path: Path,
    num_per_chunk: int,
    data_dir: Path,
    batch_size: int = 100,
):
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    chunk_list = load_chunks_from_data_dir(data_dir)
    if not chunk_list:
        print(f"⚠️ No markdown chunks found in {data_dir}")
        sys.exit(1)

    print(f"📄 Loaded {len(chunk_list)} chunks from {data_dir}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream tasks in batches (avoids holding all tasks/coros in memory at once)
    sem = asyncio.Semaphore(10)
    total = len(chunk_list) * num_per_chunk

    async def generate_with_sem(chunk_id: str, text: str, constraint: str):
        async with sem:
            return await generate_question_for_chunk(client, text, chunk_id, constraint)

    async def process_batch(
        current_batch: list[tuple[str, str, str]], done_count: int, written_count: int
    ) -> tuple[int, int]:
        if not current_batch:
            return done_count, written_count
        coros = [generate_with_sem(cid, t, c) for cid, t, c in current_batch]
        raw_results = await asyncio.gather(*coros)
        batch_results = [r for r in raw_results if r is not None]
        for item in batch_results:
            f.write(json.dumps(item) + "\n")
        done_count += len(current_batch)
        written_count += len(batch_results)
        if done_count % 500 == 0 or done_count == total:
            print(f"   ... {done_count}/{total} done")
        return done_count, written_count

    print(f"   Generating {total} questions (batch size {batch_size}, 10 concurrent)...")
    done, written = 0, 0
    batch: list[tuple[str, str, str]] = []

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk_id, text in chunk_list:
            for _ in range(num_per_chunk):
                constraint = random.choice(DIVERSITY_CONSTRAINTS)
                batch.append((chunk_id, text, constraint))
                if len(batch) >= batch_size:
                    done, written = await process_batch(batch, done, written)
                    batch = []
        if batch:
            done, written = await process_batch(batch, done, written)

    print(f"✅ Wrote {written} question–chunk pairs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic eval set for RAG benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/questions.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num-per-chunk",
        type=int,
        default=1,
        help="Questions to generate per chunk (more = more diversity)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing markdown files (same as ingest)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Process this many tasks per batch (bounds memory for large datasets)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.output, args.num_per_chunk, args.data_dir, args.batch_size))
