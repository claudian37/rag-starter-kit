"""
RAG Starter Kit: Synthetic Evaluation Set Generator

Generates question–chunk pairs for benchmarking retrieval. Uses LLM to create
questions that can be answered by each chunk. Run this AFTER ingesting your data.

Based on concepts from the "Systematically Improving RAG" course (Maven).

Usage:
    python src/generate_eval_set.py
    python src/generate_eval_set.py --num-per-chunk 2 --output eval/questions.jsonl

Output: JSONL file with {question, chunk_id, chunk_text} for benchmark.py
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

from config import LLM_MODEL
from ingest import chunk_text

load_dotenv()

# Constraints to encourage diverse questions (from course: Week 1 synthetic_questions)
DIVERSITY_CONSTRAINTS = [
    "If there's a time period mentioned, modify it slightly (e.g. 6 months instead of a year).",
    "Add some irrelevant context (e.g. a backstory or scenario not in the chunk).",
    "Change specific values (e.g. different product, location, or number).",
    "Phrase as a casual or abbreviated question a real user might ask.",
    "Ask from a different angle (e.g. 'how do I' vs 'what happens if').",
]


async def generate_question_for_chunk(
    client: AsyncOpenAI,
    chunk_text: str,
    chunk_id: str,
    constraint: str,
) -> dict | None:
    """Generate a single question that can be answered by the chunk."""
    prompt = f"""Generate a hypothetical question that can be answered using the following text chunk.

Text chunk:
{chunk_text[:2000]}

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
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        return {
            "question": data["question"],
            "chunk_id": chunk_id,
            "chunk": chunk_text[:500],  # Store truncated for reference
        }
    except Exception as e:
        print(f"⚠️ Error generating question for chunk {chunk_id[:20]}...: {e}")
        return None


def load_chunks_from_data_dir(data_dir: Path) -> list[tuple[str, str]]:
    """
    Load and chunk markdown files the same way ingest.py does.
    Returns list of (chunk_id, chunk_text).
    chunk_id format: filename|chunk_N for matching with Supabase (url + chunk_number).
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

    # Build list of (chunk_id, chunk_text, constraint) for each generation
    tasks = []
    for chunk_id, text in chunk_list:
        for _ in range(num_per_chunk):
            constraint = random.choice(DIVERSITY_CONSTRAINTS)
            tasks.append((chunk_id, text, constraint))

    # Run concurrent generations (semaphore caps to avoid rate limits)
    sem = asyncio.Semaphore(10)

    async def generate_with_sem(chunk_id: str, text: str, constraint: str):
        async with sem:
            return await generate_question_for_chunk(client, text, chunk_id, constraint)

    print(f"   Generating {len(tasks)} questions (10 concurrent)...")
    coros = [generate_with_sem(cid, t, c) for cid, t, c in tasks]
    raw_results = await asyncio.gather(*coros)
    results = [r for r in raw_results if r is not None]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Wrote {len(results)} question–chunk pairs to {output_path}")


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
    args = parser.parse_args()

    asyncio.run(main(args.output, args.num_per_chunk, args.data_dir))
