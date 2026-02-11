"""
Shared retrieval logic for app and benchmark.

Runs vector similarity search via Supabase match_documents RPC, with app-side
similarity filtering and fallback behavior (returning a configurable number
of top results when no results meet threshold). Used by both app.py and
benchmark.py to ensure benchmark results reflect production retrieval behavior.
"""

import asyncio
from typing import Any, List

from openai import AsyncOpenAI, OpenAIError

from config import EMBEDDING_MODEL, RETRIEVAL_FALLBACK_LIMIT


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Generate embedding for query text using configured model."""
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
    except OpenAIError as e:
        # Wrap OpenAI-specific errors, but include original message so callers
        # (e.g., app.py) can still inspect substrings like "invalid_api_key"
        # or "rate_limit" for targeted error handling.
        raise RuntimeError(f"Failed to generate embedding from OpenAI API: {e}") from e

    # Basic defensive check in case the API response is unexpectedly empty.
    if not getattr(response, "data", None):
        raise RuntimeError("OpenAI embedding API returned an empty response")
    return response.data[0].embedding


async def retrieve_documents(
    supabase: Any,
    openai_client: AsyncOpenAI,
    query: str,
    max_results: int = 5,
    similarity_threshold: float = 0.3,
    fallback_limit: int | None = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Retrieve relevant documents using vector similarity search.

    Uses cosine similarity (via pgvector) to find semantically similar documents.
    Applies app-side similarity filtering and falls back to top N results when
    no documents meet the threshold.

    Args:
        supabase: Supabase client
        openai_client: OpenAI client
        query: User query text
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)
        fallback_limit: When no results meet threshold, return this many anyway
        verbose: If True, print progress (for app); False for benchmark

    Returns:
        List of relevant documents (with url, chunk_number, similarity, etc.)
    """
    if fallback_limit is None:
        fallback_limit = RETRIEVAL_FALLBACK_LIMIT
    if max_results < 0:
        raise ValueError(f"max_results must be non-negative, got {max_results}")
    if fallback_limit < 0:
        raise ValueError(f"fallback_limit must be non-negative, got {fallback_limit}")
    match_count_base = max(max_results, fallback_limit)
    query_embedding = await get_embedding(query, openai_client)

    def _execute_rpc():
        return supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": match_count_base * 2,
                "similarity_threshold": similarity_threshold,
                "filter_source": None,
            },
        ).execute()

    result = await asyncio.to_thread(_execute_rpc)

    if not result.data:
        return []

    if verbose:
        print(f"   Database returned {len(result.data)} candidate(s)")

    filtered = [
        doc for doc in result.data
        if doc.get("similarity", 0) >= similarity_threshold
    ]

    if not filtered and result.data:
        if verbose:
            print(
                f"   ⚠️ No results above threshold ({similarity_threshold}), "
                f"returning top {fallback_limit} anyway"
            )
        return result.data[:fallback_limit]

    if verbose:
        print(f"   {len(filtered)} document(s) passed similarity threshold")

    return filtered[:max_results]


def build_chunk_id(row: dict) -> str:
    """Build chunk_id from Supabase row (url + chunk_number)."""
    url = row.get("url", "")
    chunk_num = row.get("chunk_number", 0)
    return f"{url}|{chunk_num}"
