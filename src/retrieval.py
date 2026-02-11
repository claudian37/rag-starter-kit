"""
Shared retrieval logic for app and benchmark.

Runs vector similarity search via Supabase match_documents RPC, with app-side
similarity filtering and fallback behavior (return top 2 when no results
meet threshold). Used by both app.py and benchmark.py to ensure benchmark
results reflect production retrieval behavior.
"""

from typing import Any, List

from openai import AsyncOpenAI

from config import EMBEDDING_MODEL, RETRIEVAL_FALLBACK_LIMIT


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Generate embedding for query text using configured model."""
    response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
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
    query_embedding = await get_embedding(query, openai_client)

    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": max_results * 2,
            "similarity_threshold": similarity_threshold,
            "filter_source": None,
        },
    ).execute()

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
