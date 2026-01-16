"""
RAG Starter Kit: Streamlit Chat Interface

A clean, production-ready RAG chat interface using OpenAI and Supabase directly.

Usage:
    streamlit run app.py
"""

import os
import asyncio
import streamlit as st
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

# Import configuration
from config import (
    APP_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MINIMUM_SIMILARITY_THRESHOLD,
    MAX_SOURCES,
)

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def init_clients() -> Tuple[Client, Optional[AsyncOpenAI]]:
    """
    Initialize Supabase and OpenAI clients.
    
    Using @st.cache_resource ensures clients are reused across reruns,
    avoiding connection overhead.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not supabase_url or not supabase_key:
        st.error(
            "⚠️ **Supabase credentials not found**\n\n"
            "**Possible causes:**\n"
            "1. `.env` file missing or not in the project root\n"
            "2. Environment variables not set in Streamlit secrets (for deployed apps)\n"
            "3. Wrong variable names (should be `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`)\n\n"
            "**Fix:**\n"
            "- Create `.env` file with your Supabase credentials\n"
            "- Or set them in Streamlit secrets (for deployed apps)\n"
            "- Get credentials from: Supabase Dashboard → Settings → API"
        )
        st.stop()
    
    # Validate Supabase URL format
    if not supabase_url.startswith("https://"):
        st.error(
            f"⚠️ **Invalid SUPABASE_URL**\n\n"
            f"URL should start with 'https://'\n"
            f"Current value: `{supabase_url[:50]}...`\n\n"
            "**Fix:** Get the correct URL from Supabase Dashboard → Settings → API → Project URL"
        )
        st.stop()
    
    # Validate Service Role Key format
    if not supabase_key.startswith("eyJ"):
        st.error(
            "⚠️ **Invalid SUPABASE_SERVICE_KEY**\n\n"
            "Service Role Key should start with 'eyJ' (JWT token)\n\n"
            "**Common mistake:** You might be using the anon key instead of service role key\n\n"
            "**Fix:** Get the Service Role Key (not anon key) from:\n"
            "Supabase Dashboard → Settings → API → Service Role Key"
        )
        st.stop()
    
    try:
        supabase = create_client(supabase_url, supabase_key)
        # Test connection by trying to query the table
        test_result = supabase.table("site_pages").select("id").limit(1).execute()
    except Exception as e:
        error_msg = str(e).lower()
        if "relation" in error_msg and "does not exist" in error_msg:
            st.error(
                "❌ **Table 'site_pages' does not exist**\n\n"
                "**Fix:** See Step 1.2 Schema in the paid Substack guide to set up the database schema."
            )
        elif "permission denied" in error_msg or "row-level security" in error_msg:
            st.error(
                "❌ **Permission denied accessing Supabase**\n\n"
                "**Common cause:** Using anon key instead of service role key\n\n"
                "**Fix:** Get the Service Role Key from:\n"
                "Supabase Dashboard → Settings → API → Service Role Key\n\n"
                "⚠️ The anon key won't work for this app - you need the service role key"
            )
        elif "invalid api key" in error_msg or "unauthorized" in error_msg:
            st.error(
                "❌ **Invalid Supabase credentials**\n\n"
                "**Fix:** Check your SUPABASE_URL and SUPABASE_SERVICE_KEY\n"
                "Make sure you're using the Service Role Key (starts with 'eyJ')"
            )
        else:
            st.error(f"❌ **Supabase connection error:** {e}\n\nCheck your credentials and network connection")
        st.stop()
    
    # Validate OpenAI key format
    if openai_key and not openai_key.startswith("sk-"):
        st.warning(
            "⚠️ **Invalid OpenAI API key format**\n\n"
            "API key should start with 'sk-'\n\n"
            "**Fix:** Get a valid key from https://platform.openai.com/api-keys"
        )
        openai_client = None
    else:
        openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None
    
    return supabase, openai_client


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """
    Generate embedding vector for query text.
    
    Uses the same embedding model as configured for document ingestion to ensure
    semantic similarity works correctly.
    """
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e).lower()
        print(f"   ❌ ERROR generating embedding: {e}")
        if "invalid_api_key" in error_msg:
            st.error("❌ **Invalid OpenAI API key**\n\nGet a new key from https://platform.openai.com/api-keys")
        elif "insufficient_quota" in error_msg or "billing" in error_msg:
            st.error("❌ **OpenAI account has insufficient quota**\n\nAdd a payment method at https://platform.openai.com/account/billing")
        elif "rate_limit" in error_msg:
            st.warning("⚠️ **OpenAI rate limit exceeded**\n\nWait a few minutes and try again")
        else:
            st.error(f"❌ **Error generating embedding:** {e}")
        return [0.0] * 1536


async def retrieve_relevant_documents(
    supabase: Client,
    openai_client: AsyncOpenAI,
    query: str,
    max_results: int = 5,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    Retrieve relevant documents using vector similarity search.
    
    Uses cosine similarity (via pgvector) to find semantically similar documents.
    The threshold filters out irrelevant results - lower values return more results
    but with lower quality. Higher values are more strict but may miss relevant
    documents with different wording.
    
    Args:
        supabase: Supabase client
        openai_client: OpenAI client
        query: User query text
        max_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)
    
    Returns:
        List of relevant documents with similarity scores
    """
    try:
        # Generate query embedding
        print(f"   Generating query embedding...")
        query_embedding = await get_embedding(query, openai_client)
        print(f"   Query embedding generated ({len(query_embedding)} dimensions)")
        
        # Call the match_documents function
        # Using an RPC function encapsulates the SQL logic and allows parameterized queries
        try:
            print(f"   Searching database (threshold: {similarity_threshold})...")
            result = supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": max_results * 2,  # Get more, filter by threshold
                    "similarity_threshold": similarity_threshold,
                    "filter_source": None,
                }
            ).execute()
        except Exception as e:
            error_msg = str(e).lower()
            if "function" in error_msg and "does not exist" in error_msg:
                st.error(
                    "❌ **Function 'match_documents' does not exist**\n\n"
                    "**Fix:** See Step 1.2 Schema in the paid Substack guide to set up the database schema."
                )
            elif "vector" in error_msg and "does not exist" in error_msg:
                st.error(
                    "❌ **pgvector extension not enabled**\n\n"
                    "**Fix:** See Step 1.2 Schema in the paid Substack guide to set up the database schema."
                )
            else:
                st.error(
                    f"❌ **Error calling match_documents:** {e}\n\n"
                    "**Fix:** See Step 1.2 Schema in the paid Substack guide to set up the database schema."
                )
            return []
        
        if not result.data:
            print(f"   ⚠️ Database returned no results")
            return []
        
        print(f"   Database returned {len(result.data)} candidate(s)")
        
        # Filter by threshold and limit results
        filtered = [
            doc for doc in result.data
            if doc.get("similarity", 0) >= similarity_threshold
        ]
        
        # If no results meet threshold, return top 2 anyway
        # Sometimes queries are novel and don't match well. Returning low-similarity
        # results is better than no results, but we limit to 2 to avoid overwhelming
        # the user with irrelevant content.
        if not filtered and result.data:
            print(f"   ⚠️ No results above threshold ({similarity_threshold}), returning top 2 anyway")
            return result.data[:2]
        
        print(f"   {len(filtered)} document(s) passed similarity threshold")
        return filtered[:max_results]
        
    except Exception as e:
        # Error already handled in get_embedding or RPC call
        print(f"   ❌ ERROR in retrieval: {e}")
        return []


async def generate_response(
    openai_client: AsyncOpenAI,
    query: str,
    context_documents: List[Dict]
) -> str:
    """
    Generate AI response using retrieved context.
    
    Args:
        openai_client: OpenAI client
        query: User query
        context_documents: Retrieved relevant documents
    
    Returns:
        Generated response text
    """
    if not context_documents:
        return "I couldn't find relevant information in the knowledge base to answer your question."
    
    # Build context from retrieved documents
    # Include title, summary, and content to give the model enough context to
    # synthesize an answer. Including similarity scores helps the model weight
    # more relevant sources higher.
    context_parts = []
    for i, doc in enumerate(context_documents, 1):
        similarity_pct = int(doc.get("similarity", 0) * 100)
        context_parts.append(
            f"""Document {i} (Relevance: {similarity_pct}%):
Title: {doc.get('title', 'Untitled')}
Summary: {doc.get('summary', 'No summary')}
Content: {doc.get('content', '')}
---"""
        )
    
    context = "\n\n".join(context_parts)
    
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents.

Guidelines:
- Answer based ONLY on the information in the context documents
- If the context doesn't contain enough information, say so
- Cite which document(s) you're using when relevant
- Be concise and actionable
- If asked about something not in the context, politely decline

Context Documents:
{context}"""
    
    try:
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": query},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        error_msg = str(e).lower()
        print(f"   ❌ ERROR generating response: {e}")
        if "invalid_api_key" in error_msg:
            return "❌ **Error:** Invalid OpenAI API key. Get a new key from https://platform.openai.com/api-keys"
        elif "insufficient_quota" in error_msg or "billing" in error_msg:
            return "❌ **Error:** OpenAI account has insufficient quota. Add a payment method at https://platform.openai.com/account/billing"
        elif "rate_limit" in error_msg:
            return "⚠️ **Error:** OpenAI rate limit exceeded. Wait a few minutes and try again."
        elif "model" in error_msg and "not found" in error_msg:
            return f"❌ **Error:** Model '{LLM_MODEL}' not found. Check config.py for valid models."
        else:
            return f"❌ **Error generating response:** {e}"


def display_sources(sources: List[Dict]):
    """
    Display source citations in an expander.
    
    Showing sources builds trust and allows users to verify information.
    """
    if not sources:
        return
    
    with st.expander(f"📚 View {len(sources)} Source(s)", expanded=False):
        for i, source in enumerate(sources, 1):
            similarity = int(source.get("similarity", 0) * 100)
            
            st.markdown(f"**{i}. {source.get('title', 'Untitled')}**")
            st.caption(f"Relevance: {similarity}%")
            
            if source.get("url"):
                # Extract filename from file:// URL
                url = source.get("url", "")
                if url.startswith("file://"):
                    filename = url.replace("file://", "")
                    st.caption(f"📄 {filename}")
                else:
                    st.caption(f"🔗 [View source]({url})")
            
            if source.get("summary"):
                st.caption(source.get("summary"))
            
            st.divider()


async def main():
    """Main application logic."""
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # RAG Settings
        st.subheader("RAG Settings")
        
        max_sources = st.slider(
            "Max Sources",
            min_value=1,
            max_value=10,
            value=MAX_SOURCES,
            help="Maximum number of documents to retrieve for each query"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=MINIMUM_SIMILARITY_THRESHOLD,
            step=0.1,
            help="Minimum similarity score (0-1). Higher = more strict matching."
        )
        
        st.caption(f"Using model: {LLM_MODEL} (configure in config.py)")
        
        st.divider()
        
        # Database stats
        st.subheader("📊 Database Stats")
        try:
            supabase, _ = init_clients()
            result = supabase.table("site_pages").select("id").execute()
            total_chunks = len(result.data)
            st.metric("Total Chunks", total_chunks)
            
            unique_urls = len(set(
                row["url"] for row in supabase.table("site_pages")
                .select("url")
                .execute()
                .data
            ))
            st.metric("Unique Documents", unique_urls)
        except Exception as e:
            st.caption(f"Could not load stats: {e}")
    
    # Main chat interface
    st.title(f"🤖 {APP_NAME}")
    st.caption("Ask questions about your documents using semantic search")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        print(f"\n{'='*60}")
        print(f"📝 User Query: {prompt}")
        print(f"{'='*60}")
        
        # Initialize clients
        supabase, openai_client = init_clients()
        
        if not openai_client:
            print("❌ ERROR: OpenAI API key not configured")
            st.error(
                "⚠️ **OpenAI API key not configured**\n\n"
                "**Fix:** Add `OPENAI_API_KEY` to your `.env` file\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )
            st.stop()
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                print(f"🔍 Retrieving documents (threshold: {similarity_threshold}, max: {max_sources})...")
                # Retrieve relevant documents
                sources = await retrieve_relevant_documents(
                    supabase,
                    openai_client,
                    prompt,
                    max_results=max_sources,
                    similarity_threshold=similarity_threshold
                )
            
            if sources:
                print(f"✅ Retrieved {len(sources)} document(s):")
                for i, source in enumerate(sources, 1):
                    similarity = int(source.get("similarity", 0) * 100)
                    title = source.get("title", "Untitled")
                    print(f"   {i}. {title} (similarity: {similarity}%)")
                
                st.success(f"✅ Found {len(sources)} relevant document(s)")
                
                with st.spinner("🤖 Generating response..."):
                    print(f"🤖 Generating response with {LLM_MODEL}...")
                    # Generate AI response
                    response = await generate_response(
                        openai_client,
                        prompt,
                        sources
                    )
                    print(f"✅ Response generated ({len(response)} characters)")
                
                # Display response
                st.markdown(response)
                
                # Display sources
                display_sources(sources)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
            else:
                print("⚠️ No relevant documents found (similarity threshold may be too high)")
                st.warning("No relevant documents found. Try rephrasing your question.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't find relevant information to answer your question.",
                    "sources": []
                })


# Handle async execution in Streamlit
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if not loop.is_running():
        asyncio.run(main())
    else:
        import nest_asyncio
        nest_asyncio.apply()
        loop.run_until_complete(main())
