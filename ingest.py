"""
RAG Starter Kit: Data Ingestion Script

This script processes Markdown files from the /data folder and ingests them into Supabase
with vector embeddings for semantic search.

Usage:
    python ingest.py

Requirements:
    - Markdown files (.md or .markdown) in ./data/ directory
    - Environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY
"""

import os
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

# Import configuration
from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    MAX_CHUNK_SIZE,
    SUMMARY_TEMPERATURE,
    SUMMARY_MAX_TOKENS,
    SUMMARY_PREVIEW_LENGTH,
)

# Import validation
from validate_setup import validate_env_vars, validate_supabase_connection, validate_openai_connection

load_dotenv()

# Initialize clients (will be validated before use)
openai_client: Optional[AsyncOpenAI] = None
supabase: Optional[Client] = None


def chunk_text(text: str, max_chunk_size: int = None) -> List[str]:
    """
    Split text into chunks for embedding.
    
    Prefers paragraph boundaries to preserve semantic meaning, but truncates
    if needed to respect the maximum chunk size.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum characters per chunk (defaults to config.MAX_CHUNK_SIZE)
    
    Returns:
        List of text chunks
    """
    if max_chunk_size is None:
        max_chunk_size = MAX_CHUNK_SIZE
    # If text is short enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first (double newlines)
    paragraphs = text.split("\n\n")
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed limit, save current chunk and start new one
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If we still have chunks that are too long, truncate them
    # Some paragraphs are extremely long (code blocks, lists). Truncation preserves
    # the beginning which usually contains the most important info.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            # Try to truncate at sentence boundary
            truncated = chunk[:max_chunk_size]
            last_period = truncated.rfind(".")
            if last_period > max_chunk_size * 0.8:  # If period in last 20%
                truncated = truncated[:last_period + 1]
            final_chunks.append(truncated)
        else:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else [text[:max_chunk_size]]


async def get_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for text using OpenAI.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector (dimensions depend on the configured model)
    """
    if not openai_client:
        raise ValueError("OpenAI client not initialized. Check your OPENAI_API_KEY.")
    
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e).lower()
        
        if "invalid_api_key" in error_msg or "incorrect api key" in error_msg:
            raise ValueError(
                "❌ Invalid OpenAI API key\n"
                "💡 Get a new key from https://platform.openai.com/api-keys\n"
                "   Make sure it starts with 'sk-' and is set in .env file"
            )
        elif "insufficient_quota" in error_msg or "billing" in error_msg:
            raise ValueError(
                "❌ OpenAI account has insufficient quota\n"
                "💡 Add a payment method at https://platform.openai.com/account/billing\n"
                "   Free tier gives $5 credit to start"
            )
        elif "rate_limit" in error_msg:
            raise ValueError(
                "❌ OpenAI rate limit exceeded\n"
                "💡 Wait a few minutes and try again\n"
                "   The script will continue with remaining files"
            )
        elif "model" in error_msg and "not found" in error_msg:
            raise ValueError(
                f"❌ Embedding model '{EMBEDDING_MODEL}' not found\n"
                "💡 Check config.py and verify the model name is correct"
            )
        else:
            raise ValueError(f"❌ Error getting embedding: {e}\n💡 Check your OpenAI API key and account status")


async def get_summary(text: str) -> str:
    """
    Generate a concise summary of the text chunk.
    
    Summaries are generated for each document chunk in /data during ingestion
    to help users quickly identify relevant documents.
    
    Args:
        text: Text to summarize
    
    Returns:
        Summary string (1-2 sentences)
    """
    system_prompt = """
    Create a concise summary of this content in 1-2 sentences.
    Focus on the main points and key information.
    """
    
    if not openai_client:
        return "Content summary unavailable (OpenAI client not initialized)"
    
    try:
        # Only send preview to save tokens
        # Summaries don't need full context. Truncating saves API costs.
        preview = text[:SUMMARY_PREVIEW_LENGTH] + "..." if len(text) > SUMMARY_PREVIEW_LENGTH else text
        
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{preview}"},
            ],
            temperature=SUMMARY_TEMPERATURE,
            max_tokens=SUMMARY_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid_api_key" in error_msg or "insufficient_quota" in error_msg:
            # Don't fail ingestion for summary errors, but log it
            print(f"⚠️ Could not generate summary (API issue): {e}")
        return "Content summary unavailable"


def extract_title_from_markdown(content: str, filename: str) -> str:
    """
    Extract title from Markdown content or use filename.
    
    Prefers H1 headers over filenames as they're usually more descriptive.
    
    Args:
        content: Markdown content
        filename: Fallback filename
    
    Returns:
        Title string
    """
    # Try to extract H1 header
    lines = content.split("\n")
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        elif line.startswith("#"):
            return line[1:].strip()
    
    # Fallback to filename without extension
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


async def process_file(file_path: Path) -> bool:
    """
    Process a single Markdown file: chunk, embed, and insert into database.
    
    Processing files sequentially ensures we don't overwhelm the API rate limits.
    For large datasets, consider batching with asyncio.gather() but add rate limiting
    to avoid 429 errors.
    
    Args:
        file_path: Path to Markdown file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📄 Processing: {file_path.name}")
        
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            print(f"⚠️ Skipping empty file: {file_path.name}")
            return False
        
        # Extract metadata from plain markdown
        title = extract_title_from_markdown(content, file_path.name)
        url = f"file://{file_path.name}"  # Use filename as URL identifier
        source = "markdown_file"
        
        # Check if already processed
        if not supabase:
            raise ValueError("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        
        try:
            existing = (
                supabase.table("site_pages")
                .select("url")
                .eq("url", url)
                .execute()
            )
            if existing.data:
                print(f"⏭️ Already exists, skipping: {file_path.name}")
                return True
        except Exception as e:
            error_msg = str(e).lower()
            if "relation" in error_msg and "does not exist" in error_msg:
                raise ValueError(
                    "❌ Table 'site_pages' does not exist in Supabase\n"
                    "💡 See Step 1.2 Schema in the paid Substack guide to set up the database schema."
                )
            elif "permission denied" in error_msg or "row-level security" in error_msg:
                raise ValueError(
                    "❌ Permission denied accessing Supabase table\n"
                    "💡 You might be using the anon key instead of service role key\n"
                    "   Get the Service Role Key from: Settings → API → Service Role Key"
                )
            else:
                raise ValueError(f"❌ Error checking existing documents: {e}\n💡 Check your Supabase credentials and table setup")
        
        # Chunk the content
        chunks = chunk_text(content)
        print(f"📝 Split into {len(chunks)} chunk(s)")
        
        # Process each chunk
        successful = 0
        for chunk_num, chunk in enumerate(chunks):
            try:
                # Generate embedding and summary in parallel
                # Parallel API calls reduce latency since OpenAI can handle concurrent requests.
                embedding, summary = await asyncio.gather(
                    get_embedding(chunk),
                    get_summary(chunk)
                )
                
                # Prepare data for insertion
                data = {
                    "url": url,
                    "chunk_number": chunk_num,
                    "title": title if chunk_num == 0 else f"{title} - Part {chunk_num + 1}",
                    "summary": summary,
                    "content": chunk,
                    "metadata": {
                        "filename": file_path.name,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks),
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "embedding": embedding,
                    "source": source,
                }
                
                # Insert into database
                try:
                    result = supabase.table("site_pages").insert(data).execute()
                    successful += 1
                    print(f"✅ Inserted chunk {chunk_num + 1}/{len(chunks)}")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "duplicate key" in error_msg or "unique constraint" in error_msg:
                        print(f"⏭️ Chunk {chunk_num + 1} already exists, skipping")
                        successful += 1  # Count as success since it's already there
                    elif "permission denied" in error_msg:
                        raise ValueError(
                            "❌ Permission denied inserting into Supabase\n"
                            "💡 Make sure you're using the Service Role Key (not anon key)\n"
                            "   Get it from: Settings → API → Service Role Key"
                        )
                    else:
                        raise ValueError(f"❌ Error inserting chunk: {e}\n💡 Check your Supabase credentials and table structure")
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except ValueError as e:
                # Re-raise validation errors
                raise
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_num}: {e}")
                print("💡 Continuing with next chunk...")
                continue
        
        print(f"✅ Successfully processed {file_path.name} ({successful}/{len(chunks)} chunks)")
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error processing file {file_path.name}: {e}")
        return False


async def main():
    """
    Main ingestion function: process all Markdown files in ./data directory.
    
    Processing files sequentially is safer for rate limits but slower.
    For production with 100+ files, consider batching 5-10 files at a time with
    asyncio.gather() and exponential backoff for rate limit errors.
    """
    data_dir = Path("./data")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("💡 Create a ./data/ directory and add your Markdown files there.")
        return
    
    # Find all Markdown files
    markdown_files = list(data_dir.glob("*.md")) + list(data_dir.glob("*.markdown"))
    
    if not markdown_files:
        print(f"⚠️ No Markdown files found in {data_dir}")
        print("💡 Add .md or .markdown files to the ./data/ directory.")
        return
    
    print(f"🚀 Starting ingestion of {len(markdown_files)} file(s)...")
    print("-" * 60)
    
    # Process files sequentially
    successful = 0
    failed = 0
    
    for file_path in markdown_files:
        if await process_file(file_path):
            successful += 1
        else:
            failed += 1
        print()  # Blank line between files
    
    # Final summary
    print("=" * 60)
    print("📊 INGESTION COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📄 Total: {len(markdown_files)}")
    
    # Show database stats
    try:
        result = supabase.table("site_pages").select("id").execute()
        total_chunks = len(result.data)
        unique_urls = len(set(
            row["url"] for row in supabase.table("site_pages")
            .select("url")
            .execute()
            .data
        ))
        print(f"📚 Total chunks in database: {total_chunks}")
        print(f"📄 Unique documents: {unique_urls}")
    except Exception as e:
        print(f"⚠️ Could not fetch database stats: {e}")


if __name__ == "__main__":
    print("🔍 Validating setup...\n")
    
    # Validate environment variables
    env_valid, env_errors = validate_env_vars()
    if not env_valid:
        print("❌ Environment validation failed:\n")
        for error in env_errors:
            print(f"{error}\n")
        print("💡 Fix the errors above and try again.")
        sys.exit(1)
    
    # Initialize clients
    try:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        sys.exit(1)
    
    # Validate connections
    print("🔍 Testing Supabase connection...")
    supabase_valid, supabase_error = validate_supabase_connection()
    if not supabase_valid:
        print(f"\n{supabase_error}\n")
        sys.exit(1)
    print("✅ Supabase connection OK")
    
    print("🔍 Testing OpenAI connection...")
    openai_valid, openai_error = validate_openai_connection()
    if not openai_valid:
        print(f"\n{openai_error}\n")
        sys.exit(1)
    print("✅ OpenAI connection OK\n")
    
    # Run async ingestion
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Ingestion interrupted by user")
        sys.exit(0)
    except ValueError as e:
        # Validation errors from our code
        print(f"\n{e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        print("💡 Check the error message above and your configuration")
        sys.exit(1)
