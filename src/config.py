"""
RAG Starter Kit: Configuration

Centralized configuration for easy customization. Edit these values to adjust
the behavior of the RAG system without modifying core code.

Configuration Priority:
1. Environment variables (.env file) - Recommended for API keys and secrets
2. Default values in this file - Used if env var is not set

To configure:
- Create a .env file in the project root (copy from .env.example)
- Add your settings: OPENAI_API_KEY=sk-..., SUPABASE_URL=..., etc.
- For advanced settings, you can override defaults via .env or edit src/config.py directly
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Application Configuration
# ============================================================================

# Application name displayed in the UI
APP_NAME = os.getenv("APP_NAME", "My RAG Starter Kit")

# ============================================================================
# OpenAI Configuration
# ============================================================================

# Embedding model for converting text to vectors
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# LLM model for generating answers
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")

# Temperature for LLM (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Maximum tokens for LLM response
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# ============================================================================
# Chunking Configuration
# ============================================================================

# Maximum characters per chunk
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "5000"))

# ============================================================================
# Retrieval Configuration
# ============================================================================

# Minimum similarity threshold for document retrieval (0.0 to 1.0)
# Only documents with similarity scores above this threshold will be returned
MINIMUM_SIMILARITY_THRESHOLD = float(os.getenv("MINIMUM_SIMILARITY_THRESHOLD", "0.3"))

# Maximum number of documents to retrieve per query
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "5"))

# ============================================================================
# Summary Generation Configuration
# ============================================================================

# Temperature for summary generation (lower = more factual)
# Summaries are generated for each document chunk in /data during ingestion
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.3"))

# Maximum tokens for summaries
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "100"))

# Characters to send for summary generation (saves tokens)
SUMMARY_PREVIEW_LENGTH = int(os.getenv("SUMMARY_PREVIEW_LENGTH", "1000"))
