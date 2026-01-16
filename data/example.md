# Example Document

This is a sample Markdown document for the RAG Starter Kit.

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines:
1. **Retrieval**: Finding relevant documents from a knowledge base
2. **Augmentation**: Adding those documents as context
3. **Generation**: Using an LLM to generate answers based on the context

## Why Use RAG?

RAG solves several problems with LLMs:
- **Hallucination**: LLMs make up facts. RAG grounds answers in real documents.
- **Outdated Knowledge**: LLMs have training cutoffs. RAG uses current documents.
- **Domain Expertise**: LLMs are generalists. RAG adds domain-specific knowledge.

## How It Works

1. User asks a question
2. System searches for relevant documents using vector similarity
3. Top documents are passed to the LLM as context
4. LLM generates an answer based on the retrieved context

This ensures answers are grounded in your actual documents, not the LLM's training data.
