"""RAG (Retrieval-Augmented Generation) pipeline for QuantMind.

Phase 2 of the QuantMind system. Ingests financial documents from:
- SEC EDGAR API (10-K, 10-Q, 8-K filings — free, no key)
- NewsAPI + RSS feeds (financial news — free tier)
- Local PDF files (research papers, analyst reports)

Stores embeddings in ChromaDB (local, persistent, no cloud needed).
Retrieves relevant context using semantic search + MMR reranking.
"""
