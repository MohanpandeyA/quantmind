"""Semantic retriever with MMR reranking for QuantMind RAG pipeline.

Retrieves the most relevant AND diverse document chunks for a query.

Two retrieval modes:
    1. Simple top-K: return the K most similar chunks (fast, may have duplicates)
    2. MMR (Maximal Marginal Relevance): return K diverse, relevant chunks (better quality)

Why MMR matters for financial RAG:
    Without MMR (top-K similarity):
        Query: "Why is AAPL revenue declining?"
        Result 1: "iPhone revenue declined 8% in Q3" ← relevant
        Result 2: "iPhone revenue fell 8% this quarter" ← DUPLICATE
        Result 3: "iPhone revenue dropped 8% YoY" ← DUPLICATE again
        → LLM gets 3 copies of the same fact, misses other context

    With MMR:
        Result 1: "iPhone revenue declined 8% in Q3" ← relevant
        Result 2: "China market share fell due to Huawei competition" ← NEW info
        Result 3: "Supply chain disruptions in Taiwan" ← NEW info
        → LLM gets diverse, complementary context → better answer

MMR Algorithm:
    For each iteration i from 1 to K:
        score(d) = λ × similarity(query, d) - (1-λ) × max_similarity(d, selected)
        Select document with highest score
        Add to selected set

    λ (lambda) controls the tradeoff:
        λ = 1.0 → pure similarity (same as top-K)
        λ = 0.5 → balanced relevance + diversity (recommended)
        λ = 0.0 → pure diversity (ignores query)

Optimization principles:
    - Fetch more candidates than needed (fetch_k > n_results)
      then rerank with MMR — better quality than fetching exactly n_results
    - Vectorized similarity: numpy dot products for fast cosine similarity
    - Context builder: formats results into LLM-ready prompt context
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config.logging_config import get_logger
from rag.embeddings import EmbeddingModel
from rag.vector_store import SearchResult, VectorStore

logger = get_logger(__name__)

# Default MMR parameters
DEFAULT_N_RESULTS = 5       # Final number of results to return
DEFAULT_FETCH_K = 20        # Candidates to fetch before MMR reranking
DEFAULT_LAMBDA = 0.5        # MMR lambda: 0.5 = balanced relevance + diversity


class Retriever:
    """Semantic retriever with MMR reranking for financial document search.

    Combines vector similarity search (ChromaDB) with MMR reranking
    to return relevant AND diverse document chunks.

    Attributes:
        vector_store: ChromaDB vector store instance.
        embedding_model: Embedding model for query encoding.
        n_results: Default number of results to return.
        fetch_k: Candidates to fetch before MMR reranking.
        mmr_lambda: MMR diversity parameter (0.0-1.0).

    Example:
        >>> retriever = Retriever(vector_store, embedding_model)
        >>> results = await retriever.retrieve(
        ...     query="Why is AAPL revenue declining?",
        ...     ticker="AAPL",
        ...     n_results=5,
        ... )
        >>> context = retriever.build_context(results)
        >>> print(context)
        [1] SEC EDGAR | Apple 10-K (2024-11-01)
        iPhone revenue declined 8% in Q3 2024...
        ...
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        n_results: int = DEFAULT_N_RESULTS,
        fetch_k: int = DEFAULT_FETCH_K,
        mmr_lambda: float = DEFAULT_LAMBDA,
    ) -> None:
        """Initialize Retriever.

        Args:
            vector_store: VectorStore instance for document search.
            embedding_model: EmbeddingModel for query encoding.
            n_results: Default number of results. Default 5.
            fetch_k: Candidates to fetch before MMR. Default 20.
                     Rule: fetch_k should be 3-5× n_results.
            mmr_lambda: MMR lambda parameter. Default 0.5.
                        0.5 = balanced relevance + diversity.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.n_results = n_results
        self.fetch_k = max(fetch_k, n_results * 2)  # Always fetch at least 2× n_results
        self.mmr_lambda = mmr_lambda

    async def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        use_mmr: bool = True,
    ) -> List[SearchResult]:
        """Retrieve relevant document chunks for a query.

        Steps:
            1. Embed the query using the embedding model
            2. Fetch fetch_k candidates from ChromaDB (with metadata filters)
            3. If use_mmr: rerank with MMR for diversity
            4. Return top n_results

        Args:
            query: Natural language query string.
            n_results: Number of results to return. Default: self.n_results.
            ticker: Filter by ticker symbol.
            doc_types: Filter by document types (e.g., ['10-K', '10-Q']).
            date_from: Filter documents after this date.
            date_to: Filter documents before this date.
            use_mmr: Whether to apply MMR reranking. Default True.

        Returns:
            List of SearchResult objects, sorted by relevance/diversity.

        Example:
            >>> results = await retriever.retrieve(
            ...     "What are Apple's main revenue drivers?",
            ...     ticker="AAPL",
            ...     doc_types=["10-K"],
            ...     n_results=5,
            ... )
        """
        if not query.strip():
            raise ValueError("query must not be empty.")

        k = n_results or self.n_results
        fetch_k = max(self.fetch_k, k * 3)

        logger.info(
            "Retriever | query=%r | ticker=%s | n_results=%d | mmr=%s",
            query[:50], ticker, k, use_mmr,
        )

        # Step 1: Embed query
        query_vector = await self.embedding_model.embed_query(query)

        # Step 2: Fetch candidates from ChromaDB
        candidates = self.vector_store.search(
            query_vector=query_vector,
            n_results=fetch_k,
            ticker=ticker,
            doc_types=doc_types,
            date_from=date_from,
            date_to=date_to,
        )

        if not candidates:
            logger.warning(
                "Retriever | no candidates found | query=%r | ticker=%s",
                query[:50], ticker,
            )
            return []

        # Step 3: MMR reranking or simple top-K
        if use_mmr and len(candidates) > k:
            results = self._mmr_rerank(candidates, query_vector, k)
        else:
            results = candidates[:k]

        logger.info(
            "Retriever | retrieved | n=%d | top_score=%.3f",
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    def _mmr_rerank(
        self,
        candidates: List[SearchResult],
        query_vector: np.ndarray,
        k: int,
    ) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance reranking.

        MMR selects documents that are both relevant to the query AND
        diverse from already-selected documents.

        Algorithm:
            selected = []
            for i in range(k):
                for each candidate d not yet selected:
                    relevance = cosine_similarity(query, d)
                    redundancy = max cosine_similarity(d, s) for s in selected
                    score = λ × relevance - (1-λ) × redundancy
                select d with highest score
                add d to selected

        Args:
            candidates: Pre-fetched candidate results (sorted by similarity).
            query_vector: Query embedding vector.
            k: Number of results to select.

        Returns:
            List of k SearchResult objects, diverse and relevant.
        """
        if len(candidates) <= k:
            return candidates

        # We need embeddings for MMR — re-embed candidate texts
        # Note: In production, store embeddings in ChromaDB and retrieve them
        # For now, use the similarity scores as a proxy for relevance
        # and compute diversity from text overlap (simpler, no extra embedding call)

        selected_indices: List[int] = []
        candidate_scores = [c.score for c in candidates]

        # Precompute pairwise text similarity using Jaccard on word sets
        # (approximation — avoids re-embedding all candidates)
        word_sets = [
            set(c.content.lower().split()) for c in candidates
        ]

        for _ in range(min(k, len(candidates))):
            best_idx = -1
            best_score = float("-inf")

            for i, candidate in enumerate(candidates):
                if i in selected_indices:
                    continue

                # Relevance: similarity to query
                relevance = candidate_scores[i]

                # Redundancy: max similarity to already-selected docs
                if not selected_indices:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        self._jaccard_similarity(word_sets[i], word_sets[j])
                        for j in selected_indices
                    )

                # MMR score
                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)

        result = [candidates[i] for i in selected_indices]
        logger.debug(
            "Retriever | MMR reranked | candidates=%d → selected=%d",
            len(candidates), len(result),
        )
        return result

    @staticmethod
    def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two word sets.

        Jaccard = |A ∩ B| / |A ∪ B|

        Used as a fast approximation for text similarity in MMR.
        Range: 0.0 (no overlap) to 1.0 (identical).

        Args:
            set_a: Word set of document A.
            set_b: Word set of document B.

        Returns:
            Jaccard similarity score.
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def build_context(
        self,
        results: List[SearchResult],
        max_context_chars: int = 8000,
    ) -> str:
        """Format search results into an LLM-ready context string.

        Creates a numbered list of relevant passages with source citations.
        The LLM uses this context to generate grounded, cited answers.

        Args:
            results: List of SearchResult objects from retrieve().
            max_context_chars: Maximum total context length. Default 8000.
                               Groq Llama 3.1 70B has 8K context window.

        Returns:
            Formatted context string with numbered passages and citations.

        Example:
            >>> context = retriever.build_context(results)
            >>> print(context)
            [1] Source: SEC EDGAR | Apple 10-K (2024-11-01) | Score: 0.892
            iPhone revenue declined 8% in Q3 2024, primarily due to...

            [2] Source: Reuters | Apple China Sales Drop (2024-10-15) | Score: 0.847
            Apple's market share in China fell to 15% as Huawei...
        """
        if not results:
            return "No relevant documents found."

        context_parts: List[str] = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            citation = result.to_citation()
            passage = f"[{i}] Source: {citation} | Score: {result.score:.3f}\n{result.content}"

            if total_chars + len(passage) > max_context_chars:
                # Truncate this passage to fit
                remaining = max_context_chars - total_chars
                if remaining > 200:  # Only add if meaningful content remains
                    passage = passage[:remaining] + "...[truncated]"
                    context_parts.append(passage)
                break

            context_parts.append(passage)
            total_chars += len(passage)

        context = "\n\n".join(context_parts)
        logger.debug(
            "Retriever | context built | passages=%d | chars=%d",
            len(context_parts), len(context),
        )
        return context

    def build_citations(self, results: List[SearchResult]) -> List[str]:
        """Extract citation strings from search results.

        Used to append source references to LLM answers.

        Args:
            results: List of SearchResult objects.

        Returns:
            List of citation strings.

        Example:
            >>> citations = retriever.build_citations(results)
            >>> print(citations[0])
            "[SEC EDGAR] Apple 10-K (2024-11-01) — https://www.sec.gov/..."
        """
        return [result.to_citation() for result in results]
