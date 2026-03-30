"""ChromaDB vector store for the QuantMind RAG pipeline.

Stores document chunk embeddings persistently on disk.
Supports metadata filtering for efficient, targeted retrieval.

Why ChromaDB:
    - Local, persistent — no cloud, no API key, no cost
    - Survives restarts — embeddings stored on disk, not re-computed
    - Metadata filtering — search only AAPL docs from last 6 months
    - Fast — uses HNSW index for approximate nearest neighbor search
    - Simple API — add, query, delete in a few lines

Optimization principles:
    - Persistent storage: embeddings survive restarts (no re-embedding)
    - Metadata filtering: reduces search space before vector comparison
      (e.g., filter by ticker reduces 100K chunks to ~500 → 200× faster)
    - Batch upsert: add 100 chunks at once, not one-by-one
    - Deduplication: upsert (not insert) — same chunk_id = update, not duplicate
    - Collection per use case: separate collections for different data types
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from config.logging_config import get_logger
from config.settings import settings
from rag.chunker import Chunk
from rag.sources.base_loader import DocumentMetadata

logger = get_logger(__name__)

# ChromaDB collection name
COLLECTION_NAME = "quantmind_financial_docs"

# Batch size for upsert operations
UPSERT_BATCH_SIZE = 100


class VectorStore:
    """ChromaDB-backed vector store for financial document chunks.

    Provides:
        - add_chunks(): Store embeddings with metadata
        - search(): Semantic search with optional metadata filters
        - delete_by_ticker(): Remove all documents for a ticker
        - get_stats(): Collection statistics

    Attributes:
        persist_dir: Directory where ChromaDB stores its data.
        collection_name: Name of the ChromaDB collection.

    Example:
        >>> store = VectorStore()
        >>> await store.add_chunks(chunks, vectors)
        >>> results = await store.search(query_vector, ticker="AAPL", n_results=5)
        >>> print(results[0].metadata.title)
        "Apple 10-K (2024-11-01)"
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """Initialize VectorStore.

        Args:
            persist_dir: Directory for ChromaDB persistence.
                         Defaults to settings.chroma_persist_dir.
            collection_name: ChromaDB collection name.
        """
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

        # Ensure persist directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

    def _get_collection(self) -> object:
        """Lazy-initialize ChromaDB client and collection.

        Returns:
            ChromaDB collection object.

        Raises:
            ImportError: If chromadb is not installed.
        """
        if self._collection is not None:
            return self._collection

        try:
            import chromadb  # type: ignore[import]
            from chromadb.config import Settings as ChromaSettings  # type: ignore[import]
        except ImportError:
            logger.error(
                "VectorStore | chromadb not installed. Run: pip install chromadb"
            )
            raise

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            "VectorStore | initialized | dir=%s | collection=%s | count=%d",
            self.persist_dir,
            self.collection_name,
            self._collection.count(),
        )
        return self._collection

    def add_chunks(
        self,
        chunks: List[Chunk],
        vectors: np.ndarray,
    ) -> int:
        """Store chunk embeddings in ChromaDB.

        Uses upsert (not insert) — if a chunk_id already exists,
        it's updated rather than duplicated. This enables safe re-ingestion.

        Processes in batches of UPSERT_BATCH_SIZE for efficiency.

        Args:
            chunks: List of Chunk objects.
            vectors: NumPy array of shape (len(chunks), dimensions).

        Returns:
            Number of chunks successfully stored.

        Raises:
            ValueError: If chunks and vectors have different lengths.

        Example:
            >>> n_stored = store.add_chunks(chunks, vectors)
            >>> print(f"Stored {n_stored} chunks")
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have equal length."
            )

        if not chunks:
            return 0

        collection = self._get_collection()
        total_stored = 0

        # Process in batches
        for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch_chunks = chunks[i: i + UPSERT_BATCH_SIZE]
            batch_vectors = vectors[i: i + UPSERT_BATCH_SIZE]

            ids = [chunk.chunk_id for chunk in batch_chunks]
            documents = [chunk.content for chunk in batch_chunks]
            metadatas = [chunk.metadata.to_chroma_dict() for chunk in batch_chunks]
            embeddings = batch_vectors.tolist()

            try:
                collection.upsert(  # type: ignore[union-attr]
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                total_stored += len(batch_chunks)
            except Exception as e:
                logger.error(
                    "VectorStore | upsert failed | batch=%d | %s", i // UPSERT_BATCH_SIZE, e
                )

        logger.info(
            "VectorStore | stored | chunks=%d | total_in_collection=%d",
            total_stored,
            collection.count(),  # type: ignore[union-attr]
        )
        return total_stored

    def search(
        self,
        query_vector: np.ndarray,
        n_results: int = 10,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Semantic search with optional metadata filtering.

        Metadata filtering happens BEFORE vector comparison — this reduces
        the search space dramatically (e.g., 100K → 500 chunks for one ticker).

        Args:
            query_vector: Query embedding vector (1D array).
            n_results: Number of results to return. Default 10.
            ticker: Filter by ticker symbol (e.g., 'AAPL').
            doc_types: Filter by document types (e.g., ['10-K', '10-Q']).
            date_from: Filter documents after this date (ISO format 'YYYY-MM-DD').
            date_to: Filter documents before this date (ISO format 'YYYY-MM-DD').
            where: Raw ChromaDB where clause (overrides other filters).

        Returns:
            List of SearchResult objects, sorted by relevance (highest first).

        Example:
            >>> results = store.search(
            ...     query_vector=q_vec,
            ...     ticker="AAPL",
            ...     doc_types=["10-K", "10-Q"],
            ...     date_from="2024-01-01",
            ...     n_results=5,
            ... )
            >>> for r in results:
            ...     print(f"{r.score:.3f} | {r.metadata.title}")
        """
        collection = self._get_collection()

        if collection.count() == 0:  # type: ignore[union-attr]
            logger.warning("VectorStore | collection is empty | no results")
            return []

        # Build metadata filter
        filter_clause = where or self._build_filter(ticker, doc_types, date_from, date_to)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector.tolist()],
            "n_results": min(n_results, collection.count()),  # type: ignore[union-attr]
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_clause:
            query_kwargs["where"] = filter_clause

        try:
            results = collection.query(**query_kwargs)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("VectorStore | search failed | %s", e)
            return []

        # Parse results
        search_results: List[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 = identical, -1 = opposite
            score = 1.0 - (dist / 2.0)

            search_results.append(SearchResult(
                content=doc,
                metadata=DocumentMetadata.from_chroma_dict(meta),
                score=score,
                distance=dist,
            ))

        logger.debug(
            "VectorStore | search | ticker=%s | n_results=%d | top_score=%.3f",
            ticker, len(search_results),
            search_results[0].score if search_results else 0.0,
        )
        return search_results

    def delete_by_ticker(self, ticker: str) -> int:
        """Delete all documents for a given ticker.

        Useful for refreshing stale data — delete old docs, re-ingest new ones.

        Args:
            ticker: Ticker symbol to delete.

        Returns:
            Number of chunks deleted.
        """
        collection = self._get_collection()

        try:
            # Get all IDs for this ticker
            results = collection.get(  # type: ignore[union-attr]
                where={"ticker": {"$eq": ticker.upper()}},
                include=[],
            )
            ids_to_delete = results.get("ids", [])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)  # type: ignore[union-attr]
                logger.info(
                    "VectorStore | deleted | ticker=%s | chunks=%d",
                    ticker, len(ids_to_delete),
                )
            return len(ids_to_delete)

        except Exception as e:
            logger.error("VectorStore | delete failed | ticker=%s | %s", ticker, e)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics.

        Returns:
            Dictionary with total_chunks, collection_name, persist_dir.
        """
        try:
            collection = self._get_collection()
            count = collection.count()  # type: ignore[union-attr]
        except Exception:
            count = 0

        return {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "total_chunks": count,
        }

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk already exists in the store.

        Used for deduplication — skip re-embedding if chunk already stored.

        Args:
            chunk_id: Chunk ID to check.

        Returns:
            True if chunk exists.
        """
        collection = self._get_collection()
        try:
            result = collection.get(ids=[chunk_id], include=[])  # type: ignore[union-attr]
            return len(result.get("ids", [])) > 0
        except Exception:
            return False

    @staticmethod
    def _build_filter(
        ticker: Optional[str],
        doc_types: Optional[List[str]],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Build a ChromaDB where clause from filter parameters.

        ChromaDB uses MongoDB-style query operators:
            $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or

        Args:
            ticker: Ticker filter.
            doc_types: Document type filter.
            date_from: Start date filter.
            date_to: End date filter.

        Returns:
            ChromaDB where clause dict, or None if no filters.
        """
        conditions: List[Dict[str, Any]] = []

        if ticker:
            conditions.append({"ticker": {"$eq": ticker.upper()}})

        if doc_types:
            if len(doc_types) == 1:
                conditions.append({"doc_type": {"$eq": doc_types[0]}})
            else:
                conditions.append({"doc_type": {"$in": doc_types}})

        if date_from:
            conditions.append({"date": {"$gte": date_from}})

        if date_to:
            conditions.append({"date": {"$lte": date_to}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


class SearchResult:
    """A single result from a vector store search.

    Attributes:
        content: Text content of the matching chunk.
        metadata: Document metadata (ticker, source, date, url, title).
        score: Similarity score (0.0 to 1.0, higher = more similar).
        distance: Raw ChromaDB cosine distance (lower = more similar).
    """

    def __init__(
        self,
        content: str,
        metadata: DocumentMetadata,
        score: float,
        distance: float,
    ) -> None:
        self.content = content
        self.metadata = metadata
        self.score = score
        self.distance = distance

    def to_citation(self) -> str:
        """Format this result as a citation string.

        Returns:
            Human-readable citation string.

        Example:
            >>> result.to_citation()
            "[SEC EDGAR] Apple 10-K (2024-11-01) — https://www.sec.gov/..."
        """
        parts = [f"[{self.metadata.source}]"]
        if self.metadata.title:
            parts.append(self.metadata.title)
        if self.metadata.date:
            parts.append(f"({self.metadata.date})")
        if self.metadata.url:
            parts.append(f"— {self.metadata.url}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.score:.3f}, "
            f"ticker={self.metadata.ticker!r}, "
            f"source={self.metadata.source!r}, "
            f"date={self.metadata.date!r})"
        )
