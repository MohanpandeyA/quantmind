"""Embedding model wrapper for the QuantMind RAG pipeline.

Converts text chunks into dense vector representations for semantic search.

Free tier (default):
    sentence-transformers with 'all-MiniLM-L6-v2' model
    - 384-dimensional vectors
    - Runs locally on CPU — no API key, no cost, no rate limits
    - ~50ms per batch of 100 chunks on modern CPU

Paid upgrade:
    OpenAI 'text-embedding-3-small' (1536 dims, much better quality)
    Configured via OPENAI_API_KEY in .env

Optimization principles:
    - Batch embedding: embed 100 chunks in one call (10-50× faster than one-by-one)
    - Model caching: load model once at startup, reuse for all embeddings
    - Lazy initialization: model loaded only when first needed (not at import)
    - Normalize vectors: L2 normalization for cosine similarity (ChromaDB default)
    - Async wrapper: runs CPU-bound embedding in thread pool
"""

from __future__ import annotations

import asyncio
import threading
from typing import List, Optional

import numpy as np

from config.logging_config import get_logger
from config.settings import settings
from rag.chunker import Chunk

logger = get_logger(__name__)

# Default free model
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64  # Chunks per embedding batch

# Module-level singleton + lock to prevent race condition when multiple
# parallel requests (e.g. Compare tab with 5 tickers) each try to load
# SentenceTransformer simultaneously. torch 2.8.0 raises:
#   "Cannot copy out of meta tensor" when concurrent .to(device) calls happen.
_SENTENCE_TRANSFORMER_SINGLETON = None
_SENTENCE_TRANSFORMER_LOCK = threading.Lock()


class EmbeddingModel:
    """Wrapper around sentence-transformers for text embedding.

    Supports two backends:
        - Local (free): sentence-transformers (runs on CPU)
        - OpenAI (paid): text-embedding-3-small (better quality)

    The backend is selected automatically based on available API keys.
    If OPENAI_API_KEY is set, uses OpenAI. Otherwise uses local model.

    Attributes:
        model_name: Name of the embedding model.
        batch_size: Number of texts to embed per batch.
        use_openai: Whether to use OpenAI embeddings.
        dimensions: Output vector dimensions.

    Example:
        >>> model = EmbeddingModel()
        >>> vectors = await model.embed_chunks(chunks)
        >>> print(vectors.shape)
        (47, 384)  # 47 chunks, 384 dimensions
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize EmbeddingModel.

        Args:
            model_name: Model name. Auto-selected if None.
            batch_size: Chunks per embedding batch. Default 64.
        """
        self.use_openai: bool = bool(settings.openai_api_key)
        self.batch_size = batch_size
        self._model = None  # Lazy initialization

        if self.use_openai:
            self.model_name = model_name or "text-embedding-3-small"
            self.dimensions = 1536
            logger.info("EmbeddingModel | using OpenAI | model=%s", self.model_name)
        else:
            self.model_name = model_name or DEFAULT_MODEL
            self.dimensions = 384
            logger.info(
                "EmbeddingModel | using local sentence-transformers | model=%s",
                self.model_name,
            )

    def _load_model(self) -> object:
        """Lazy-load the embedding model.

        Called on first use — not at import time. This avoids slow startup
        when the model isn't needed (e.g., during unit tests).

        Returns:
            Loaded model object.

        Raises:
            ImportError: If required package is not installed.
        """
        if self._model is not None:
            return self._model

        if self.use_openai:
            try:
                from openai import OpenAI  # type: ignore[import]
                self._model = OpenAI(api_key=settings.openai_api_key)
                logger.info("EmbeddingModel | OpenAI client loaded")
            except ImportError:
                logger.error(
                    "EmbeddingModel | openai not installed. Run: pip install openai"
                )
                raise
        else:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                global _SENTENCE_TRANSFORMER_SINGLETON
                # Double-checked locking: fast path (no lock) if already loaded
                if _SENTENCE_TRANSFORMER_SINGLETON is None:
                    with _SENTENCE_TRANSFORMER_LOCK:
                        # Re-check inside lock (another thread may have loaded it)
                        if _SENTENCE_TRANSFORMER_SINGLETON is None:
                            _SENTENCE_TRANSFORMER_SINGLETON = SentenceTransformer(self.model_name)
                self._model = _SENTENCE_TRANSFORMER_SINGLETON
                logger.info(
                    "EmbeddingModel | sentence-transformers loaded | model=%s",
                    self.model_name,
                )
            except ImportError:
                logger.error(
                    "EmbeddingModel | sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise

        return self._model

    async def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Embed a list of Chunks into vectors. Async, batch-optimized.

        Processes chunks in batches of batch_size. Runs CPU-bound embedding
        in a thread pool to avoid blocking the async event loop.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            NumPy array of shape (len(chunks), dimensions).
            Each row is the embedding vector for the corresponding chunk.

        Raises:
            RuntimeError: If embedding fails.

        Example:
            >>> vectors = await model.embed_chunks(chunks)
            >>> vectors.shape
            (47, 384)
            >>> # Cosine similarity between first two chunks:
            >>> similarity = np.dot(vectors[0], vectors[1])
        """
        if not chunks:
            return np.empty((0, self.dimensions), dtype=np.float32)

        texts = [chunk.content for chunk in chunks]
        return await self.embed_texts(texts)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of raw text strings into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimensions).

        Example:
            >>> vectors = await model.embed_texts(["Apple revenue grew", "iPhone sales"])
            >>> vectors.shape
            (2, 384)
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        loop = asyncio.get_event_loop()

        if self.use_openai:
            vectors = await loop.run_in_executor(
                None, self._embed_openai_batch, texts
            )
        else:
            vectors = await loop.run_in_executor(
                None, self._embed_local_batch, texts
            )

        logger.debug(
            "EmbeddingModel | embedded | n=%d | shape=%s", len(texts), vectors.shape
        )
        return vectors

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string for retrieval.

        Queries are embedded the same way as documents — this ensures
        the query vector is in the same space as document vectors.

        Args:
            query: Query text string.

        Returns:
            1D NumPy array of shape (dimensions,).

        Example:
            >>> q_vec = await model.embed_query("Why is AAPL revenue declining?")
            >>> q_vec.shape
            (384,)
        """
        vectors = await self.embed_texts([query])
        return vectors[0]

    def _embed_local_batch(self, texts: List[str]) -> np.ndarray:
        """Embed texts using local sentence-transformers model.

        Processes in batches of batch_size. Normalizes vectors for
        cosine similarity.

        Args:
            texts: List of text strings.

        Returns:
            Normalized float32 NumPy array of shape (n, dimensions).
        """
        model = self._load_model()
        all_vectors: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            # encode() returns numpy array, normalize_embeddings=True for cosine sim
            batch_vectors = model.encode(  # type: ignore[union-attr]
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            all_vectors.append(batch_vectors)

        return np.vstack(all_vectors).astype(np.float32)

    def _embed_openai_batch(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI embeddings API.

        Processes in batches of batch_size (OpenAI max: 2048 per request).

        Args:
            texts: List of text strings.

        Returns:
            Float32 NumPy array of shape (n, dimensions).
        """
        client = self._load_model()
        all_vectors: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            # Truncate texts to avoid token limit errors
            batch = [t[:8000] for t in batch]

            response = client.embeddings.create(  # type: ignore[union-attr]
                model=self.model_name,
                input=batch,
            )
            batch_vectors = np.array(
                [item.embedding for item in response.data],
                dtype=np.float32,
            )
            all_vectors.append(batch_vectors)

        return np.vstack(all_vectors)

    def get_model_info(self) -> dict[str, object]:
        """Return model configuration info.

        Returns:
            Dictionary with model name, dimensions, and backend.
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "backend": "openai" if self.use_openai else "sentence-transformers",
            "batch_size": self.batch_size,
        }
