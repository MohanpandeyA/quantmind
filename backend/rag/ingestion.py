"""Ingestion pipeline orchestrator for the QuantMind RAG pipeline.

Coordinates the full document ingestion workflow:
    Load → Deduplicate → Chunk → Embed → Store

This is the single entry point for adding new financial documents
to the RAG system. Call ingest_ticker() to load all documents for
a stock ticker from all configured sources.

Optimization principles:
    - Async concurrent loading: all sources fetched simultaneously
    - Deduplication: SHA-256 hash prevents re-embedding same document
    - Batch embedding: 64 chunks per embedding call (not one-by-one)
    - Incremental ingestion: only process new documents, skip existing
    - Error isolation: one source failing doesn't stop others
    - Progress logging: detailed logs at each pipeline stage

Pipeline flow:
    1. Load documents from all sources concurrently (asyncio.gather)
    2. Deduplicate by doc_id hash (skip already-ingested documents)
    3. Chunk each document with RecursiveChunker (overlap preserved)
    4. Batch embed all chunks (sentence-transformers, local, free)
    5. Upsert into ChromaDB (persistent, survives restarts)
    6. Return IngestionReport with statistics
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config.logging_config import get_logger
from rag.chunker import Chunk, RecursiveChunker
from rag.embeddings import EmbeddingModel
from rag.sources.base_loader import BaseLoader, Document, LoaderError
from rag.sources.news_loader import NewsLoader
from rag.sources.pdf_loader import PDFLoader
from rag.sources.sec_loader import SECLoader
from rag.vector_store import VectorStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ingestion report
# ---------------------------------------------------------------------------

@dataclass
class IngestionReport:
    """Statistics from a single ingestion run.

    Attributes:
        ticker: Ticker that was ingested.
        docs_loaded: Total documents loaded from all sources.
        docs_skipped: Documents skipped (already in store).
        chunks_created: Total chunks created from documents.
        chunks_stored: Chunks successfully stored in ChromaDB.
        sources_used: List of source names that returned documents.
        errors: List of error messages from failed sources.
        duration_seconds: Total ingestion time.
    """

    ticker: str
    docs_loaded: int = 0
    docs_skipped: int = 0
    chunks_created: int = 0
    chunks_stored: int = 0
    sources_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Serialize report to dictionary.

        Returns:
            Dictionary of report fields.
        """
        return {
            "ticker": self.ticker,
            "docs_loaded": self.docs_loaded,
            "docs_skipped": self.docs_skipped,
            "chunks_created": self.chunks_created,
            "chunks_stored": self.chunks_stored,
            "sources_used": self.sources_used,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }

    def __repr__(self) -> str:
        return (
            f"IngestionReport(ticker={self.ticker!r}, "
            f"docs={self.docs_loaded}, chunks={self.chunks_stored}, "
            f"time={self.duration_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Orchestrates the full document ingestion pipeline.

    Coordinates loading, deduplication, chunking, embedding, and storage
    for financial documents from multiple sources.

    Attributes:
        loaders: List of document loaders (SEC, News, PDF).
        chunker: Text chunker for splitting documents.
        embedding_model: Model for converting text to vectors.
        vector_store: ChromaDB store for persistent storage.

    Example:
        >>> pipeline = IngestionPipeline.create_default()
        >>> report = await pipeline.ingest_ticker("AAPL")
        >>> print(report)
        IngestionReport(ticker='AAPL', docs=12, chunks=487, time=8.3s)

        >>> # Query after ingestion
        >>> results = await pipeline.retriever.retrieve(
        ...     "Why is AAPL revenue declining?",
        ...     ticker="AAPL",
        ... )
    """

    def __init__(
        self,
        loaders: List[BaseLoader],
        chunker: RecursiveChunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        """Initialize IngestionPipeline.

        Args:
            loaders: List of document loaders.
            chunker: Text chunker instance.
            embedding_model: Embedding model instance.
            vector_store: Vector store instance.
        """
        self.loaders = loaders
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    @classmethod
    def create_default(
        cls,
        persist_dir: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        sec_filing_types: Optional[List[str]] = None,
        max_sec_filings: int = 5,
        max_news_articles: int = 20,
    ) -> "IngestionPipeline":
        """Create a default pipeline with all standard sources.

        This is the recommended way to create a pipeline. Uses:
        - SECLoader (free, no key)
        - NewsLoader (NewsAPI + RSS, free tier)
        - PDFLoader (local files)
        - RecursiveChunker (1000 chars, 200 overlap)
        - EmbeddingModel (local sentence-transformers, free)
        - VectorStore (ChromaDB, local persistent)

        Args:
            persist_dir: ChromaDB persistence directory.
            chunk_size: Chunk size in characters. Default 1000.
            chunk_overlap: Overlap between chunks. Default 200.
            sec_filing_types: SEC filing types. Default ['10-K', '10-Q', '8-K'].
            max_sec_filings: Max filings per type. Default 5.
            max_news_articles: Max news articles. Default 20.

        Returns:
            Configured IngestionPipeline instance.

        Example:
            >>> pipeline = IngestionPipeline.create_default()
            >>> report = await pipeline.ingest_ticker("AAPL")
        """
        loaders: List[BaseLoader] = [
            SECLoader(
                filing_types=sec_filing_types or ["10-K", "10-Q", "8-K"],
                max_filings=max_sec_filings,
            ),
            NewsLoader(
                use_newsapi=True,
                use_rss=True,
                max_articles=max_news_articles,
            ),
            PDFLoader(),
        ]

        chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embedding_model = EmbeddingModel()
        vector_store = VectorStore(persist_dir=persist_dir)

        return cls(
            loaders=loaders,
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

    async def ingest_ticker(
        self,
        ticker: str,
        force_refresh: bool = False,
        **loader_kwargs: object,
    ) -> IngestionReport:
        """Ingest all documents for a ticker from all sources.

        This is the main entry point. Runs the full pipeline:
        Load → Deduplicate → Chunk → Embed → Store

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
            force_refresh: If True, re-ingest even if documents already exist.
                           If False (default), skip already-ingested documents.
            **loader_kwargs: Additional kwargs passed to all loaders.

        Returns:
            IngestionReport with statistics about the ingestion run.

        Example:
            >>> report = await pipeline.ingest_ticker("AAPL")
            >>> print(f"Stored {report.chunks_stored} chunks in {report.duration_seconds:.1f}s")
        """
        start_time = time.perf_counter()
        report = IngestionReport(ticker=ticker)

        logger.info(
            "IngestionPipeline | starting | ticker=%s | sources=%d | force=%s",
            ticker, len(self.loaders), force_refresh,
        )

        # Step 1: Load documents from all sources concurrently
        documents = await self._load_all_sources(ticker, report, **loader_kwargs)

        if not documents:
            logger.warning(
                "IngestionPipeline | no documents loaded | ticker=%s", ticker
            )
            report.duration_seconds = time.perf_counter() - start_time
            return report

        # Step 2: Deduplicate
        if not force_refresh:
            documents = self._deduplicate(documents, report)

        if not documents:
            logger.info(
                "IngestionPipeline | all documents already ingested | ticker=%s", ticker
            )
            report.duration_seconds = time.perf_counter() - start_time
            return report

        # Step 3: Chunk documents
        chunks = self._chunk_documents(documents, report)

        if not chunks:
            logger.warning(
                "IngestionPipeline | no chunks created | ticker=%s", ticker
            )
            report.duration_seconds = time.perf_counter() - start_time
            return report

        # Step 4: Embed chunks
        vectors = await self._embed_chunks(chunks, report)

        if vectors is None or len(vectors) == 0:
            logger.error(
                "IngestionPipeline | embedding failed | ticker=%s", ticker
            )
            report.duration_seconds = time.perf_counter() - start_time
            return report

        # Step 5: Store in ChromaDB
        self._store_chunks(chunks, vectors, report)

        report.duration_seconds = time.perf_counter() - start_time

        logger.info(
            "IngestionPipeline | complete | ticker=%s | docs=%d | chunks=%d | "
            "stored=%d | time=%.1fs",
            ticker,
            report.docs_loaded,
            report.chunks_created,
            report.chunks_stored,
            report.duration_seconds,
        )
        return report

    async def ingest_multiple_tickers(
        self,
        tickers: List[str],
        concurrent: bool = False,
    ) -> Dict[str, IngestionReport]:
        """Ingest documents for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            concurrent: If True, ingest all tickers concurrently.
                        If False (default), ingest sequentially.
                        Sequential is safer for rate limits.

        Returns:
            Dictionary mapping ticker → IngestionReport.

        Example:
            >>> reports = await pipeline.ingest_multiple_tickers(
            ...     ["AAPL", "MSFT", "GOOGL"]
            ... )
            >>> for ticker, report in reports.items():
            ...     print(f"{ticker}: {report.chunks_stored} chunks")
        """
        reports: Dict[str, IngestionReport] = {}

        if concurrent:
            tasks = [self.ingest_ticker(ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for ticker, result in zip(tickers, results):
                if isinstance(result, Exception):
                    logger.error(
                        "IngestionPipeline | ticker failed | ticker=%s | %s",
                        ticker, result,
                    )
                    reports[ticker] = IngestionReport(
                        ticker=ticker,
                        errors=[str(result)],
                    )
                else:
                    reports[ticker] = result  # type: ignore[assignment]
        else:
            for ticker in tickers:
                try:
                    report = await self.ingest_ticker(ticker)
                    reports[ticker] = report
                except Exception as e:
                    logger.error(
                        "IngestionPipeline | ticker failed | ticker=%s | %s", ticker, e
                    )
                    reports[ticker] = IngestionReport(ticker=ticker, errors=[str(e)])

        return reports

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    async def _load_all_sources(
        self,
        ticker: str,
        report: IngestionReport,
        **kwargs: object,
    ) -> List[Document]:
        """Load documents from all sources concurrently.

        Args:
            ticker: Ticker symbol.
            report: IngestionReport to update.
            **kwargs: Additional kwargs for loaders.

        Returns:
            Combined list of documents from all sources.
        """
        tasks = [loader.load(ticker, **kwargs) for loader in self.loaders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_documents: List[Document] = []
        for loader, result in zip(self.loaders, results):
            if isinstance(result, Exception):
                error_msg = f"{loader.get_source_name()}: {result}"
                report.errors.append(error_msg)
                logger.warning(
                    "IngestionPipeline | loader failed | source=%s | %s",
                    loader.get_source_name(), result,
                )
            elif isinstance(result, list):
                if result:
                    report.sources_used.append(loader.get_source_name())
                all_documents.extend(result)

        report.docs_loaded = len(all_documents)
        logger.info(
            "IngestionPipeline | loaded | ticker=%s | docs=%d | sources=%s",
            ticker, len(all_documents), report.sources_used,
        )
        return all_documents

    def _deduplicate(
        self,
        documents: List[Document],
        report: IngestionReport,
    ) -> List[Document]:
        """Remove documents that are already in the vector store.

        Uses doc_id (SHA-256 hash) to check if a document has been
        previously ingested. Skips re-embedding of existing documents.

        Args:
            documents: Documents to deduplicate.
            report: IngestionReport to update.

        Returns:
            List of new documents not yet in the store.
        """
        new_docs: List[Document] = []
        skipped = 0

        for doc in documents:
            # Check if any chunk from this document already exists
            # Use first chunk ID as proxy: {doc_id}_chunk_0000
            first_chunk_id = f"{doc.doc_id}_chunk_0000"
            if self.vector_store.chunk_exists(first_chunk_id):
                skipped += 1
                logger.debug(
                    "IngestionPipeline | skipping existing doc | doc_id=%s", doc.doc_id
                )
            else:
                new_docs.append(doc)

        report.docs_skipped = skipped
        logger.info(
            "IngestionPipeline | deduplication | new=%d | skipped=%d",
            len(new_docs), skipped,
        )
        return new_docs

    def _chunk_documents(
        self,
        documents: List[Document],
        report: IngestionReport,
    ) -> List[Chunk]:
        """Split documents into overlapping chunks.

        Args:
            documents: Documents to chunk.
            report: IngestionReport to update.

        Returns:
            List of all chunks from all documents.
        """
        chunks = self.chunker.chunk_documents(documents)
        report.chunks_created = len(chunks)
        logger.info(
            "IngestionPipeline | chunked | docs=%d → chunks=%d",
            len(documents), len(chunks),
        )
        return chunks

    async def _embed_chunks(
        self,
        chunks: List[Chunk],
        report: IngestionReport,
    ) -> Optional[np.ndarray]:
        """Embed all chunks into vectors.

        Args:
            chunks: Chunks to embed.
            report: IngestionReport to update.

        Returns:
            NumPy array of shape (len(chunks), dimensions), or None on failure.
        """
        try:
            vectors = await self.embedding_model.embed_chunks(chunks)
            logger.info(
                "IngestionPipeline | embedded | chunks=%d | shape=%s",
                len(chunks), vectors.shape,
            )
            return vectors
        except Exception as e:
            error_msg = f"Embedding failed: {e}"
            report.errors.append(error_msg)
            logger.error("IngestionPipeline | embedding error | %s", e, exc_info=True)
            return None

    def _store_chunks(
        self,
        chunks: List[Chunk],
        vectors: np.ndarray,
        report: IngestionReport,
    ) -> None:
        """Store chunks and vectors in ChromaDB.

        Args:
            chunks: Chunks to store.
            vectors: Corresponding embedding vectors.
            report: IngestionReport to update.
        """
        try:
            n_stored = self.vector_store.add_chunks(chunks, vectors)
            report.chunks_stored = n_stored
            logger.info(
                "IngestionPipeline | stored | chunks=%d", n_stored
            )
        except Exception as e:
            error_msg = f"Storage failed: {e}"
            report.errors.append(error_msg)
            logger.error("IngestionPipeline | storage error | %s", e, exc_info=True)
