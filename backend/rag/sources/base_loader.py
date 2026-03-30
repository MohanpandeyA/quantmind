"""Abstract base class and core data models for all document loaders.

Every document source (SEC, News, PDF) must inherit from BaseLoader
and implement load(). This enforces a consistent interface so the
Ingestion pipeline can process any source without knowing its internals.

Design principles applied:
    - Open/Closed: Add new sources without changing Ingestion pipeline
    - Single Responsibility: Each loader handles one source type
    - Deduplication: SHA-256 doc_id prevents re-embedding same document
    - Typed metadata: Pydantic model validates all document metadata
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Document type enum
# ---------------------------------------------------------------------------

class DocType(str, Enum):
    """Supported financial document types.

    Used for metadata filtering in ChromaDB — allows queries like
    "only search 10-K filings from the last 2 years".
    """

    SEC_10K = "10-K"          # Annual report — comprehensive financials
    SEC_10Q = "10-Q"          # Quarterly report — interim financials
    SEC_8K = "8-K"            # Material event — M&A, CEO change, etc.
    NEWS = "news"             # Financial news article
    RSS = "rss"               # RSS feed article
    PDF = "pdf"               # Research paper, analyst report
    EARNINGS = "earnings"     # Earnings call transcript
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    """Structured metadata for every ingested document.

    Stored alongside embeddings in ChromaDB. Enables:
    - Filtered search: "only AAPL documents from 2024"
    - Source citation: "Based on Apple 10-K (2024-01-15)"
    - Deduplication: doc_id prevents re-ingesting same document

    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        source: Human-readable source name (e.g., 'SEC EDGAR', 'Reuters').
        doc_type: Document type enum.
        date: Publication/filing date (ISO format string for ChromaDB compat).
        url: Source URL for citation.
        title: Document title or headline.
        filing_period: For SEC filings, the period covered (e.g., 'Q3 2024').
        page_number: For PDFs, the page number of this chunk.
        extra: Additional source-specific metadata.
    """

    ticker: str
    source: str
    doc_type: DocType
    date: str                          # ISO format: "2024-01-15"
    url: str = ""
    title: str = ""
    filing_period: str = ""
    page_number: int = 0
    extra: Dict[str, str] = field(default_factory=dict)

    def to_chroma_dict(self) -> Dict[str, str]:
        """Serialize metadata to ChromaDB-compatible flat dictionary.

        ChromaDB requires all metadata values to be str, int, float, or bool.
        Nested dicts are not supported — flatten everything.

        Returns:
            Flat dictionary of metadata key-value pairs.

        Example:
            >>> meta = DocumentMetadata(ticker="AAPL", source="SEC", ...)
            >>> meta.to_chroma_dict()
            {"ticker": "AAPL", "source": "SEC", "doc_type": "10-K", ...}
        """
        return {
            "ticker": self.ticker,
            "source": self.source,
            "doc_type": self.doc_type.value,
            "date": self.date,
            "url": self.url,
            "title": self.title[:500],  # ChromaDB has metadata value length limits
            "filing_period": self.filing_period,
            "page_number": str(self.page_number),
            **{f"extra_{k}": str(v)[:200] for k, v in self.extra.items()},
        }

    @classmethod
    def from_chroma_dict(cls, d: Dict[str, str]) -> DocumentMetadata:
        """Reconstruct DocumentMetadata from a ChromaDB metadata dictionary.

        Args:
            d: Flat metadata dictionary from ChromaDB.

        Returns:
            DocumentMetadata instance.
        """
        extra = {
            k[6:]: v for k, v in d.items() if k.startswith("extra_")
        }
        return cls(
            ticker=d.get("ticker", ""),
            source=d.get("source", ""),
            doc_type=DocType(d.get("doc_type", "unknown")),
            date=d.get("date", ""),
            url=d.get("url", ""),
            title=d.get("title", ""),
            filing_period=d.get("filing_period", ""),
            page_number=int(d.get("page_number", "0")),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Core Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single financial document ready for chunking and embedding.

    The fundamental unit of the RAG pipeline. Every loader produces
    a list of Documents. The Chunker splits them into smaller chunks.
    The EmbeddingModel converts chunks to vectors.

    Attributes:
        content: Raw text content of the document.
        metadata: Structured metadata (ticker, source, date, etc.).
        doc_id: SHA-256 hash of content for deduplication.
                Computed automatically if not provided.

    Example:
        >>> doc = Document(
        ...     content="Apple reported revenue of $89.5B in Q3 2024...",
        ...     metadata=DocumentMetadata(
        ...         ticker="AAPL",
        ...         source="SEC EDGAR",
        ...         doc_type=DocType.SEC_10Q,
        ...         date="2024-08-01",
        ...     )
        ... )
        >>> doc.doc_id  # SHA-256 hash — unique fingerprint
        "a3f2b1c4..."
    """

    content: str
    metadata: DocumentMetadata
    doc_id: str = field(default="")

    def __post_init__(self) -> None:
        """Auto-compute doc_id from content hash if not provided."""
        if not self.doc_id:
            self.doc_id = self._compute_hash(self.content)

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA-256 hash of document content.

        Used for deduplication — if the same document is ingested twice,
        the hash matches and it's skipped. Zero duplicate embeddings.

        Args:
            content: Document text content.

        Returns:
            First 16 characters of SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def is_empty(self) -> bool:
        """Return True if document has no meaningful content.

        Args:
            None

        Returns:
            True if content is empty or whitespace-only.
        """
        return not self.content or not self.content.strip()

    def word_count(self) -> int:
        """Return approximate word count of the document.

        Returns:
            Number of whitespace-separated tokens.
        """
        return len(self.content.split())

    def __repr__(self) -> str:
        return (
            f"Document(ticker={self.metadata.ticker!r}, "
            f"type={self.metadata.doc_type.value!r}, "
            f"date={self.metadata.date!r}, "
            f"words={self.word_count()}, "
            f"id={self.doc_id!r})"
        )


# ---------------------------------------------------------------------------
# Abstract base loader
# ---------------------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for all document source loaders.

    Subclasses must implement:
        - load(): fetch documents for a given ticker/query
        - get_source_name(): return human-readable source name

    Subclasses may override:
        - validate_query(): custom query validation

    The Ingestion pipeline calls loader.load(ticker) for each source
    without knowing whether it's talking to SEC, NewsAPI, or a PDF.

    Example:
        >>> class MyLoader(BaseLoader):
        ...     def get_source_name(self) -> str:
        ...         return "MySource"
        ...     async def load(self, ticker: str, **kwargs) -> List[Document]:
        ...         return [Document(content="...", metadata=...)]
    """

    @abstractmethod
    def get_source_name(self) -> str:
        """Return human-readable name of this data source.

        Returns:
            Source name string (e.g., "SEC EDGAR", "NewsAPI").
        """
        ...

    @abstractmethod
    async def load(self, ticker: str, **kwargs: object) -> List[Document]:
        """Fetch documents for a given ticker symbol.

        This is an async method — all loaders support concurrent fetching.
        The Ingestion pipeline uses asyncio.gather() to fetch all sources
        simultaneously, not sequentially.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
            **kwargs: Source-specific parameters (e.g., max_results, date_from).

        Returns:
            List of Document objects ready for chunking.
            Returns empty list if no documents found (never raises on empty).

        Raises:
            LoaderError: If the source is unreachable or returns invalid data.
        """
        ...

    def validate_query(self, ticker: str) -> None:
        """Validate ticker symbol format.

        Args:
            ticker: Ticker to validate.

        Raises:
            ValueError: If ticker is empty or too long.
        """
        if not ticker or not ticker.strip():
            raise ValueError("ticker must not be empty.")
        if len(ticker) > 20:
            raise ValueError(
                f"ticker '{ticker}' is too long (max 20 chars). "
                "Use standard ticker symbols like 'AAPL', 'RELIANCE.NS'."
            )

    def _log_load_result(self, ticker: str, docs: List[Document]) -> None:
        """Log the result of a load() call.

        Args:
            ticker: Ticker that was loaded.
            docs: Documents returned.
        """
        total_words = sum(d.word_count() for d in docs)
        logger.info(
            "%s | loaded | ticker=%s | docs=%d | total_words=%d",
            self.get_source_name(), ticker, len(docs), total_words,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.get_source_name()!r})"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LoaderError(Exception):
    """Raised when a document loader fails to fetch data.

    Attributes:
        source: Name of the loader that failed.
        ticker: Ticker that was being loaded.
        message: Human-readable error description.
    """

    def __init__(self, source: str, ticker: str, message: str) -> None:
        self.source = source
        self.ticker = ticker
        self.message = message
        super().__init__(f"[{source}] Failed to load '{ticker}': {message}")


class RateLimitError(LoaderError):
    """Raised when an API rate limit is exceeded.

    The Ingestion pipeline catches this and implements exponential backoff.
    """
    pass
