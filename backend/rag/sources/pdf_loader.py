"""PDF document loader for QuantMind RAG pipeline.

Loads local PDF files (research papers, analyst reports, earnings transcripts).
Uses pypdf for text extraction — pure Python, no external dependencies.

Optimization principles:
    - Page-level chunking: each PDF page becomes a separate Document
      (better granularity for citation — "Based on page 12 of report")
    - Lazy loading: pages extracted on demand, not all at once
    - Metadata extraction: title, author, creation date from PDF metadata
    - Error resilience: corrupted pages are skipped, not fatal
    - Async wrapper: runs synchronous pypdf in thread pool
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List, Optional

from config.logging_config import get_logger
from rag.sources.base_loader import (
    BaseLoader,
    Document,
    DocumentMetadata,
    DocType,
    LoaderError,
)

logger = get_logger(__name__)

# Minimum characters per page to be worth embedding
MIN_PAGE_CHARS = 100


class PDFLoader(BaseLoader):
    """Loads PDF documents from local file paths.

    Each page of the PDF becomes a separate Document with page number
    in metadata — enabling precise citations like "Based on page 5 of
    Goldman Sachs AAPL Research Report (2024-03-15)".

    Attributes:
        pdf_dir: Directory to scan for PDF files. Default: './data/pdfs'.

    Example:
        >>> loader = PDFLoader(pdf_dir="./data/pdfs")
        >>> docs = await loader.load("AAPL")
        >>> # Returns docs from all PDFs in ./data/pdfs/AAPL/
        >>> print(docs[0].metadata.page_number)
        1
    """

    def __init__(self, pdf_dir: str = "./data/pdfs") -> None:
        """Initialize PDFLoader.

        Args:
            pdf_dir: Root directory for PDF files.
                     Structure: {pdf_dir}/{ticker}/*.pdf
                     or {pdf_dir}/*.pdf (all tickers)
        """
        self.pdf_dir = Path(pdf_dir)

    def get_source_name(self) -> str:
        """Return source name.

        Returns:
            'PDF'
        """
        return "PDF"

    async def load(self, ticker: str, **kwargs: object) -> List[Document]:
        """Load all PDF files for a given ticker.

        Looks for PDFs in:
            1. {pdf_dir}/{ticker}/*.pdf  (ticker-specific folder)
            2. {pdf_dir}/*.pdf           (root folder, all tickers)

        Args:
            ticker: Stock ticker symbol.
            **kwargs: Optional overrides:
                - pdf_path (str): Load a specific PDF file path.
                - max_pages (int): Max pages per PDF. Default: 50.

        Returns:
            List of Document objects, one per PDF page.
        """
        self.validate_query(ticker)
        max_pages = int(kwargs.get("max_pages", 50))
        specific_path = kwargs.get("pdf_path")

        if specific_path:
            # Load a specific file
            pdf_files = [Path(str(specific_path))]
        else:
            # Scan directories for PDFs
            pdf_files = self._find_pdf_files(ticker)

        if not pdf_files:
            logger.info(
                "PDFLoader | no PDFs found | ticker=%s | dir=%s",
                ticker, self.pdf_dir,
            )
            return []

        logger.info(
            "PDFLoader | loading | ticker=%s | files=%d", ticker, len(pdf_files)
        )

        # Load all PDFs concurrently
        tasks = [
            self._load_pdf(pdf_path, ticker, max_pages)
            for pdf_path in pdf_files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: List[Document] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("PDFLoader | file failed | %s", result)
                continue
            if isinstance(result, list):
                documents.extend(result)

        self._log_load_result(ticker, documents)
        return documents

    async def load_file(self, pdf_path: str, ticker: str) -> List[Document]:
        """Load a single PDF file directly.

        Convenience method for loading a specific PDF without
        directory scanning.

        Args:
            pdf_path: Full path to the PDF file.
            ticker: Ticker symbol to associate with this document.

        Returns:
            List of Document objects, one per page.

        Raises:
            LoaderError: If file does not exist or cannot be read.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise LoaderError(
                self.get_source_name(), ticker,
                f"PDF file not found: {pdf_path}"
            )
        return await self._load_pdf(path, ticker, max_pages=200)

    def _find_pdf_files(self, ticker: str) -> List[Path]:
        """Find all PDF files for a ticker in the pdf_dir.

        Args:
            ticker: Ticker symbol.

        Returns:
            List of PDF file paths.
        """
        pdf_files: List[Path] = []

        # Check ticker-specific subdirectory
        ticker_dir = self.pdf_dir / ticker.upper()
        if ticker_dir.exists():
            pdf_files.extend(ticker_dir.glob("*.pdf"))
            pdf_files.extend(ticker_dir.glob("*.PDF"))

        # Check root directory
        if self.pdf_dir.exists():
            pdf_files.extend(self.pdf_dir.glob("*.pdf"))
            pdf_files.extend(self.pdf_dir.glob("*.PDF"))

        # Deduplicate
        return list(set(pdf_files))

    async def _load_pdf(
        self, pdf_path: Path, ticker: str, max_pages: int
    ) -> List[Document]:
        """Load a single PDF file and extract text page by page.

        Runs pypdf synchronously in a thread pool to avoid blocking
        the async event loop.

        Args:
            pdf_path: Path to the PDF file.
            ticker: Ticker symbol for metadata.
            max_pages: Maximum pages to extract.

        Returns:
            List of Document objects, one per page.
        """
        loop = asyncio.get_event_loop()
        try:
            documents = await loop.run_in_executor(
                None,
                self._extract_pdf_sync,
                pdf_path,
                ticker,
                max_pages,
            )
            return documents
        except Exception as e:
            logger.warning(
                "PDFLoader | extraction failed | file=%s | %s", pdf_path.name, e
            )
            return []

    def _extract_pdf_sync(
        self, pdf_path: Path, ticker: str, max_pages: int
    ) -> List[Document]:
        """Synchronous PDF text extraction using pypdf.

        Args:
            pdf_path: Path to the PDF file.
            ticker: Ticker symbol.
            max_pages: Maximum pages to extract.

        Returns:
            List of Document objects.
        """
        try:
            import pypdf  # type: ignore[import]
        except ImportError:
            logger.error(
                "PDFLoader | pypdf not installed. Run: pip install pypdf"
            )
            return []

        documents: List[Document] = []

        try:
            reader = pypdf.PdfReader(str(pdf_path))
        except Exception as e:
            logger.warning("PDFLoader | cannot open PDF | file=%s | %s", pdf_path.name, e)
            return []

        # Extract PDF metadata
        pdf_meta = reader.metadata or {}
        pdf_title = str(pdf_meta.get("/Title", pdf_path.stem))
        pdf_date = self._extract_pdf_date(pdf_meta)

        total_pages = min(len(reader.pages), max_pages)
        logger.debug(
            "PDFLoader | extracting | file=%s | pages=%d", pdf_path.name, total_pages
        )

        for page_num in range(total_pages):
            try:
                page = reader.pages[page_num]
                text = page.extract_text() or ""
            except Exception as e:
                logger.debug(
                    "PDFLoader | page extraction failed | page=%d | %s", page_num + 1, e
                )
                continue

            # Skip pages with too little content
            if len(text.strip()) < MIN_PAGE_CHARS:
                continue

            metadata = DocumentMetadata(
                ticker=ticker.upper(),
                source=self.get_source_name(),
                doc_type=DocType.PDF,
                date=pdf_date,
                url=str(pdf_path.absolute()),
                title=f"{pdf_title} — Page {page_num + 1}",
                page_number=page_num + 1,
                extra={
                    "filename": pdf_path.name,
                    "total_pages": str(len(reader.pages)),
                },
            )

            documents.append(
                Document(content=text.strip(), metadata=metadata)
            )

        logger.debug(
            "PDFLoader | extracted | file=%s | pages_with_content=%d",
            pdf_path.name, len(documents),
        )
        return documents

    @staticmethod
    def _extract_pdf_date(pdf_meta: object) -> str:
        """Extract creation date from PDF metadata.

        Args:
            pdf_meta: pypdf DocumentInformation object.

        Returns:
            ISO date string or today's date.
        """
        from datetime import datetime

        try:
            creation_date = getattr(pdf_meta, "get", lambda k, d: d)("/CreationDate", "")
            if creation_date and len(str(creation_date)) >= 8:
                # PDF dates: "D:20240115120000+05'30'"
                date_str = str(creation_date)
                if date_str.startswith("D:"):
                    date_str = date_str[2:]
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except Exception:
            pass

        return datetime.now().strftime("%Y-%m-%d")
