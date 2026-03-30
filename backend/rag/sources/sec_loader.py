"""SEC EDGAR document loader for QuantMind RAG pipeline.

Fetches SEC filings (10-K, 10-Q, 8-K) from the EDGAR API.
Completely free — no API key required. Rate limit: 10 req/sec.

EDGAR API endpoints used:
    - Company search: https://efts.sec.gov/LATEST/search-index?q={ticker}&dateRange=custom
    - Submissions:    https://data.sec.gov/submissions/CIK{cik}.json
    - Filing index:   https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
    - Full text:      https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}

Optimization principles:
    - Incremental loading: only fetch filings newer than last ingested date
    - Rate limiting: 0.1s delay between requests (EDGAR allows 10 req/sec)
    - Async HTTP: aiohttp for non-blocking requests
    - Text extraction: strip HTML/XML tags, keep only meaningful text
    - Deduplication: doc_id hash prevents re-processing same filing
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp

from config.logging_config import get_logger
from rag.sources.base_loader import (
    BaseLoader,
    Document,
    DocumentMetadata,
    DocType,
    LoaderError,
    RateLimitError,
)

logger = get_logger(__name__)

# EDGAR API constants
EDGAR_BASE_URL = "https://data.sec.gov"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
EDGAR_RATE_LIMIT_DELAY = 0.11  # 10 req/sec max → 0.1s between requests

# User-Agent required by SEC EDGAR (they block requests without it)
EDGAR_HEADERS = {
    "User-Agent": "QuantMind Research quantmind@research.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Filing types to fetch
SUPPORTED_FILING_TYPES = {
    "10-K": DocType.SEC_10K,
    "10-Q": DocType.SEC_10Q,
    "8-K": DocType.SEC_8K,
}

# Max characters to extract per filing (avoid huge documents)
MAX_FILING_CHARS = 50_000


class SECLoader(BaseLoader):
    """Loads SEC EDGAR filings for a given ticker symbol.

    Fetches 10-K (annual), 10-Q (quarterly), and 8-K (material events)
    filings from the SEC EDGAR API. Completely free, no API key needed.

    The SEC requires a User-Agent header identifying your application.
    We use "QuantMind Research quantmind@research.com" — this is fine
    for research/personal use.

    Attributes:
        filing_types: List of filing types to fetch (default: 10-K, 10-Q, 8-K).
        max_filings: Maximum number of filings per type (default: 5).
        days_back: Only fetch filings from the last N days (default: 365).

    Example:
        >>> loader = SECLoader(filing_types=["10-K", "10-Q"], max_filings=3)
        >>> docs = await loader.load("AAPL")
        >>> print(docs[0])
        Document(ticker='AAPL', type='10-K', date='2024-11-01', words=8432, id='a3f2b1c4')
    """

    def __init__(
        self,
        filing_types: Optional[List[str]] = None,
        max_filings: int = 5,
        days_back: int = 365,
    ) -> None:
        """Initialize SECLoader.

        Args:
            filing_types: Filing types to fetch. Default: ['10-K', '10-Q', '8-K'].
            max_filings: Max filings per type per ticker. Default: 5.
            days_back: Only fetch filings from last N days. Default: 365.
        """
        self.filing_types: List[str] = filing_types or list(SUPPORTED_FILING_TYPES.keys())
        self.max_filings: int = max_filings
        self.days_back: int = days_back
        self._cik_cache: Dict[str, str] = {}  # ticker → CIK cache

        # Validate filing types
        for ft in self.filing_types:
            if ft not in SUPPORTED_FILING_TYPES:
                raise ValueError(
                    f"Unsupported filing type '{ft}'. "
                    f"Supported: {list(SUPPORTED_FILING_TYPES.keys())}"
                )

    def get_source_name(self) -> str:
        """Return source name.

        Returns:
            'SEC EDGAR'
        """
        return "SEC EDGAR"

    async def load(self, ticker: str, **kwargs: object) -> List[Document]:
        """Fetch SEC filings for a ticker symbol.

        Steps:
            1. Look up CIK (Central Index Key) for the ticker
            2. Fetch filing list from EDGAR submissions API
            3. Filter by filing type and date range
            4. Download and extract text from each filing
            5. Return as Document objects

        Args:
            ticker: Stock ticker (e.g., 'AAPL', 'MSFT').
            **kwargs: Optional overrides:
                - max_filings (int): Override instance max_filings.
                - days_back (int): Override instance days_back.

        Returns:
            List of Document objects, one per filing section.

        Raises:
            LoaderError: If CIK lookup fails or EDGAR is unreachable.
        """
        self.validate_query(ticker)
        max_filings = int(kwargs.get("max_filings", self.max_filings))
        days_back = int(kwargs.get("days_back", self.days_back))
        cutoff_date = datetime.now() - timedelta(days=days_back)

        logger.info(
            "SECLoader | loading | ticker=%s | types=%s | max=%d | days_back=%d",
            ticker, self.filing_types, max_filings, days_back,
        )

        async with aiohttp.ClientSession(headers=EDGAR_HEADERS) as session:
            # Step 1: Get CIK for ticker
            cik = await self._get_cik(session, ticker)
            if not cik:
                logger.warning("SECLoader | CIK not found | ticker=%s", ticker)
                return []

            # Step 2: Fetch filing metadata
            filings_meta = await self._get_filings_metadata(session, cik, cutoff_date)
            if not filings_meta:
                logger.info("SECLoader | no filings found | ticker=%s", ticker)
                return []

            # Step 3: Download filing texts (with rate limiting)
            documents: List[Document] = []
            count_per_type: Dict[str, int] = {}

            for filing in filings_meta:
                filing_type = filing["form"]
                if filing_type not in self.filing_types:
                    continue

                count = count_per_type.get(filing_type, 0)
                if count >= max_filings:
                    continue

                # Rate limit: 0.11s between requests
                await asyncio.sleep(EDGAR_RATE_LIMIT_DELAY)

                doc = await self._download_filing(session, cik, filing, ticker)
                if doc and not doc.is_empty():
                    documents.append(doc)
                    count_per_type[filing_type] = count + 1

        self._log_load_result(ticker, documents)
        return documents

    async def _get_cik(self, session: aiohttp.ClientSession, ticker: str) -> Optional[str]:
        """Look up the CIK (Central Index Key) for a ticker symbol.

        CIK is EDGAR's internal company identifier. Required for all
        subsequent API calls.

        Args:
            session: Active aiohttp session.
            ticker: Stock ticker symbol.

        Returns:
            CIK string (zero-padded to 10 digits), or None if not found.
        """
        # Check cache first
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        url = f"{EDGAR_BASE_URL}/submissions/CIK{ticker.upper()}.json"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    cik = str(data.get("cik", "")).zfill(10)
                    self._cik_cache[ticker] = cik
                    logger.debug("SECLoader | CIK found | ticker=%s | cik=%s", ticker, cik)
                    return cik
                elif resp.status == 404:
                    # Try company search as fallback
                    return await self._search_cik(session, ticker)
                else:
                    logger.warning(
                        "SECLoader | CIK lookup failed | ticker=%s | status=%d",
                        ticker, resp.status,
                    )
                    return None
        except aiohttp.ClientError as e:
            logger.error("SECLoader | CIK request error | ticker=%s | %s", ticker, e)
            return None

    async def _search_cik(
        self, session: aiohttp.ClientSession, ticker: str
    ) -> Optional[str]:
        """Search for CIK using EDGAR company search API.

        Fallback when direct CIK lookup fails (e.g., for tickers that
        don't match the EDGAR CIK format directly).

        Args:
            session: Active aiohttp session.
            ticker: Stock ticker symbol.

        Returns:
            CIK string or None.
        """
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {"q": f'"{ticker}"', "dateRange": "custom", "forms": "10-K"}

        try:
            await asyncio.sleep(EDGAR_RATE_LIMIT_DELAY)
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    hits = data.get("hits", {}).get("hits", [])
                    if hits:
                        entity_id = hits[0].get("_source", {}).get("entity_id", "")
                        if entity_id:
                            cik = str(entity_id).zfill(10)
                            self._cik_cache[ticker] = cik
                            return cik
        except aiohttp.ClientError as e:
            logger.debug("SECLoader | search CIK error | %s", e)

        return None

    async def _get_filings_metadata(
        self,
        session: aiohttp.ClientSession,
        cik: str,
        cutoff_date: datetime,
    ) -> List[Dict[str, str]]:
        """Fetch filing metadata list from EDGAR submissions API.

        Args:
            session: Active aiohttp session.
            cik: Company CIK (zero-padded 10 digits).
            cutoff_date: Only return filings after this date.

        Returns:
            List of filing metadata dicts with keys:
            form, filingDate, accessionNumber, primaryDocument.
        """
        url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"

        try:
            await asyncio.sleep(EDGAR_RATE_LIMIT_DELAY)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json(content_type=None)
                recent = data.get("filings", {}).get("recent", {})

                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                accessions = recent.get("accessionNumber", [])
                primary_docs = recent.get("primaryDocument", [])

                filings = []
                for form, date_str, accession, primary_doc in zip(
                    forms, dates, accessions, primary_docs
                ):
                    try:
                        filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        continue

                    if filing_date < cutoff_date:
                        continue

                    if form in self.filing_types:
                        filings.append({
                            "form": form,
                            "filingDate": date_str,
                            "accessionNumber": accession,
                            "primaryDocument": primary_doc,
                        })

                logger.debug(
                    "SECLoader | filings found | cik=%s | count=%d", cik, len(filings)
                )
                return filings

        except aiohttp.ClientError as e:
            logger.error("SECLoader | filings metadata error | cik=%s | %s", cik, e)
            return []

    async def _download_filing(
        self,
        session: aiohttp.ClientSession,
        cik: str,
        filing: Dict[str, str],
        ticker: str,
    ) -> Optional[Document]:
        """Download and extract text from a single SEC filing.

        Args:
            session: Active aiohttp session.
            cik: Company CIK.
            filing: Filing metadata dict.
            ticker: Ticker symbol for metadata.

        Returns:
            Document object with extracted text, or None on failure.
        """
        accession = filing["accessionNumber"].replace("-", "")
        primary_doc = filing["primaryDocument"]
        filing_type = filing["form"]
        filing_date = filing["filingDate"]

        # Construct filing URL
        url = f"{EDGAR_ARCHIVES_URL}/{int(cik)}/{accession}/{primary_doc}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.debug(
                        "SECLoader | filing download failed | url=%s | status=%d",
                        url, resp.status,
                    )
                    return None

                raw_text = await resp.text(encoding="utf-8", errors="replace")

        except aiohttp.ClientError as e:
            logger.debug("SECLoader | filing download error | url=%s | %s", url, e)
            return None

        # Extract clean text from HTML/XML
        clean_text = self._extract_text(raw_text)

        if len(clean_text.strip()) < 100:
            logger.debug("SECLoader | filing too short after extraction | url=%s", url)
            return None

        # Truncate to max chars to avoid huge documents
        if len(clean_text) > MAX_FILING_CHARS:
            clean_text = clean_text[:MAX_FILING_CHARS] + "\n[TRUNCATED]"

        doc_type = SUPPORTED_FILING_TYPES.get(filing_type, DocType.UNKNOWN)

        metadata = DocumentMetadata(
            ticker=ticker.upper(),
            source=self.get_source_name(),
            doc_type=doc_type,
            date=filing_date,
            url=url,
            title=f"{ticker.upper()} {filing_type} ({filing_date})",
            filing_period=filing_date[:7],  # "2024-01"
            extra={
                "accession_number": filing["accessionNumber"],
                "cik": cik,
            },
        )

        logger.debug(
            "SECLoader | filing downloaded | ticker=%s | type=%s | date=%s | chars=%d",
            ticker, filing_type, filing_date, len(clean_text),
        )

        return Document(content=clean_text, metadata=metadata)

    @staticmethod
    def _extract_text(raw: str) -> str:
        """Extract clean text from HTML/XML SEC filing content.

        Removes:
        - HTML/XML tags
        - XBRL markup
        - Excessive whitespace
        - Page headers/footers
        - Table formatting artifacts

        Args:
            raw: Raw HTML/XML text from EDGAR.

        Returns:
            Clean plain text suitable for embedding.
        """
        # Remove XBRL/XML declarations
        text = re.sub(r"<\?xml[^>]*\?>", "", raw)
        text = re.sub(r"<!DOCTYPE[^>]*>", "", text)

        # Remove script and style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode common HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&nbsp;", " ")
        text = text.replace("&#160;", " ")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove lines that are just numbers or dashes (table artifacts)
        lines = text.split("\n")
        clean_lines = [
            line for line in lines
            if len(line.strip()) > 10  # Skip very short lines
            and not re.match(r"^[\d\s\-\.\,\$\%]+$", line.strip())  # Skip pure number lines
        ]

        return "\n".join(clean_lines).strip()
