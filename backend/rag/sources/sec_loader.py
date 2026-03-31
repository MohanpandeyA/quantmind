"""SEC EDGAR document loader for QuantMind RAG pipeline.

WHY THIS LOADER EXISTS:
    SEC filings (10-K annual, 10-Q quarterly, 8-K material events) are the
    most authoritative source of financial information. They contain:
    - Revenue, profit, debt, cash flow (10-K/10-Q)
    - Material events: acquisitions, CEO changes, lawsuits (8-K)
    - Risk factors that explain price movements
    - Management's discussion of business performance

    Unlike news articles, SEC filings are legally required to be accurate.
    They're the gold standard for fundamental analysis.

CRITICAL FIX — CIK LOOKUP:
    Problem: SEC EDGAR uses CIK (Central Index Key) numbers, not ticker symbols.
    Our previous code tried to guess CIK from ticker directly, which failed
    for most companies:
        SECLoader | CIK not found | ticker=NVDA
        SECLoader | CIK not found | ticker=JPM
        SECLoader | CIK not found | ticker=AAPL

    This meant the RAG pipeline had NO SEC filing data for most tickers,
    severely degrading the quality of AI explanations.

    Fix: EDGAR provides a free JSON file mapping ALL tickers to CIKs:
        https://www.sec.gov/files/company_tickers.json

    We download this file ONCE at first use and cache it in memory.
    Subsequent lookups are O(1) dictionary lookups — instant.

    Result: AAPL, MSFT, NVDA, JPM, TSLA, GOOGL all get SEC filings now.

Completely free — no API key required. Rate limit: 10 req/sec.
"""

from __future__ import annotations

import asyncio
import re
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
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
EDGAR_RATE_LIMIT_DELAY = 0.11  # 10 req/sec max → 0.11s between requests

# EDGAR company tickers JSON — maps ALL tickers to CIKs (free, no key)
# Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# User-Agent required by SEC EDGAR (they block requests without it)
EDGAR_HEADERS = {
    "User-Agent": "QuantMind Research quantmind@research.com",
    "Accept-Encoding": "gzip, deflate",
}

# Filing types to fetch
SUPPORTED_FILING_TYPES = {
    "10-K": DocType.SEC_10K,
    "10-Q": DocType.SEC_10Q,
    "8-K": DocType.SEC_8K,
}

# Max characters to extract per filing (avoid huge documents)
MAX_FILING_CHARS = 50_000

# Module-level CIK cache — loaded once, reused for all tickers
# WHY module-level: The JSON file has ~10,000 companies. Loading it once
# at first use and caching in memory means all subsequent lookups are O(1).
# Memory: ~10,000 entries × ~50 bytes = ~500KB. Negligible.
_ticker_to_cik: Dict[str, str] = {}
_cik_cache_loaded = False


async def _load_ticker_cik_map(session: aiohttp.ClientSession) -> None:
    """Load the EDGAR ticker-to-CIK mapping from SEC's JSON file.

    WHY THIS FUNCTION:
        SEC EDGAR uses CIK numbers internally, not ticker symbols.
        The company_tickers.json file provides the complete mapping
        for all ~10,000 public companies. We download it once and
        cache it in the module-level _ticker_to_cik dict.

    Args:
        session: Active aiohttp session.
    """
    global _ticker_to_cik, _cik_cache_loaded

    if _cik_cache_loaded:
        return

    try:
        logger.info("SECLoader | loading ticker-to-CIK map from EDGAR...")
        async with session.get(
            EDGAR_TICKERS_URL,
            headers=EDGAR_HEADERS,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    "SECLoader | failed to load CIK map | status=%d", resp.status
                )
                return

            data = await resp.json(content_type=None)

            # Build ticker → CIK mapping
            # JSON format: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
            for entry in data.values():
                ticker = str(entry.get("ticker", "")).upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                if ticker and cik:
                    _ticker_to_cik[ticker] = cik

            _cik_cache_loaded = True
            logger.info(
                "SECLoader | CIK map loaded | %d companies indexed",
                len(_ticker_to_cik),
            )

    except Exception as e:
        logger.warning("SECLoader | CIK map load failed: %s", e)


class SECLoader(BaseLoader):
    """Loads SEC EDGAR filings for a given ticker symbol.

    Fetches 10-K (annual), 10-Q (quarterly), and 8-K (material events)
    filings from the SEC EDGAR API. Completely free, no API key needed.

    CRITICAL FIX: Uses EDGAR's company_tickers.json for reliable CIK lookup.
    Previously, CIK lookup failed for most tickers (NVDA, JPM, AAPL, etc.)
    because we tried to guess CIK from ticker directly. Now we use the
    official EDGAR mapping file which covers all ~10,000 public companies.

    Attributes:
        filing_types: List of filing types to fetch (default: 10-K, 10-Q, 8-K).
        max_filings: Maximum number of filings per type (default: 5).
        days_back: Only fetch filings from the last N days (default: 365).

    Example:
        >>> loader = SECLoader(filing_types=["10-K", "10-Q"], max_filings=3)
        >>> docs = await loader.load("AAPL")
        >>> print(docs[0])
        Document(ticker='AAPL', type='10-K', date='2024-11-01', words=8432)
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

        # Validate filing types
        for ft in self.filing_types:
            if ft not in SUPPORTED_FILING_TYPES:
                raise ValueError(
                    f"Unsupported filing type '{ft}'. "
                    f"Supported: {list(SUPPORTED_FILING_TYPES.keys())}"
                )

    def get_source_name(self) -> str:
        """Return source name."""
        return "SEC EDGAR"

    async def load(self, ticker: str, **kwargs: object) -> List[Document]:
        """Fetch SEC filings for a ticker symbol.

        Steps:
            1. Load EDGAR ticker-to-CIK map (once, cached)
            2. Look up CIK for the ticker (O(1) dict lookup)
            3. Fetch filing list from EDGAR submissions API
            4. Filter by filing type and date range
            5. Download and extract text from each filing

        Args:
            ticker: Stock ticker (e.g., 'AAPL', 'MSFT', 'NVDA').
            **kwargs: Optional overrides:
                - max_filings (int): Override instance max_filings.
                - days_back (int): Override instance days_back.

        Returns:
            List of Document objects, one per filing section.
        """
        self.validate_query(ticker)
        ticker_upper = ticker.upper()
        max_filings = int(kwargs.get("max_filings", self.max_filings))
        days_back = int(kwargs.get("days_back", self.days_back))
        cutoff_date = datetime.now() - timedelta(days=days_back)

        logger.info(
            "SECLoader | loading | ticker=%s | types=%s | max=%d | days_back=%d",
            ticker_upper, self.filing_types, max_filings, days_back,
        )

        async with aiohttp.ClientSession(headers=EDGAR_HEADERS) as session:
            # Step 1: Load CIK map (cached after first call)
            await _load_ticker_cik_map(session)

            # Step 2: Look up CIK — O(1) dict lookup
            cik = _ticker_to_cik.get(ticker_upper)

            if not cik:
                logger.warning(
                    "SECLoader | ticker not in EDGAR map | ticker=%s | "
                    "total_companies_indexed=%d",
                    ticker_upper, len(_ticker_to_cik),
                )
                return []

            logger.debug("SECLoader | CIK found | ticker=%s | cik=%s", ticker_upper, cik)

            # Step 3: Fetch filing metadata
            filings_meta = await self._get_filings_metadata(session, cik, cutoff_date)
            if not filings_meta:
                logger.info("SECLoader | no filings found | ticker=%s", ticker_upper)
                return []

            # Step 4: Download filing texts (with rate limiting)
            documents: List[Document] = []
            count_per_type: Dict[str, int] = {}

            for filing in filings_meta:
                filing_type = filing["form"]
                if filing_type not in self.filing_types:
                    continue

                count = count_per_type.get(filing_type, 0)
                if count >= max_filings:
                    continue

                # Rate limit: 0.11s between requests (EDGAR allows 10 req/sec)
                await asyncio.sleep(EDGAR_RATE_LIMIT_DELAY)

                doc = await self._download_filing(session, cik, filing, ticker_upper)
                if doc and not doc.is_empty():
                    documents.append(doc)
                    count_per_type[filing_type] = count + 1

        self._log_load_result(ticker_upper, documents)
        return documents

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
            List of filing metadata dicts.
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
                    return None

                raw_text = await resp.text(encoding="utf-8", errors="replace")

        except aiohttp.ClientError as e:
            logger.debug("SECLoader | filing download error | url=%s | %s", url, e)
            return None

        # Extract clean text from HTML/XML
        clean_text = self._extract_text(raw_text)

        if len(clean_text.strip()) < 100:
            return None

        # Truncate to max chars
        if len(clean_text) > MAX_FILING_CHARS:
            clean_text = clean_text[:MAX_FILING_CHARS] + "\n[TRUNCATED]"

        doc_type = SUPPORTED_FILING_TYPES.get(filing_type, DocType.UNKNOWN)

        metadata = DocumentMetadata(
            ticker=ticker,
            source=self.get_source_name(),
            doc_type=doc_type,
            date=filing_date,
            url=url,
            title=f"{ticker} {filing_type} ({filing_date})",
            filing_period=filing_date[:7],
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

        Removes HTML/XML tags, XBRL markup, and table artifacts.

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
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&#160;", " ").replace("&quot;", '"')

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove lines that are just numbers or dashes (table artifacts)
        lines = text.split("\n")
        clean_lines = [
            line for line in lines
            if len(line.strip()) > 10
            and not re.match(r"^[\d\s\-\.\,\$\%]+$", line.strip())
        ]

        return "\n".join(clean_lines).strip()
