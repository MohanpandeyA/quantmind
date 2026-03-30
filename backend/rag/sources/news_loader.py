"""Financial news loader for QuantMind RAG pipeline.

Fetches financial news from two free sources:
    1. NewsAPI (https://newsapi.org) — 100 req/day free tier
    2. RSS feeds — Reuters, Yahoo Finance, Seeking Alpha (unlimited, no key)

Optimization principles:
    - Async concurrent fetching: all RSS feeds fetched simultaneously
    - Deduplication: URL-based dedup prevents same article twice
    - Rate limiting: respects NewsAPI 100 req/day limit
    - Fallback: if NewsAPI fails, RSS feeds still work
    - Content filtering: removes boilerplate, keeps financial content
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote

import aiohttp
import feedparser  # type: ignore[import]

from config.logging_config import get_logger
from config.settings import settings
from rag.sources.base_loader import (
    BaseLoader,
    Document,
    DocumentMetadata,
    DocType,
    LoaderError,
    RateLimitError,
)

logger = get_logger(__name__)

# NewsAPI endpoint
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Free RSS feeds — no API key needed, unlimited requests
RSS_FEEDS: Dict[str, str] = {
    "Yahoo Finance": "https://finance.yahoo.com/rss/topstories",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",
    "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
}

# Max article content length
MAX_ARTICLE_CHARS = 5_000


class NewsLoader(BaseLoader):
    """Loads financial news articles for a given ticker symbol.

    Combines NewsAPI (structured, ticker-specific) with RSS feeds
    (unstructured, general financial news). Both are free.

    Attributes:
        use_newsapi: Whether to use NewsAPI (requires key). Default True.
        use_rss: Whether to use RSS feeds. Default True.
        max_articles: Maximum articles to return. Default 20.
        days_back: Only fetch articles from last N days. Default 30.

    Example:
        >>> loader = NewsLoader(max_articles=10, days_back=7)
        >>> docs = await loader.load("AAPL")
        >>> print(docs[0].metadata.title)
        "Apple Reports Record Q3 Revenue..."
    """

    def __init__(
        self,
        use_newsapi: bool = True,
        use_rss: bool = True,
        max_articles: int = 20,
        days_back: int = 30,
    ) -> None:
        """Initialize NewsLoader.

        Args:
            use_newsapi: Use NewsAPI if key is configured. Default True.
            use_rss: Use RSS feeds (always free). Default True.
            max_articles: Max articles to return total. Default 20.
            days_back: Only fetch articles from last N days. Default 30.
        """
        self.use_newsapi = use_newsapi and bool(settings.news_api_key)
        self.use_rss = use_rss
        self.max_articles = max_articles
        self.days_back = days_back
        self._seen_urls: set[str] = set()  # URL-based deduplication

    def get_source_name(self) -> str:
        """Return source name.

        Returns:
            'NewsAPI+RSS'
        """
        return "NewsAPI+RSS"

    async def load(self, ticker: str, **kwargs: object) -> List[Document]:
        """Fetch news articles for a ticker symbol.

        Fetches from NewsAPI and RSS feeds concurrently using asyncio.gather().
        Deduplicates by URL and returns up to max_articles documents.

        Args:
            ticker: Stock ticker (e.g., 'AAPL', 'MSFT').
            **kwargs: Optional overrides:
                - max_articles (int): Override instance max_articles.
                - days_back (int): Override instance days_back.

        Returns:
            List of Document objects, one per article.
        """
        self.validate_query(ticker)
        max_articles = int(kwargs.get("max_articles", self.max_articles))
        days_back = int(kwargs.get("days_back", self.days_back))
        self._seen_urls.clear()

        logger.info(
            "NewsLoader | loading | ticker=%s | newsapi=%s | rss=%s | max=%d",
            ticker, self.use_newsapi, self.use_rss, max_articles,
        )

        tasks = []
        async with aiohttp.ClientSession() as session:
            if self.use_newsapi:
                tasks.append(self._fetch_newsapi(session, ticker, days_back))
            if self.use_rss:
                tasks.append(self._fetch_rss_feeds(ticker))

            if not tasks:
                logger.warning("NewsLoader | no sources enabled | ticker=%s", ticker)
                return []

            # Fetch all sources concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: List[Document] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("NewsLoader | source failed | %s", result)
                continue
            if isinstance(result, list):
                documents.extend(result)

        # Deduplicate and limit
        seen_ids: set[str] = set()
        unique_docs: List[Document] = []
        for doc in documents:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                unique_docs.append(doc)

        # Sort by date (newest first) and limit
        unique_docs.sort(key=lambda d: d.metadata.date, reverse=True)
        final_docs = unique_docs[:max_articles]

        self._log_load_result(ticker, final_docs)
        return final_docs

    async def _fetch_newsapi(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        days_back: int,
    ) -> List[Document]:
        """Fetch articles from NewsAPI.

        Args:
            session: Active aiohttp session.
            ticker: Ticker symbol to search for.
            days_back: Number of days back to search.

        Returns:
            List of Document objects from NewsAPI.
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": ticker,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": settings.news_api_key,
            "pageSize": min(self.max_articles, 100),
        }

        try:
            async with session.get(
                NEWSAPI_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 429:
                    raise RateLimitError(
                        self.get_source_name(), ticker,
                        "NewsAPI rate limit exceeded (100 req/day on free tier)"
                    )
                if resp.status == 401:
                    logger.warning("NewsLoader | NewsAPI key invalid | ticker=%s", ticker)
                    return []
                if resp.status != 200:
                    logger.warning(
                        "NewsLoader | NewsAPI error | status=%d | ticker=%s",
                        resp.status, ticker,
                    )
                    return []

                data = await resp.json()

        except aiohttp.ClientError as e:
            logger.warning("NewsLoader | NewsAPI request failed | %s", e)
            return []

        articles = data.get("articles", [])
        documents: List[Document] = []

        for article in articles:
            doc = self._article_to_document(article, ticker, "NewsAPI")
            if doc and not doc.is_empty():
                documents.append(doc)

        logger.debug(
            "NewsLoader | NewsAPI | ticker=%s | articles=%d", ticker, len(documents)
        )
        return documents

    async def _fetch_rss_feeds(self, ticker: str) -> List[Document]:
        """Fetch articles from all RSS feeds concurrently.

        RSS feeds are general financial news — not ticker-specific.
        We filter articles that mention the ticker in title or description.

        Args:
            ticker: Ticker to filter for in article content.

        Returns:
            List of Document objects from RSS feeds.
        """
        # Fetch all RSS feeds concurrently
        tasks = [
            self._fetch_single_rss(name, url, ticker)
            for name, url in RSS_FEEDS.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: List[Document] = []
        for result in results:
            if isinstance(result, list):
                documents.extend(result)

        logger.debug(
            "NewsLoader | RSS | ticker=%s | articles=%d", ticker, len(documents)
        )
        return documents

    async def _fetch_single_rss(
        self, feed_name: str, feed_url: str, ticker: str
    ) -> List[Document]:
        """Fetch and parse a single RSS feed.

        Args:
            feed_name: Human-readable feed name.
            feed_url: RSS feed URL.
            ticker: Ticker to filter for.

        Returns:
            List of Document objects from this feed.
        """
        try:
            # feedparser is synchronous — run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
        except Exception as e:
            logger.debug("NewsLoader | RSS parse error | feed=%s | %s", feed_name, e)
            return []

        documents: List[Document] = []
        ticker_upper = ticker.upper()
        company_name = self._ticker_to_company(ticker)

        for entry in feed.entries[:50]:  # Limit per feed
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link = getattr(entry, "link", "")
            published = getattr(entry, "published", "")

            # Filter: only include articles mentioning the ticker or company
            combined = f"{title} {summary}".upper()
            if ticker_upper not in combined and company_name.upper() not in combined:
                continue

            # Skip if already seen this URL
            if link in self._seen_urls:
                continue
            self._seen_urls.add(link)

            content = self._clean_html(f"{title}\n\n{summary}")
            if len(content.strip()) < 50:
                continue

            # Parse date
            date_str = self._parse_rss_date(published)

            metadata = DocumentMetadata(
                ticker=ticker_upper,
                source=feed_name,
                doc_type=DocType.RSS,
                date=date_str,
                url=link,
                title=title[:200],
            )

            documents.append(Document(content=content[:MAX_ARTICLE_CHARS], metadata=metadata))

        return documents

    def _article_to_document(
        self,
        article: Dict[str, object],
        ticker: str,
        source: str,
    ) -> Optional[Document]:
        """Convert a NewsAPI article dict to a Document.

        Args:
            article: NewsAPI article dictionary.
            ticker: Ticker symbol.
            source: Source name.

        Returns:
            Document object or None if article is invalid.
        """
        title = str(article.get("title", "") or "")
        description = str(article.get("description", "") or "")
        content = str(article.get("content", "") or "")
        url = str(article.get("url", "") or "")
        published_at = str(article.get("publishedAt", "") or "")
        source_name = str((article.get("source") or {}).get("name", source))

        # Skip removed articles
        if title == "[Removed]" or not title:
            return None

        # Skip if already seen
        if url in self._seen_urls:
            return None
        self._seen_urls.add(url)

        # Combine all text fields
        full_content = f"{title}\n\n{description}\n\n{content}"
        full_content = self._clean_html(full_content)

        if len(full_content.strip()) < 50:
            return None

        # Parse date
        date_str = published_at[:10] if published_at else datetime.now().strftime("%Y-%m-%d")

        metadata = DocumentMetadata(
            ticker=ticker.upper(),
            source=source_name,
            doc_type=DocType.NEWS,
            date=date_str,
            url=url,
            title=title[:200],
        )

        return Document(
            content=full_content[:MAX_ARTICLE_CHARS],
            metadata=metadata,
        )

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags and clean whitespace from text.

        Args:
            text: Raw text possibly containing HTML.

        Returns:
            Clean plain text.
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&#160;", " ")
        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _parse_rss_date(date_str: str) -> str:
        """Parse RSS date string to ISO format YYYY-MM-DD.

        Args:
            date_str: RSS date string (various formats).

        Returns:
            ISO date string or today's date if parsing fails.
        """
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")

        # Try common RSS date formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str[:30], fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _ticker_to_company(ticker: str) -> str:
        """Map common tickers to company names for RSS filtering.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Company name string.
        """
        mapping = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "GOOG": "Google",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta",
            "NVDA": "Nvidia",
            "NFLX": "Netflix",
            "JPM": "JPMorgan",
            "BAC": "Bank of America",
            "GS": "Goldman Sachs",
            "RELIANCE": "Reliance",
            "TCS": "Tata Consultancy",
            "INFY": "Infosys",
        }
        return mapping.get(ticker.upper(), ticker)
