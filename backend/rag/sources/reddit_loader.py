"""Social sentiment loader for QuantMind RAG pipeline.

Fetches social media posts about a ticker from:
1. StockTwits RSS (free, no auth, finance-focused social network)
2. Reddit via PRAW (optional, requires credentials in .env)

WHY SOCIAL SENTIMENT:
    Social media captures retail investor sentiment — what regular people think.
    This is different from SEC filings (official) and news (journalist opinion).
    StockTwits is like Twitter but only for stocks — very finance-focused.

STOCKTWITS:
    - Free RSS feed, no API key needed
    - URL: https://api.stocktwits.com/api/2/streams/symbol/{ticker}.rss
    - Returns recent posts about the ticker
    - No rate limits for RSS

REDDIT (optional):
    - Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
    - Register at: https://www.reddit.com/prefs/apps (type: script)
    - Searches r/wallstreetbets, r/investing, r/stocks
    - Falls back gracefully if credentials not configured
"""

from __future__ import annotations

import asyncio
from typing import List

import feedparser  # Already installed (used by NewsLoader)

from config.logging_config import get_logger
from config.settings import settings
from rag.sources.base_loader import BaseLoader, Document, DocumentMetadata, DocType

logger = get_logger(__name__)

# StockTwits RSS endpoint (free, no auth)
STOCKTWITS_RSS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.rss"

# Reddit subreddits (optional, requires credentials)
FINANCIAL_SUBREDDITS = ["wallstreetbets", "investing", "stocks"]
MAX_POSTS_PER_SUB = 8
TIME_FILTER = "week"


class RedditLoader(BaseLoader):
    """Loads social sentiment from StockTwits RSS + optional Reddit.

    StockTwits works without any credentials.
    Reddit requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.

    Example:
        >>> loader = RedditLoader()
        >>> docs = await loader.load("AAPL")
        >>> print(docs[0].content[:100])
        'AAPL looking bullish after iPhone 16 launch...'
    """

    source_name = "Social"

    def __init__(self) -> None:
        """Initialize with optional Reddit credentials from settings."""
        self.reddit_client_id = getattr(settings, "reddit_client_id", "")
        self.reddit_client_secret = getattr(settings, "reddit_client_secret", "")
        self.user_agent = "QuantMind/1.0 (sentiment analysis)"

    async def load(self, ticker: str) -> List[Document]:
        """Fetch social posts about a ticker.

        Tries StockTwits RSS first (always works), then Reddit if configured.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            List of Document objects with social posts.
        """
        logger.info("SocialLoader | loading | ticker=%s", ticker)

        docs: List[Document] = []

        # 1. StockTwits RSS (always available, no credentials)
        stocktwits_docs = await self._load_stocktwits(ticker)
        docs.extend(stocktwits_docs)

        # 2. Reddit (optional, only if credentials configured)
        if self.reddit_client_id and self.reddit_client_secret:
            reddit_docs = await self._load_reddit(ticker)
            docs.extend(reddit_docs)
        else:
            logger.debug(
                "SocialLoader | Reddit skipped (no credentials) | "
                "Add REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET to .env to enable"
            )

        logger.info(
            "SocialLoader | complete | ticker=%s | docs=%d "
            "(stocktwits=%d, reddit=%d)",
            ticker, len(docs), len(stocktwits_docs),
            len(docs) - len(stocktwits_docs),
        )
        return docs

    async def _load_stocktwits(self, ticker: str) -> List[Document]:
        """Fetch StockTwits RSS feed for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of Document objects from StockTwits.
        """
        try:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None, self._fetch_stocktwits_sync, ticker
            )
            return docs
        except Exception as e:
            logger.debug("SocialLoader | StockTwits failed | ticker=%s | %s", ticker, e)
            return []

    def _fetch_stocktwits_sync(self, ticker: str) -> List[Document]:
        """Synchronous StockTwits RSS fetch.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of Document objects.
        """
        url = STOCKTWITS_RSS_URL.format(ticker=ticker.upper())
        feed = feedparser.parse(url)

        if not feed.entries:
            logger.debug("SocialLoader | StockTwits | no entries | ticker=%s", ticker)
            return []

        documents = []
        for entry in feed.entries[:15]:  # Max 15 posts
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            content = title
            if summary and summary != title:
                content += f" {summary[:200]}"

            if len(content.strip()) < 10:
                continue

            doc = Document(
                content=content.strip(),
                metadata=DocumentMetadata(
                    ticker=ticker,
                    source="StockTwits",
                    doc_type=DocType.NEWS,
                    url=getattr(entry, "link", ""),
                    title=title[:100],
                ),
            )
            documents.append(doc)

        logger.info(
            "SocialLoader | StockTwits | ticker=%s | posts=%d",
            ticker, len(documents),
        )
        return documents

    async def _load_reddit(self, ticker: str) -> List[Document]:
        """Fetch Reddit posts (requires credentials).

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of Document objects from Reddit.
        """
        try:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None, self._fetch_reddit_sync, ticker
            )
            return docs
        except Exception as e:
            logger.debug("SocialLoader | Reddit failed | ticker=%s | %s", ticker, e)
            return []

    def _fetch_reddit_sync(self, ticker: str) -> List[Document]:
        """Synchronous Reddit fetch via PRAW.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of Document objects from Reddit.
        """
        try:
            import praw  # type: ignore[import]
        except ImportError:
            logger.warning("SocialLoader | praw not installed. Run: pip install praw")
            return []

        reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent=self.user_agent,
        )

        documents: List[Document] = []
        search_queries = [ticker, f"${ticker}"]

        for subreddit_name in FINANCIAL_SUBREDDITS:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for query in search_queries[:1]:  # Just ticker, not $ticker
                    posts = subreddit.search(
                        query=query,
                        time_filter=TIME_FILTER,
                        limit=MAX_POSTS_PER_SUB,
                        sort="relevance",
                    )
                    for post in posts:
                        content = post.title
                        if post.selftext and len(post.selftext) > 20:
                            content += f"\n{post.selftext[:300]}"

                        if len(content.strip()) < 10:
                            continue

                        doc = Document(
                            content=content,
                            metadata=DocumentMetadata(
                                ticker=ticker,
                                source=f"Reddit r/{subreddit_name}",
                                doc_type=DocType.NEWS,
                                url=f"https://reddit.com{post.permalink}",
                                title=post.title[:100],
                            ),
                        )
                        documents.append(doc)
            except Exception as e:
                logger.debug("SocialLoader | r/%s error | %s", subreddit_name, e)
                continue

        # Deduplicate
        seen = set()
        unique = []
        for doc in documents:
            url = doc.metadata.url or doc.content[:50]
            if url not in seen:
                seen.add(url)
                unique.append(doc)

        logger.info(
            "SocialLoader | Reddit | ticker=%s | posts=%d",
            ticker, len(unique),
        )
        return unique
