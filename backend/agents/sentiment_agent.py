"""SentimentAgent — FinBERT-powered sentiment analysis on news + Reddit.

This is the 7th agent in the LangGraph workflow. It runs AFTER RAGAgent
and BEFORE StrategyAgent, adding sentiment context to the analysis.

WHY SENTIMENT ANALYSIS:
    Fundamental analysis (SEC filings) tells you what happened.
    Technical analysis (price patterns) tells you what the chart shows.
    Sentiment analysis tells you what people FEEL about the stock.

    These three together give a much more complete picture:
    - Strong fundamentals + bullish sentiment → high conviction BUY
    - Strong fundamentals + bearish sentiment → wait for sentiment to improve
    - Weak fundamentals + bullish sentiment → potential short-term pump

WHY FINBERT:
    Generic sentiment models (VADER, TextBlob) are trained on movie reviews
    and social media. They don't understand financial language:
    - "Apple beat estimates" → generic model: neutral (no emotion words)
    - "Apple beat estimates" → FinBERT: positive (0.97)

    FinBERT (ProsusAI/finbert) is fine-tuned on:
    - Financial news articles
    - Earnings call transcripts
    - SEC filings
    → Understands financial jargon and context

SOURCES ANALYZED:
    1. RAG context (SEC filings + news already retrieved by RAGAgent)
    2. Reddit posts (r/wallstreetbets, r/investing, r/stocks) via PRAW

OUTPUT:
    sentiment_score: float from -1.0 (very bearish) to +1.0 (very bullish)
    sentiment_label: "BULLISH" / "BEARISH" / "NEUTRAL"
    sentiment_confidence: 0.0 to 1.0
    sentiment_details: list of top scored sentences with scores

LangGraph node contract:
    Input:  TradingState with rag_context, ticker
    Output: TradingState with sentiment_score, sentiment_label,
            sentiment_confidence, sentiment_details
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from config.logging_config import get_logger
from config.settings import settings
from graph.state import TradingState
from rag.sources.reddit_loader import RedditLoader

logger = get_logger(__name__)

# FinBERT model name (free, runs locally)
FINBERT_MODEL = "ProsusAI/finbert"

# Sentiment thresholds
BULLISH_THRESHOLD = 0.15   # score > 0.15 → BULLISH
BEARISH_THRESHOLD = -0.15  # score < -0.15 → BEARISH

# Max sentences to analyze (performance limit)
MAX_SENTENCES = 30

# Module-level FinBERT singleton (loaded once, reused)
_finbert_pipeline = None
_finbert_lock = None


def _get_finbert_lock():
    """Get or create the threading lock for FinBERT loading."""
    global _finbert_lock
    if _finbert_lock is None:
        import threading
        _finbert_lock = threading.Lock()
    return _finbert_lock


def _load_finbert():
    """Load FinBERT pipeline with double-checked locking.

    Returns:
        transformers pipeline or None if loading fails.
    """
    global _finbert_pipeline

    if _finbert_pipeline is not None:
        return _finbert_pipeline

    lock = _get_finbert_lock()
    with lock:
        if _finbert_pipeline is not None:
            return _finbert_pipeline

        try:
            from transformers import pipeline  # type: ignore[import]
            logger.info("SentimentAgent | loading FinBERT | model=%s", FINBERT_MODEL)
            _finbert_pipeline = pipeline(
                "text-classification",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                device=-1,  # CPU (MPS causes issues with FinBERT on Python 3.9)
                top_k=None,  # Return all labels with scores
                truncation=True,
                max_length=512,
            )
            logger.info("SentimentAgent | FinBERT loaded | model=%s", FINBERT_MODEL)
        except Exception as e:
            logger.warning(
                "SentimentAgent | FinBERT load failed | %s | "
                "Using fallback keyword scoring", e,
            )
            _finbert_pipeline = None

    return _finbert_pipeline


async def sentiment_agent(state: TradingState) -> TradingState:
    """Score sentiment from RAG context + Reddit. LangGraph node function.

    Analyzes text from SEC filings, news, and Reddit posts using FinBERT
    to produce a sentiment score that StrategyAgent uses for decisions.

    Args:
        state: TradingState with rag_context and ticker populated.

    Returns:
        Updated TradingState with sentiment_score, sentiment_label,
        sentiment_confidence, sentiment_details.

    Example:
        >>> state = await sentiment_agent(state)
        >>> state["sentiment_score"]
        0.42
        >>> state["sentiment_label"]
        'BULLISH'
    """
    ticker = state.get("ticker", "")
    rag_context = state.get("rag_context", "")

    logger.info("SentimentAgent | analyzing | ticker=%s", ticker)

    try:
        # 1. Collect text from RAG context
        rag_sentences = _extract_sentences(rag_context)

        # 2. Fetch Reddit posts (if credentials configured)
        reddit_sentences = await _fetch_reddit_sentences(ticker)

        # 3. Combine all sentences (limit total)
        all_sentences = (rag_sentences + reddit_sentences)[:MAX_SENTENCES]

        if not all_sentences:
            logger.warning(
                "SentimentAgent | no text to analyze | ticker=%s", ticker
            )
            return _neutral_state(state)

        # 4. Score with FinBERT (or fallback)
        scored = await _score_sentences(all_sentences)

        if not scored:
            return _neutral_state(state)

        # 5. Aggregate scores
        sentiment_score, confidence = _aggregate_scores(scored)
        sentiment_label = _score_to_label(sentiment_score)

        # 6. Get top sentences for display
        top_positive = sorted(
            [(s, sc) for s, sc, _ in scored if sc > 0],
            key=lambda x: x[1], reverse=True
        )[:3]
        top_negative = sorted(
            [(s, sc) for s, sc, _ in scored if sc < 0],
            key=lambda x: x[1]
        )[:3]

        sentiment_details = {
            "top_positive": [{"text": s[:120], "score": round(sc, 3)} for s, sc in top_positive],
            "top_negative": [{"text": s[:120], "score": round(sc, 3)} for s, sc in top_negative],
            "total_sentences": len(scored),
            "rag_sentences": len(rag_sentences),
            "reddit_sentences": len(reddit_sentences),
        }

        logger.info(
            "SentimentAgent | complete | ticker=%s | score=%.3f | label=%s | "
            "confidence=%.3f | sentences=%d",
            ticker, sentiment_score, sentiment_label, confidence, len(scored),
        )

        return {
            **state,
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "sentiment_confidence": round(confidence, 3),
            "sentiment_details": sentiment_details,
        }

    except Exception as e:
        logger.error(
            "SentimentAgent | failed | ticker=%s | %s", ticker, e, exc_info=True
        )
        return _neutral_state(state)


def _extract_sentences(text: str) -> List[str]:
    """Split RAG context into individual sentences.

    Args:
        text: RAG context string (may contain [Source N] headers).

    Returns:
        List of clean sentences.
    """
    if not text:
        return []

    # Remove [Source N] headers and section markers
    text = re.sub(r'\[Source \d+\][^\n]*\n?', '', text)
    text = re.sub(r'---+', '', text)

    # Split on sentence boundaries AND newlines
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)

    # Filter: keep meaningful text (8+ chars, not just headers)
    clean = []
    for s in parts:
        s = s.strip()
        # Skip very short, pure numbers, or header-like lines
        if len(s) >= 8 and not s.startswith('[') and not re.match(r'^[\d\s\$\%\.\,]+$', s):
            # Truncate very long sentences
            clean.append(s[:300])

    return clean[:25]  # Max 25 from RAG


async def _fetch_reddit_sentences(ticker: str) -> List[str]:
    """Fetch Reddit posts and extract sentences.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of sentences from Reddit posts.
    """
    try:
        loader = RedditLoader()
        docs = await loader.load(ticker)

        sentences = []
        for doc in docs[:10]:  # Max 10 Reddit posts
            post_sentences = _extract_sentences(doc.content)
            sentences.extend(post_sentences[:2])  # Max 2 sentences per post

        logger.info(
            "SentimentAgent | reddit | ticker=%s | posts=%d | sentences=%d",
            ticker, len(docs), len(sentences),
        )
        return sentences[:10]  # Max 10 from Reddit

    except Exception as e:
        logger.debug("SentimentAgent | reddit fetch failed | %s", e)
        return []


async def _score_sentences(
    sentences: List[str],
) -> List[Tuple[str, float, float]]:
    """Score sentences with FinBERT (or fallback keyword scoring).

    Args:
        sentences: List of text sentences.

    Returns:
        List of (sentence, score, confidence) tuples.
        score: -1.0 (very negative) to +1.0 (very positive)
        confidence: 0.0 to 1.0
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _score_sync, sentences)


def _score_sync(sentences: List[str]) -> List[Tuple[str, float, float]]:
    """Synchronous FinBERT scoring (runs in thread pool).

    Args:
        sentences: List of text sentences.

    Returns:
        List of (sentence, score, confidence) tuples.
    """
    pipe = _load_finbert()

    if pipe is not None:
        return _score_with_finbert(pipe, sentences)
    else:
        return _score_with_keywords(sentences)


def _score_with_finbert(
    pipe: Any,
    sentences: List[str],
) -> List[Tuple[str, float, float]]:
    """Score sentences using FinBERT pipeline.

    FinBERT returns 3 labels: positive, negative, neutral
    We convert to a single score: positive=+1, negative=-1, neutral=0

    Args:
        pipe: transformers pipeline.
        sentences: List of sentences.

    Returns:
        List of (sentence, score, confidence) tuples.
    """
    results = []
    for sentence in sentences:
        try:
            outputs = pipe(sentence)
            # outputs is a list of dicts: [{"label": "positive", "score": 0.97}, ...]
            if isinstance(outputs, list) and outputs:
                # Handle both list-of-lists and list-of-dicts
                label_scores = outputs[0] if isinstance(outputs[0], list) else outputs

                label_map = {item["label"]: item["score"] for item in label_scores}
                pos = label_map.get("positive", 0.0)
                neg = label_map.get("negative", 0.0)
                neu = label_map.get("neutral", 0.0)

                # Weighted score: positive=+1, negative=-1, neutral=0
                score = pos - neg
                confidence = max(pos, neg, neu)
                results.append((sentence, score, confidence))

        except Exception as e:
            logger.debug("FinBERT scoring error: %s", e)
            continue

    return results


def _score_with_keywords(sentences: List[str]) -> List[Tuple[str, float, float]]:
    """Fallback keyword-based sentiment scoring when FinBERT unavailable.

    Args:
        sentences: List of sentences.

    Returns:
        List of (sentence, score, confidence) tuples.
    """
    POSITIVE_WORDS = {
        "beat", "exceeded", "grew", "growth", "record", "strong", "bullish",
        "upgrade", "outperform", "buy", "positive", "profit", "revenue",
        "increase", "rose", "gained", "rally", "surge", "momentum", "recovery",
    }
    NEGATIVE_WORDS = {
        "miss", "declined", "fell", "weak", "bearish", "downgrade", "sell",
        "negative", "loss", "decrease", "dropped", "slump", "concern",
        "risk", "headwind", "competition", "lawsuit", "investigation",
    }

    results = []
    for sentence in sentences:
        words = set(sentence.lower().split())
        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)

        if pos_count + neg_count == 0:
            score = 0.0
            confidence = 0.3
        else:
            score = (pos_count - neg_count) / (pos_count + neg_count)
            confidence = min(0.7, (pos_count + neg_count) * 0.15)

        results.append((sentence, score, confidence))

    return results


def _aggregate_scores(
    scored: List[Tuple[str, float, float]],
) -> Tuple[float, float]:
    """Aggregate sentence scores into a single portfolio score.

    Uses confidence-weighted average: high-confidence sentences
    contribute more to the final score.

    Args:
        scored: List of (sentence, score, confidence) tuples.

    Returns:
        Tuple of (weighted_score, mean_confidence).
    """
    if not scored:
        return 0.0, 0.0

    total_weight = sum(conf for _, _, conf in scored)
    if total_weight == 0:
        return 0.0, 0.0

    weighted_score = sum(score * conf for _, score, conf in scored) / total_weight
    mean_confidence = total_weight / len(scored)

    # Clamp to [-1, 1]
    weighted_score = max(-1.0, min(1.0, weighted_score))

    return weighted_score, mean_confidence


def _score_to_label(score: float) -> str:
    """Convert numeric score to human-readable label.

    Args:
        score: Sentiment score from -1.0 to +1.0.

    Returns:
        'BULLISH', 'BEARISH', or 'NEUTRAL'.
    """
    if score > BULLISH_THRESHOLD:
        return "BULLISH"
    elif score < BEARISH_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def _neutral_state(state: TradingState) -> TradingState:
    """Return state with neutral sentiment (used on error or no data).

    Args:
        state: Current TradingState.

    Returns:
        State with neutral sentiment values.
    """
    return {
        **state,
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "sentiment_confidence": 0.0,
        "sentiment_details": {},
    }
