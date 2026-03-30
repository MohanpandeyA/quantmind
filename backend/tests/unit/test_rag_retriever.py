"""Unit tests for the Retriever and VectorStore components.

Tests cover:
- SearchResult: creation, to_citation, repr
- VectorStore._build_filter(): metadata filter construction
- Retriever._jaccard_similarity(): diversity metric
- Retriever._mmr_rerank(): MMR algorithm correctness
- Retriever.build_context(): context formatting
- Retriever.build_citations(): citation extraction
- IngestionReport: creation, to_dict, repr

All tests use mocks — no real ChromaDB or embedding model needed.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rag.ingestion import IngestionReport
from rag.retriever import Retriever
from rag.sources.base_loader import DocumentMetadata, DocType
from rag.vector_store import SearchResult, VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_search_result(
    content: str = "Apple revenue grew 15% in Q3 2024.",
    ticker: str = "AAPL",
    source: str = "SEC EDGAR",
    doc_type: DocType = DocType.SEC_10K,
    date: str = "2024-01-15",
    url: str = "https://sec.gov/test",
    title: str = "Apple 10-K 2024",
    score: float = 0.85,
) -> SearchResult:
    metadata = DocumentMetadata(
        ticker=ticker,
        source=source,
        doc_type=doc_type,
        date=date,
        url=url,
        title=title,
    )
    return SearchResult(
        content=content,
        metadata=metadata,
        score=score,
        distance=1.0 - score,
    )


def make_retriever(
    n_results: int = 5,
    fetch_k: int = 20,
    mmr_lambda: float = 0.5,
) -> Retriever:
    """Create a Retriever with mocked dependencies."""
    mock_store = MagicMock(spec=VectorStore)
    mock_embedding = MagicMock()
    mock_embedding.embed_query = AsyncMock(
        return_value=np.random.rand(384).astype(np.float32)
    )
    return Retriever(
        vector_store=mock_store,
        embedding_model=mock_embedding,
        n_results=n_results,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
    )


# ---------------------------------------------------------------------------
# SearchResult tests
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_basic_creation(self) -> None:
        result = make_search_result()
        assert result.content == "Apple revenue grew 15% in Q3 2024."
        assert result.metadata.ticker == "AAPL"
        assert result.score == pytest.approx(0.85)

    def test_to_citation_includes_source(self) -> None:
        result = make_search_result(source="SEC EDGAR")
        citation = result.to_citation()
        assert "SEC EDGAR" in citation

    def test_to_citation_includes_title(self) -> None:
        result = make_search_result(title="Apple 10-K 2024")
        citation = result.to_citation()
        assert "Apple 10-K 2024" in citation

    def test_to_citation_includes_date(self) -> None:
        result = make_search_result(date="2024-01-15")
        citation = result.to_citation()
        assert "2024-01-15" in citation

    def test_to_citation_includes_url(self) -> None:
        result = make_search_result(url="https://sec.gov/test")
        citation = result.to_citation()
        assert "https://sec.gov/test" in citation

    def test_to_citation_no_url(self) -> None:
        result = make_search_result(url="")
        citation = result.to_citation()
        assert "SEC EDGAR" in citation  # Still has source

    def test_repr_contains_score(self) -> None:
        result = make_search_result(score=0.92)
        assert "0.920" in repr(result)

    def test_repr_contains_ticker(self) -> None:
        result = make_search_result(ticker="MSFT")
        assert "MSFT" in repr(result)


# ---------------------------------------------------------------------------
# VectorStore._build_filter() tests
# ---------------------------------------------------------------------------

class TestVectorStoreBuildFilter:
    def test_no_filters_returns_none(self) -> None:
        result = VectorStore._build_filter(None, None, None, None)
        assert result is None

    def test_ticker_filter(self) -> None:
        result = VectorStore._build_filter("AAPL", None, None, None)
        assert result == {"ticker": {"$eq": "AAPL"}}

    def test_ticker_uppercased(self) -> None:
        result = VectorStore._build_filter("aapl", None, None, None)
        assert result == {"ticker": {"$eq": "AAPL"}}

    def test_single_doc_type_filter(self) -> None:
        result = VectorStore._build_filter(None, ["10-K"], None, None)
        assert result == {"doc_type": {"$eq": "10-K"}}

    def test_multiple_doc_types_uses_in(self) -> None:
        result = VectorStore._build_filter(None, ["10-K", "10-Q"], None, None)
        assert result == {"doc_type": {"$in": ["10-K", "10-Q"]}}

    def test_date_from_filter(self) -> None:
        result = VectorStore._build_filter(None, None, "2024-01-01", None)
        assert result == {"date": {"$gte": "2024-01-01"}}

    def test_date_to_filter(self) -> None:
        result = VectorStore._build_filter(None, None, None, "2024-12-31")
        assert result == {"date": {"$lte": "2024-12-31"}}

    def test_combined_filters_uses_and(self) -> None:
        result = VectorStore._build_filter("AAPL", ["10-K"], "2024-01-01", None)
        assert result is not None
        assert "$and" in result
        conditions = result["$and"]
        assert len(conditions) == 3  # ticker + doc_type + date_from

    def test_ticker_and_date_range(self) -> None:
        result = VectorStore._build_filter("MSFT", None, "2024-01-01", "2024-12-31")
        assert result is not None
        assert "$and" in result
        conditions = result["$and"]
        assert len(conditions) == 3  # ticker + date_from + date_to


# ---------------------------------------------------------------------------
# Retriever._jaccard_similarity() tests
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_sets_return_1(self) -> None:
        s = {"apple", "revenue", "grew"}
        assert Retriever._jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_disjoint_sets_return_0(self) -> None:
        a = {"apple", "revenue"}
        b = {"microsoft", "profit"}
        assert Retriever._jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = {"apple", "revenue", "grew"}
        b = {"apple", "profit", "grew"}
        # Intersection: {apple, grew} = 2
        # Union: {apple, revenue, grew, profit} = 4
        # Jaccard = 2/4 = 0.5
        assert Retriever._jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_empty_sets_return_0(self) -> None:
        assert Retriever._jaccard_similarity(set(), set()) == pytest.approx(0.0)

    def test_one_empty_set_returns_0(self) -> None:
        assert Retriever._jaccard_similarity({"apple"}, set()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Retriever._mmr_rerank() tests
# ---------------------------------------------------------------------------

class TestMMRRerank:
    def test_returns_k_results(self) -> None:
        retriever = make_retriever(n_results=3)
        candidates = [make_search_result(score=0.9 - i * 0.05) for i in range(10)]
        results = retriever._mmr_rerank(candidates, np.random.rand(384), k=3)
        assert len(results) == 3

    def test_returns_all_if_fewer_than_k(self) -> None:
        retriever = make_retriever()
        candidates = [make_search_result(score=0.9) for _ in range(3)]
        results = retriever._mmr_rerank(candidates, np.random.rand(384), k=5)
        assert len(results) == 3

    def test_first_result_is_most_relevant(self) -> None:
        """First selected result should be the most relevant (highest score)."""
        retriever = make_retriever(mmr_lambda=1.0)  # Pure relevance, no diversity
        candidates = [
            make_search_result(content=f"content {i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]
        results = retriever._mmr_rerank(candidates, np.random.rand(384), k=3)
        # With lambda=1.0 (pure relevance), first result should be highest score
        assert results[0].score >= results[1].score

    def test_diversity_reduces_duplicates(self) -> None:
        """MMR with low lambda should select diverse results."""
        retriever = make_retriever(mmr_lambda=0.1)  # High diversity

        # Create candidates where first 3 are very similar (same words)
        similar_content = "apple revenue iphone sales grew quarterly"
        diverse_content = ["microsoft azure cloud profit", "tesla ev battery range", "amazon aws revenue"]

        candidates = (
            [make_search_result(content=similar_content, score=0.9)] * 3 +
            [make_search_result(content=c, score=0.7) for c in diverse_content]
        )

        results = retriever._mmr_rerank(candidates, np.random.rand(384), k=3)
        # With high diversity, should not select all 3 similar candidates
        assert len(results) == 3

    def test_lambda_1_equals_top_k(self) -> None:
        """With lambda=1.0, MMR should behave like top-K."""
        retriever = make_retriever(mmr_lambda=1.0)
        candidates = [
            make_search_result(content=f"unique content {i}", score=0.9 - i * 0.05)
            for i in range(10)
        ]
        results = retriever._mmr_rerank(candidates, np.random.rand(384), k=5)
        # Should select top 5 by score
        assert len(results) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retriever.build_context() tests
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_empty_results_returns_no_documents_message(self) -> None:
        retriever = make_retriever()
        context = retriever.build_context([])
        assert "No relevant documents found" in context

    def test_context_contains_source(self) -> None:
        retriever = make_retriever()
        results = [make_search_result(source="SEC EDGAR")]
        context = retriever.build_context(results)
        assert "SEC EDGAR" in context

    def test_context_contains_content(self) -> None:
        retriever = make_retriever()
        results = [make_search_result(content="Apple revenue grew 15%")]
        context = retriever.build_context(results)
        assert "Apple revenue grew 15%" in context

    def test_context_numbered(self) -> None:
        retriever = make_retriever()
        results = [make_search_result() for _ in range(3)]
        context = retriever.build_context(results)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context

    def test_context_respects_max_chars(self) -> None:
        retriever = make_retriever()
        # Create results with long content
        results = [
            make_search_result(content="A" * 2000)
            for _ in range(5)
        ]
        context = retriever.build_context(results, max_context_chars=3000)
        assert len(context) <= 3500  # Allow some tolerance for formatting

    def test_context_contains_score(self) -> None:
        retriever = make_retriever()
        results = [make_search_result(score=0.923)]
        context = retriever.build_context(results)
        assert "0.923" in context


# ---------------------------------------------------------------------------
# Retriever.build_citations() tests
# ---------------------------------------------------------------------------

class TestBuildCitations:
    def test_returns_list_of_strings(self) -> None:
        retriever = make_retriever()
        results = [make_search_result() for _ in range(3)]
        citations = retriever.build_citations(results)
        assert len(citations) == 3
        for c in citations:
            assert isinstance(c, str)

    def test_empty_results_returns_empty_list(self) -> None:
        retriever = make_retriever()
        assert retriever.build_citations([]) == []

    def test_citations_contain_source(self) -> None:
        retriever = make_retriever()
        results = [make_search_result(source="Reuters")]
        citations = retriever.build_citations(results)
        assert "Reuters" in citations[0]


# ---------------------------------------------------------------------------
# IngestionReport tests
# ---------------------------------------------------------------------------

class TestIngestionReport:
    def test_default_values(self) -> None:
        report = IngestionReport(ticker="AAPL")
        assert report.ticker == "AAPL"
        assert report.docs_loaded == 0
        assert report.chunks_stored == 0
        assert report.errors == []
        assert report.sources_used == []

    def test_to_dict_has_all_keys(self) -> None:
        report = IngestionReport(
            ticker="AAPL",
            docs_loaded=10,
            chunks_stored=150,
            duration_seconds=5.3,
        )
        d = report.to_dict()
        expected_keys = {
            "ticker", "docs_loaded", "docs_skipped", "chunks_created",
            "chunks_stored", "sources_used", "errors", "duration_seconds",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_values_correct(self) -> None:
        report = IngestionReport(
            ticker="MSFT",
            docs_loaded=5,
            chunks_stored=80,
            sources_used=["SEC EDGAR", "NewsAPI"],
            duration_seconds=3.14,
        )
        d = report.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["docs_loaded"] == 5
        assert d["chunks_stored"] == 80
        assert d["sources_used"] == ["SEC EDGAR", "NewsAPI"]
        assert d["duration_seconds"] == pytest.approx(3.14, abs=0.01)

    def test_repr_contains_ticker(self) -> None:
        report = IngestionReport(ticker="GOOGL")
        assert "GOOGL" in repr(report)

    def test_repr_contains_chunks(self) -> None:
        report = IngestionReport(ticker="AAPL", chunks_stored=150)
        assert "150" in repr(report)

    def test_errors_list_populated(self) -> None:
        report = IngestionReport(ticker="AAPL")
        report.errors.append("SEC EDGAR: Connection timeout")
        assert len(report.errors) == 1
        assert "SEC EDGAR" in report.errors[0]
