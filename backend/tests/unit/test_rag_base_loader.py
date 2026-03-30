"""Unit tests for RAG base_loader — Document, DocumentMetadata, BaseLoader.

Tests cover:
- Document dataclass: creation, hash, word_count, is_empty
- DocumentMetadata: to_chroma_dict, from_chroma_dict round-trip
- DocType enum values
- BaseLoader: abstract enforcement, validate_query
- LoaderError / RateLimitError exceptions
"""

from __future__ import annotations

import pytest

from rag.sources.base_loader import (
    BaseLoader,
    Document,
    DocumentMetadata,
    DocType,
    LoaderError,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metadata(
    ticker: str = "AAPL",
    source: str = "SEC EDGAR",
    doc_type: DocType = DocType.SEC_10K,
    date: str = "2024-01-15",
    url: str = "https://sec.gov/test",
    title: str = "Apple 10-K 2024",
) -> DocumentMetadata:
    return DocumentMetadata(
        ticker=ticker,
        source=source,
        doc_type=doc_type,
        date=date,
        url=url,
        title=title,
    )


def make_document(
    content: str = "Apple reported revenue of $89.5B in Q3 2024.",
    ticker: str = "AAPL",
) -> Document:
    return Document(
        content=content,
        metadata=make_metadata(ticker=ticker),
    )


# ---------------------------------------------------------------------------
# DocType enum tests
# ---------------------------------------------------------------------------

class TestDocType:
    def test_sec_10k_value(self) -> None:
        assert DocType.SEC_10K.value == "10-K"

    def test_sec_10q_value(self) -> None:
        assert DocType.SEC_10Q.value == "10-Q"

    def test_sec_8k_value(self) -> None:
        assert DocType.SEC_8K.value == "8-K"

    def test_news_value(self) -> None:
        assert DocType.NEWS.value == "news"

    def test_rss_value(self) -> None:
        assert DocType.RSS.value == "rss"

    def test_pdf_value(self) -> None:
        assert DocType.PDF.value == "pdf"

    def test_unknown_value(self) -> None:
        assert DocType.UNKNOWN.value == "unknown"

    def test_from_string(self) -> None:
        assert DocType("10-K") == DocType.SEC_10K
        assert DocType("news") == DocType.NEWS


# ---------------------------------------------------------------------------
# DocumentMetadata tests
# ---------------------------------------------------------------------------

class TestDocumentMetadata:
    def test_basic_creation(self) -> None:
        meta = make_metadata()
        assert meta.ticker == "AAPL"
        assert meta.source == "SEC EDGAR"
        assert meta.doc_type == DocType.SEC_10K
        assert meta.date == "2024-01-15"

    def test_default_values(self) -> None:
        meta = DocumentMetadata(
            ticker="MSFT",
            source="NewsAPI",
            doc_type=DocType.NEWS,
            date="2024-06-01",
        )
        assert meta.url == ""
        assert meta.title == ""
        assert meta.filing_period == ""
        assert meta.page_number == 0
        assert meta.extra == {}

    def test_to_chroma_dict_has_required_keys(self) -> None:
        meta = make_metadata()
        d = meta.to_chroma_dict()
        required_keys = {"ticker", "source", "doc_type", "date", "url", "title"}
        assert required_keys.issubset(set(d.keys()))

    def test_to_chroma_dict_values_are_strings(self) -> None:
        meta = make_metadata()
        d = meta.to_chroma_dict()
        for key, val in d.items():
            assert isinstance(val, str), f"Key {key!r} has non-string value: {val!r}"

    def test_to_chroma_dict_ticker_uppercase(self) -> None:
        meta = make_metadata(ticker="aapl")
        d = meta.to_chroma_dict()
        assert d["ticker"] == "aapl"  # Stored as-is, not uppercased in metadata

    def test_to_chroma_dict_doc_type_is_string(self) -> None:
        meta = make_metadata(doc_type=DocType.SEC_10K)
        d = meta.to_chroma_dict()
        assert d["doc_type"] == "10-K"

    def test_to_chroma_dict_title_truncated(self) -> None:
        long_title = "A" * 600
        meta = make_metadata(title=long_title)
        d = meta.to_chroma_dict()
        assert len(d["title"]) <= 500

    def test_to_chroma_dict_extra_fields_prefixed(self) -> None:
        meta = DocumentMetadata(
            ticker="AAPL",
            source="SEC",
            doc_type=DocType.SEC_10K,
            date="2024-01-01",
            extra={"cik": "0000320193", "accession": "0001234567"},
        )
        d = meta.to_chroma_dict()
        assert "extra_cik" in d
        assert "extra_accession" in d
        assert d["extra_cik"] == "0000320193"

    def test_from_chroma_dict_round_trip(self) -> None:
        original = make_metadata(
            ticker="MSFT",
            source="Reuters",
            doc_type=DocType.NEWS,
            date="2024-06-15",
            url="https://reuters.com/test",
            title="Microsoft Q2 Results",
        )
        d = original.to_chroma_dict()
        restored = DocumentMetadata.from_chroma_dict(d)

        assert restored.ticker == original.ticker
        assert restored.source == original.source
        assert restored.doc_type == original.doc_type
        assert restored.date == original.date
        assert restored.url == original.url
        assert restored.title == original.title

    def test_from_chroma_dict_with_extra(self) -> None:
        meta = DocumentMetadata(
            ticker="AAPL",
            source="SEC",
            doc_type=DocType.SEC_10K,
            date="2024-01-01",
            extra={"cik": "0000320193"},
        )
        d = meta.to_chroma_dict()
        restored = DocumentMetadata.from_chroma_dict(d)
        assert restored.extra.get("cik") == "0000320193"

    def test_from_chroma_dict_unknown_doc_type(self) -> None:
        d = {
            "ticker": "AAPL",
            "source": "Test",
            "doc_type": "unknown",
            "date": "2024-01-01",
        }
        meta = DocumentMetadata.from_chroma_dict(d)
        assert meta.doc_type == DocType.UNKNOWN


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------

class TestDocument:
    def test_basic_creation(self) -> None:
        doc = make_document()
        assert doc.content == "Apple reported revenue of $89.5B in Q3 2024."
        assert doc.metadata.ticker == "AAPL"

    def test_doc_id_auto_computed(self) -> None:
        doc = make_document()
        assert doc.doc_id != ""
        assert len(doc.doc_id) == 16  # First 16 chars of SHA-256

    def test_same_content_same_doc_id(self) -> None:
        doc1 = make_document(content="Same content here.")
        doc2 = make_document(content="Same content here.")
        assert doc1.doc_id == doc2.doc_id

    def test_different_content_different_doc_id(self) -> None:
        doc1 = make_document(content="Content A")
        doc2 = make_document(content="Content B")
        assert doc1.doc_id != doc2.doc_id

    def test_custom_doc_id_preserved(self) -> None:
        doc = Document(
            content="Test",
            metadata=make_metadata(),
            doc_id="custom_id_123",
        )
        assert doc.doc_id == "custom_id_123"

    def test_word_count(self) -> None:
        doc = make_document(content="Apple revenue grew fifteen percent")
        assert doc.word_count() == 5

    def test_word_count_empty(self) -> None:
        doc = make_document(content="")
        assert doc.word_count() == 0

    def test_is_empty_true_for_empty_content(self) -> None:
        doc = make_document(content="")
        assert doc.is_empty() is True

    def test_is_empty_true_for_whitespace(self) -> None:
        doc = make_document(content="   \n\t  ")
        assert doc.is_empty() is True

    def test_is_empty_false_for_content(self) -> None:
        doc = make_document(content="Apple revenue grew.")
        assert doc.is_empty() is False

    def test_repr_contains_ticker(self) -> None:
        doc = make_document()
        assert "AAPL" in repr(doc)

    def test_repr_contains_doc_type(self) -> None:
        doc = make_document()
        assert "10-K" in repr(doc)


# ---------------------------------------------------------------------------
# BaseLoader abstract enforcement tests
# ---------------------------------------------------------------------------

class TestBaseLoaderAbstract:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseLoader()  # type: ignore[abstract]

    def test_subclass_without_load_raises(self) -> None:
        class IncompleteLoader(BaseLoader):
            def get_source_name(self) -> str:
                return "Incomplete"
            # Missing load()

        with pytest.raises(TypeError):
            IncompleteLoader()  # type: ignore[abstract]

    def test_subclass_without_get_source_name_raises(self) -> None:
        import asyncio
        from typing import List

        class IncompleteLoader(BaseLoader):
            async def load(self, ticker: str, **kwargs: object) -> List[Document]:
                return []
            # Missing get_source_name()

        with pytest.raises(TypeError):
            IncompleteLoader()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# BaseLoader validate_query tests
# ---------------------------------------------------------------------------

class TestBaseLoaderValidation:
    """Tests for validate_query() using a concrete subclass."""

    def setup_method(self) -> None:
        import asyncio
        from typing import List

        class ConcreteLoader(BaseLoader):
            def get_source_name(self) -> str:
                return "TestLoader"

            async def load(self, ticker: str, **kwargs: object) -> List[Document]:
                return []

        self.loader = ConcreteLoader()

    def test_valid_ticker_passes(self) -> None:
        self.loader.validate_query("AAPL")  # Should not raise

    def test_valid_ticker_with_dot(self) -> None:
        self.loader.validate_query("RELIANCE.NS")  # Should not raise

    def test_empty_ticker_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.loader.validate_query("")

    def test_whitespace_ticker_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.loader.validate_query("   ")

    def test_too_long_ticker_raises(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            self.loader.validate_query("A" * 21)

    def test_repr_contains_source_name(self) -> None:
        assert "TestLoader" in repr(self.loader)


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_loader_error_message(self) -> None:
        err = LoaderError("SEC EDGAR", "AAPL", "Connection timeout")
        assert "SEC EDGAR" in str(err)
        assert "AAPL" in str(err)
        assert "Connection timeout" in str(err)

    def test_loader_error_attributes(self) -> None:
        err = LoaderError("NewsAPI", "MSFT", "Rate limit exceeded")
        assert err.source == "NewsAPI"
        assert err.ticker == "MSFT"
        assert err.message == "Rate limit exceeded"

    def test_rate_limit_error_is_loader_error(self) -> None:
        err = RateLimitError("NewsAPI", "AAPL", "100 req/day exceeded")
        assert isinstance(err, LoaderError)

    def test_rate_limit_error_message(self) -> None:
        err = RateLimitError("NewsAPI", "AAPL", "100 req/day exceeded")
        assert "NewsAPI" in str(err)
        assert "AAPL" in str(err)
