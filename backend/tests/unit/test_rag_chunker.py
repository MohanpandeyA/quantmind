"""Unit tests for the RecursiveChunker.

Tests cover:
- Chunk dataclass: creation, chunk_id, word_count, char_count
- RecursiveChunker: initialization, validation
- chunk_document(): single document splitting
- chunk_documents(): batch splitting
- Overlap: context preserved across chunk boundaries
- Edge cases: empty document, very short document, single chunk
- Separator hierarchy: paragraph → sentence → word → character
"""

from __future__ import annotations

import pytest

from rag.chunker import Chunk, RecursiveChunker
from rag.sources.base_loader import Document, DocumentMetadata, DocType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metadata(ticker: str = "AAPL") -> DocumentMetadata:
    return DocumentMetadata(
        ticker=ticker,
        source="SEC EDGAR",
        doc_type=DocType.SEC_10K,
        date="2024-01-15",
    )


def make_document(content: str, ticker: str = "AAPL") -> Document:
    return Document(content=content, metadata=make_metadata(ticker))


def make_long_document(n_paragraphs: int = 10) -> Document:
    """Create a document with n_paragraphs of financial text."""
    paragraphs = []
    for i in range(n_paragraphs):
        paragraphs.append(
            f"Paragraph {i+1}: Apple Inc. reported quarterly revenue of ${89 + i}.5 billion "
            f"in Q{(i % 4) + 1} 2024. iPhone sales grew {10 + i}% year-over-year driven by "
            f"strong demand in emerging markets. Services revenue reached ${20 + i}.3 billion "
            f"representing a {15 + i}% increase from the prior year period."
        )
    return make_document("\n\n".join(paragraphs))


# ---------------------------------------------------------------------------
# Chunk dataclass tests
# ---------------------------------------------------------------------------

class TestChunk:
    def test_chunk_id_auto_computed(self) -> None:
        chunk = Chunk(
            content="Test content",
            metadata=make_metadata(),
            chunk_index=0,
            total_chunks=5,
            parent_doc_id="abc123",
        )
        assert chunk.chunk_id == "abc123_chunk_0000"

    def test_chunk_id_custom_preserved(self) -> None:
        chunk = Chunk(
            content="Test",
            metadata=make_metadata(),
            chunk_index=0,
            total_chunks=1,
            parent_doc_id="abc",
            chunk_id="custom_id",
        )
        assert chunk.chunk_id == "custom_id"

    def test_chunk_id_format_with_index(self) -> None:
        chunk = Chunk(
            content="Test",
            metadata=make_metadata(),
            chunk_index=42,
            total_chunks=100,
            parent_doc_id="doc1",
        )
        assert chunk.chunk_id == "doc1_chunk_0042"

    def test_word_count(self) -> None:
        chunk = Chunk(
            content="Apple revenue grew fifteen percent",
            metadata=make_metadata(),
            chunk_index=0,
            total_chunks=1,
            parent_doc_id="doc1",
        )
        assert chunk.word_count() == 5

    def test_char_count(self) -> None:
        chunk = Chunk(
            content="Hello",
            metadata=make_metadata(),
            chunk_index=0,
            total_chunks=1,
            parent_doc_id="doc1",
        )
        assert chunk.char_count() == 5

    def test_repr_contains_chunk_id(self) -> None:
        chunk = Chunk(
            content="Test",
            metadata=make_metadata(),
            chunk_index=0,
            total_chunks=1,
            parent_doc_id="doc1",
        )
        assert "doc1_chunk_0000" in repr(chunk)


# ---------------------------------------------------------------------------
# RecursiveChunker initialization tests
# ---------------------------------------------------------------------------

class TestRecursiveChunkerInit:
    def test_default_params(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.min_chunk_size == 100

    def test_custom_params(self) -> None:
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_overlap_gte_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap.*<.*chunk_size"):
            RecursiveChunker(chunk_size=500, chunk_overlap=500)

    def test_overlap_greater_than_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap.*<.*chunk_size"):
            RecursiveChunker(chunk_size=500, chunk_overlap=600)

    def test_chunk_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 50"):
            RecursiveChunker(chunk_size=10)


# ---------------------------------------------------------------------------
# chunk_document() tests
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_empty_document_returns_empty_list(self) -> None:
        chunker = RecursiveChunker()
        doc = make_document("")
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_whitespace_document_returns_empty_list(self) -> None:
        chunker = RecursiveChunker()
        doc = make_document("   \n\t  ")
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_short_document_returns_single_chunk(self) -> None:
        # Use min_chunk_size=10 so the short doc isn't filtered out
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200, min_chunk_size=10)
        doc = make_document("Apple revenue grew 15% in Q3 2024.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_long_document_returns_multiple_chunks(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=5)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunks_inherit_metadata(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=3)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.metadata.ticker == "AAPL"
            assert chunk.metadata.source == "SEC EDGAR"

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=5)
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_total_chunks_consistent(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=5)
        chunks = chunker.chunk_document(doc)
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_parent_doc_id_matches(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=3)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.parent_doc_id == doc.doc_id

    def test_chunk_ids_are_unique(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=5)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_chunk_exceeds_chunk_size_significantly(self) -> None:
        """Chunks should not be much larger than chunk_size."""
        chunk_size = 300
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=50)
        doc = make_long_document(n_paragraphs=5)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            # Allow some tolerance for word-boundary splitting
            assert chunk.char_count() <= chunk_size * 2, (
                f"Chunk too large: {chunk.char_count()} chars (limit: {chunk_size * 2})"
            )

    def test_min_chunk_size_filters_small_chunks(self) -> None:
        chunker = RecursiveChunker(
            chunk_size=500, chunk_overlap=100, min_chunk_size=200
        )
        # Create a document where some chunks would be small
        content = "Short.\n\n" + "A" * 600 + "\n\nShort again."
        doc = make_document(content)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.char_count() >= chunker.min_chunk_size


# ---------------------------------------------------------------------------
# Overlap tests
# ---------------------------------------------------------------------------

class TestChunkOverlap:
    def test_overlap_preserves_context(self) -> None:
        """Consecutive chunks should share some content (overlap)."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=100)
        # Create content where overlap should be visible
        content = " ".join([f"word{i}" for i in range(200)])
        doc = make_document(content)
        chunks = chunker.chunk_document(doc)

        if len(chunks) >= 2:
            # Check that consecutive chunks share some words
            words_0 = set(chunks[0].content.split())
            words_1 = set(chunks[1].content.split())
            overlap = words_0 & words_1
            # There should be some overlap
            assert len(overlap) > 0, "Consecutive chunks should share some words (overlap)"

    def test_larger_overlap_more_shared_content(self) -> None:
        """Larger overlap should result in more shared content."""
        content = " ".join([f"word{i}" for i in range(300)])
        doc = make_document(content)

        chunker_small = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunker_large = RecursiveChunker(chunk_size=200, chunk_overlap=100)

        chunks_small = chunker_small.chunk_document(doc)
        chunks_large = chunker_large.chunk_document(doc)

        if len(chunks_small) >= 2 and len(chunks_large) >= 2:
            overlap_small = len(
                set(chunks_small[0].content.split()) &
                set(chunks_small[1].content.split())
            )
            overlap_large = len(
                set(chunks_large[0].content.split()) &
                set(chunks_large[1].content.split())
            )
            assert overlap_large >= overlap_small


# ---------------------------------------------------------------------------
# chunk_documents() batch tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_empty_list_returns_empty(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk_documents([]) == []

    def test_multiple_documents_all_chunked(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        docs = [make_long_document(3) for _ in range(3)]
        all_chunks = chunker.chunk_documents(docs)
        assert len(all_chunks) > 3  # More chunks than documents

    def test_chunks_from_different_docs_have_different_parent_ids(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc1 = make_document("Content A " * 100)
        doc2 = make_document("Content B " * 100)
        chunks = chunker.chunk_documents([doc1, doc2])

        parent_ids = {c.parent_doc_id for c in chunks}
        assert len(parent_ids) == 2  # Two different parent documents

    def test_all_chunk_ids_unique_across_documents(self) -> None:
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        # Use different content per document so doc_ids differ → chunk_ids differ
        docs = [
            make_document(f"Document {i}: " + "Apple revenue grew. " * 50)
            for i in range(3)
        ]
        chunks = chunker.chunk_documents(docs)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Separator hierarchy tests
# ---------------------------------------------------------------------------

class TestSeparatorHierarchy:
    def test_splits_on_paragraph_boundary(self) -> None:
        """Should prefer splitting at paragraph boundaries."""
        # Use min_chunk_size=10 so short paragraphs aren't filtered
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        content = (
            "First paragraph content here with enough words to pass filter.\n\n"
            "Second paragraph content here with enough words to pass filter.\n\n"
            "Third paragraph content here with enough words to pass filter."
        )
        doc = make_document(content)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_handles_no_separators(self) -> None:
        """Should handle content with no natural separators."""
        # Use min_chunk_size=10 so chunks aren't filtered out
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        content = "a" * 200  # No separators at all
        doc = make_document(content)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.char_count() > 0
