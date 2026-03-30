"""Text chunker for the QuantMind RAG pipeline.

Splits long financial documents into overlapping chunks suitable for embedding.

Why chunking matters:
    - LLMs have context limits (e.g., 8K tokens for Groq Llama)
    - A 10-K filing is 50,000+ words — can't embed it whole
    - Chunks must be small enough to embed but large enough to be meaningful
    - Overlap prevents context loss at chunk boundaries

Chunking strategy (RecursiveCharacterTextSplitter):
    Try to split on paragraph boundaries first (\n\n),
    then sentence boundaries (\n, ". "),
    then word boundaries (" "),
    finally character boundaries as last resort.

    This preserves semantic coherence — a chunk about "iPhone revenue"
    won't be split mid-sentence.

Overlap strategy:
    Each chunk overlaps with the previous by `chunk_overlap` characters.
    This ensures context is preserved across boundaries:

    Without overlap:
        Chunk 1: "...revenue grew 15% in Q3. The primary driver was..."
        Chunk 2: "iPhone sales in China, which increased..."
        → "primary driver" and "iPhone sales" are disconnected

    With 200-char overlap:
        Chunk 1: "...revenue grew 15% in Q3. The primary driver was iPhone sales..."
        Chunk 2: "...primary driver was iPhone sales in China, which increased 23%..."
        → Context preserved!

Optimization principles:
    - Sentence-aware splitting: never cut mid-sentence
    - Configurable chunk size: tune for your embedding model's optimal input
    - Metadata inheritance: each chunk inherits parent document metadata
    - Chunk index tracking: enables ordered reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config.logging_config import get_logger
from rag.sources.base_loader import Document, DocumentMetadata

logger = get_logger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000      # Characters per chunk (~200-250 tokens)
DEFAULT_CHUNK_OVERLAP = 200    # Overlap between consecutive chunks
DEFAULT_MIN_CHUNK_SIZE = 100   # Skip chunks smaller than this

# Separators tried in order (most preferred → least preferred)
# This is the "recursive" part of RecursiveCharacterTextSplitter
SEPARATORS = [
    "\n\n",    # Paragraph break (best — preserves semantic units)
    "\n",      # Line break
    ". ",      # Sentence end
    "! ",      # Exclamation sentence
    "? ",      # Question sentence
    "; ",      # Semicolon
    ", ",      # Comma
    " ",       # Word boundary
    "",        # Character boundary (last resort)
]


@dataclass
class Chunk:
    """A single text chunk ready for embedding.

    Attributes:
        content: Text content of this chunk.
        metadata: Inherited from parent Document, with chunk-specific additions.
        chunk_index: Position of this chunk within the parent document (0-based).
        total_chunks: Total number of chunks from the parent document.
        parent_doc_id: doc_id of the parent Document.
        chunk_id: Unique identifier for this chunk (parent_id + index).
    """

    content: str
    metadata: DocumentMetadata
    chunk_index: int
    total_chunks: int
    parent_doc_id: str
    chunk_id: str = field(default="")

    def __post_init__(self) -> None:
        """Auto-compute chunk_id if not provided."""
        if not self.chunk_id:
            self.chunk_id = f"{self.parent_doc_id}_chunk_{self.chunk_index:04d}"

    def word_count(self) -> int:
        """Return approximate word count.

        Returns:
            Number of whitespace-separated tokens.
        """
        return len(self.content.split())

    def char_count(self) -> int:
        """Return character count.

        Returns:
            Number of characters in content.
        """
        return len(self.content)

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"ticker={self.metadata.ticker!r}, "
            f"index={self.chunk_index}/{self.total_chunks}, "
            f"chars={self.char_count()})"
        )


class RecursiveChunker:
    """Splits documents into overlapping chunks using recursive character splitting.

    Tries to split on semantic boundaries (paragraphs, sentences) before
    falling back to character-level splitting. This preserves meaning
    better than fixed-size splitting.

    Attributes:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        min_chunk_size: Minimum chunk size — smaller chunks are discarded.
        separators: List of separators to try, in order of preference.

    Example:
        >>> chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        >>> chunks = chunker.chunk_document(document)
        >>> print(f"Split into {len(chunks)} chunks")
        Split into 47 chunks
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
        separators: Optional[List[str]] = None,
    ) -> None:
        """Initialize RecursiveChunker.

        Args:
            chunk_size: Target chunk size in characters. Default 1000.
                        Rule of thumb: 1000 chars ≈ 200-250 tokens.
                        Most embedding models work best with 200-500 tokens.
            chunk_overlap: Overlap between consecutive chunks. Default 200.
                           20% of chunk_size is a good rule of thumb.
            min_chunk_size: Discard chunks smaller than this. Default 100.
            separators: Custom separator list. Default: SEPARATORS.

        Raises:
            ValueError: If chunk_overlap >= chunk_size.
        """
        if chunk_size < 50:
            raise ValueError(f"chunk_size must be >= 50, got {chunk_size}.")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or SEPARATORS

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into overlapping Chunks.

        Args:
            document: Document to split.

        Returns:
            List of Chunk objects. Empty list if document is empty.

        Example:
            >>> doc = Document(content="Long financial text...", metadata=...)
            >>> chunks = chunker.chunk_document(doc)
            >>> chunks[0].chunk_index
            0
            >>> chunks[0].total_chunks
            47
        """
        if document.is_empty():
            logger.debug("RecursiveChunker | skipping empty document | %s", document)
            return []

        # Split text into raw text pieces
        text_pieces = self._split_text(document.content)

        # Merge pieces into chunks with overlap
        raw_chunks = self._merge_with_overlap(text_pieces)

        # Filter out chunks that are too small
        raw_chunks = [c for c in raw_chunks if len(c) >= self.min_chunk_size]

        if not raw_chunks:
            logger.debug(
                "RecursiveChunker | no valid chunks | doc_id=%s", document.doc_id
            )
            return []

        total = len(raw_chunks)
        chunks: List[Chunk] = []

        for i, text in enumerate(raw_chunks):
            chunk = Chunk(
                content=text.strip(),
                metadata=document.metadata,
                chunk_index=i,
                total_chunks=total,
                parent_doc_id=document.doc_id,
            )
            chunks.append(chunk)

        logger.debug(
            "RecursiveChunker | chunked | doc_id=%s | chunks=%d | avg_chars=%.0f",
            document.doc_id,
            total,
            sum(c.char_count() for c in chunks) / total if total > 0 else 0,
        )
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple Documents into Chunks.

        Args:
            documents: List of Documents to split.

        Returns:
            Flat list of all Chunks from all documents.

        Example:
            >>> all_chunks = chunker.chunk_documents(documents)
            >>> print(f"Total chunks: {len(all_chunks)}")
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        total_words = sum(c.word_count() for c in all_chunks)
        logger.info(
            "RecursiveChunker | batch chunked | docs=%d | chunks=%d | total_words=%d",
            len(documents), len(all_chunks), total_words,
        )
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursively split text using the separator hierarchy.

        Tries each separator in order. If a piece is still larger than
        chunk_size after splitting, recursively splits it with the next
        separator.

        Args:
            text: Text to split.

        Returns:
            List of text pieces, each <= chunk_size characters.
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text with fallback separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text pieces.
        """
        if not text:
            return []

        # Base case: text fits in one chunk
        if len(text) <= self.chunk_size:
            return [text]

        # No more separators — force split at chunk_size
        if not separators:
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Try splitting with current separator
        if separator == "":
            # Character-level split (last resort)
            return self._force_split(text)

        parts = text.split(separator)

        if len(parts) == 1:
            # Separator not found — try next separator
            return self._recursive_split(text, remaining_separators)

        # Recursively split any parts that are still too large
        result: List[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # Part is still too large — recurse with next separator
                sub_parts = self._recursive_split(part, remaining_separators)
                result.extend(sub_parts)

        return result

    def _force_split(self, text: str) -> List[str]:
        """Force-split text at chunk_size boundaries (last resort).

        Used when no semantic separator works. Splits at word boundaries
        near the chunk_size limit.

        Args:
            text: Text to split.

        Returns:
            List of text pieces.
        """
        pieces: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                pieces.append(text[start:])
                break

            # Try to find a word boundary near the end
            # Look back up to 50 chars for a space
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

            pieces.append(text[start:end])
            start = end

        return pieces

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Merge text pieces into chunks with overlap.

        Takes the split pieces and combines them into chunks of approximately
        chunk_size characters, with chunk_overlap characters of overlap
        between consecutive chunks.

        Args:
            pieces: List of text pieces from _split_text().

        Returns:
            List of chunk strings with overlap applied.
        """
        if not pieces:
            return []

        chunks: List[str] = []
        current_pieces: List[str] = []
        current_size = 0

        for piece in pieces:
            piece_size = len(piece)

            # If adding this piece would exceed chunk_size, finalize current chunk
            if current_size + piece_size > self.chunk_size and current_pieces:
                chunk_text = " ".join(current_pieces).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Keep overlap: remove pieces from the front until we're
                # within chunk_overlap of the chunk_size
                while current_pieces and current_size > self.chunk_overlap:
                    removed = current_pieces.pop(0)
                    current_size -= len(removed) + 1  # +1 for space

            current_pieces.append(piece)
            current_size += piece_size + 1  # +1 for space separator

        # Add the final chunk
        if current_pieces:
            chunk_text = " ".join(current_pieces).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks
