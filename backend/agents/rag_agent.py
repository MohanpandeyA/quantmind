"""RAGAgent — retrieves relevant financial documents for a ticker.

This is the SECOND agent in the LangGraph workflow. It:
1. Triggers document ingestion if ticker not yet in ChromaDB
2. Retrieves the most relevant chunks using semantic search + MMR
3. Builds a formatted context string for the ExplainerAgent
4. Extracts source citations for the final answer

Connects Phase 2 (RAG Pipeline) to Phase 3 (LangGraph Agents).

LangGraph node contract:
    Input:  TradingState with ticker and query set
    Output: TradingState with retrieved_docs, rag_context, citations populated
"""

from __future__ import annotations

from config.logging_config import get_logger
from graph.state import TradingState
from rag.embeddings import EmbeddingModel
from rag.ingestion import IngestionPipeline
from rag.retriever import Retriever
from rag.vector_store import VectorStore

logger = get_logger(__name__)

# Singleton instances — created once, reused across requests
_vector_store: VectorStore | None = None
_embedding_model: EmbeddingModel | None = None
_retriever: Retriever | None = None
_ingestion_pipeline: IngestionPipeline | None = None


def _get_retriever() -> Retriever:
    """Lazy-initialize and return the singleton Retriever.

    Returns:
        Configured Retriever instance.
    """
    global _vector_store, _embedding_model, _retriever, _ingestion_pipeline

    if _retriever is None:
        _vector_store = VectorStore()
        _embedding_model = EmbeddingModel()
        _retriever = Retriever(
            vector_store=_vector_store,
            embedding_model=_embedding_model,
            n_results=5,
            fetch_k=20,
            mmr_lambda=0.5,
        )
        _ingestion_pipeline = IngestionPipeline.create_default()
        logger.info("RAGAgent | retriever initialized")

    return _retriever


def _get_ingestion_pipeline() -> IngestionPipeline:
    """Return the singleton IngestionPipeline.

    Returns:
        Configured IngestionPipeline instance.
    """
    _get_retriever()  # Ensures pipeline is initialized
    return _ingestion_pipeline  # type: ignore[return-value]


async def rag_agent(state: TradingState) -> TradingState:
    """Retrieve relevant financial documents. LangGraph node function.

    Steps:
        1. Check if ticker has documents in ChromaDB
        2. If not, trigger ingestion (SEC + News + RSS)
        3. Retrieve top-5 relevant chunks using MMR
        4. Build formatted context for ExplainerAgent
        5. Extract citations for final answer

    Args:
        state: Current TradingState with ticker and query set.

    Returns:
        Updated TradingState with retrieved_docs, rag_context, citations.

    Example:
        >>> state = await rag_agent(state)
        >>> print(state["rag_context"][:200])
        [1] Source: SEC EDGAR | Apple 10-K (2024-11-01) | Score: 0.892
        iPhone revenue declined 8% in Q3 2024...
    """
    ticker = state.get("ticker", "")
    query = state.get("query", f"What is the financial outlook for {ticker}?")

    logger.info("RAGAgent | starting | ticker=%s | query=%r", ticker, query[:50])

    try:
        retriever = _get_retriever()
        pipeline = _get_ingestion_pipeline()
        store = _vector_store  # type: ignore[assignment]

        # Check if ticker has documents — ingest if not
        stats = store.get_stats()  # type: ignore[union-attr]
        if stats.get("total_chunks", 0) == 0:
            logger.info("RAGAgent | empty store — triggering ingestion | ticker=%s", ticker)
            report = await pipeline.ingest_ticker(ticker)
            logger.info(
                "RAGAgent | ingestion complete | ticker=%s | chunks=%d",
                ticker, report.chunks_stored,
            )
        else:
            # Check if this specific ticker has documents
            ticker_results = store.search(  # type: ignore[union-attr]
                query_vector=__import__("numpy").zeros(retriever.embedding_model.dimensions),
                n_results=1,
                ticker=ticker,
            )
            if not ticker_results:
                logger.info(
                    "RAGAgent | ticker not in store — ingesting | ticker=%s", ticker
                )
                report = await pipeline.ingest_ticker(ticker)
                logger.info(
                    "RAGAgent | ingestion complete | ticker=%s | chunks=%d",
                    ticker, report.chunks_stored,
                )

        # Retrieve relevant documents
        results = await retriever.retrieve(
            query=query,
            ticker=ticker,
            n_results=5,
            use_mmr=True,
        )

        if not results:
            logger.warning(
                "RAGAgent | no documents found | ticker=%s", ticker
            )
            return {
                **state,
                "retrieved_docs": [],
                "rag_context": f"No financial documents found for {ticker}.",
                "citations": [],
            }

        # Build context and citations
        rag_context = retriever.build_context(results, max_context_chars=6000)
        citations = retriever.build_citations(results)
        retrieved_docs = [r.content for r in results]

        logger.info(
            "RAGAgent | complete | ticker=%s | docs=%d | context_chars=%d",
            ticker, len(results), len(rag_context),
        )

        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "rag_context": rag_context,
            "citations": citations,
        }

    except Exception as e:
        logger.error("RAGAgent | failed | ticker=%s | %s", ticker, e, exc_info=True)
        return {
            **state,
            "retrieved_docs": [],
            "rag_context": f"Document retrieval failed: {e}",
            "citations": [],
            "error": f"RAGAgent failed: {e}",
        }
