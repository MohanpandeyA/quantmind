"""Segment Tree implementation for O(log n) range queries on price data.

Used in the backtesting engine to efficiently compute:
- Range maximum (highest high in a date window)
- Range minimum (lowest low in a date window)
- Support and resistance level detection

Time Complexity:
    - Build:  O(n)
    - Query:  O(log n)
    - Update: O(log n)

Space Complexity: O(n)
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


class AggregationType(str, Enum):
    """Supported aggregation types for the segment tree."""

    MAX = "max"
    MIN = "min"
    SUM = "sum"


class SegmentTree:
    """Generic segment tree supporting range max, min, and sum queries.

    This data structure enables O(log n) range queries on a static or
    dynamically updated array — critical for fast support/resistance
    detection across large OHLCV price datasets.

    Attributes:
        n: Number of elements in the original array.
        agg_type: Aggregation type (MAX, MIN, or SUM).
        tree: Internal tree array of size 4*n.

    Example:
        >>> prices = [10.0, 15.0, 8.0, 20.0, 12.0]
        >>> st = SegmentTree(prices, AggregationType.MAX)
        >>> st.query(1, 3)  # max in index range [1, 3]
        20.0
        >>> st.update(2, 25.0)  # update index 2 to 25.0
        >>> st.query(0, 4)  # new max across all
        25.0
    """

    def __init__(
        self,
        data: List[float],
        agg_type: AggregationType = AggregationType.MAX,
    ) -> None:
        """Initialize and build the segment tree from input data.

        Args:
            data: List of float values (e.g., closing prices).
            agg_type: Aggregation type — MAX, MIN, or SUM.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("SegmentTree requires a non-empty data list.")

        self.n: int = len(data)
        self.agg_type: AggregationType = agg_type
        self._agg_fn: Callable[[float, float], float] = self._get_agg_fn(agg_type)
        self._identity: float = self._get_identity(agg_type)

        # Tree array: 1-indexed, size 4*n is safe upper bound
        self.tree: List[float] = [self._identity] * (4 * self.n)

        self._build(data, node=1, start=0, end=self.n - 1)
        logger.debug(
            "SegmentTree built | n=%d | agg=%s", self.n, agg_type.value
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, left: int, right: int) -> float:
        """Return the aggregated value over the range [left, right] (inclusive).

        Args:
            left: Left index (0-based, inclusive).
            right: Right index (0-based, inclusive).

        Returns:
            Aggregated float value (max / min / sum) over the range.

        Raises:
            ValueError: If indices are out of bounds or left > right.

        Example:
            >>> st = SegmentTree([3.0, 1.0, 4.0, 1.0, 5.0], AggregationType.MIN)
            >>> st.query(0, 4)
            1.0
        """
        self._validate_range(left, right)
        result = self._query(node=1, start=0, end=self.n - 1, left=left, right=right)
        logger.debug("query([%d, %d]) = %.4f", left, right, result)
        return result

    def update(self, index: int, value: float) -> None:
        """Update the value at a given index and propagate changes up the tree.

        Args:
            index: 0-based index to update.
            value: New float value.

        Raises:
            IndexError: If index is out of bounds.

        Example:
            >>> st = SegmentTree([1.0, 2.0, 3.0], AggregationType.SUM)
            >>> st.update(1, 10.0)
            >>> st.query(0, 2)
            14.0
        """
        if not (0 <= index < self.n):
            raise IndexError(
                f"Index {index} out of bounds for tree of size {self.n}."
            )
        self._update(node=1, start=0, end=self.n - 1, index=index, value=value)
        logger.debug("update(index=%d, value=%.4f)", index, value)

    def range_max(self, left: int, right: int) -> float:
        """Convenience method: range maximum query.

        Args:
            left: Left index (0-based, inclusive).
            right: Right index (0-based, inclusive).

        Returns:
            Maximum value in the range.

        Raises:
            ValueError: If tree was not built with AggregationType.MAX.
        """
        if self.agg_type != AggregationType.MAX:
            raise ValueError(
                f"range_max() requires AggregationType.MAX, got {self.agg_type}."
            )
        return self.query(left, right)

    def range_min(self, left: int, right: int) -> float:
        """Convenience method: range minimum query.

        Args:
            left: Left index (0-based, inclusive).
            right: Right index (0-based, inclusive).

        Returns:
            Minimum value in the range.

        Raises:
            ValueError: If tree was not built with AggregationType.MIN.
        """
        if self.agg_type != AggregationType.MIN:
            raise ValueError(
                f"range_min() requires AggregationType.MIN, got {self.agg_type}."
            )
        return self.query(left, right)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build(
        self, data: List[float], node: int, start: int, end: int
    ) -> None:
        """Recursively build the segment tree.

        Args:
            data: Original data array.
            node: Current tree node index (1-based).
            start: Left boundary of current segment.
            end: Right boundary of current segment.
        """
        if start == end:
            self.tree[node] = data[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        self._build(data, left_child, start, mid)
        self._build(data, right_child, mid + 1, end)
        self.tree[node] = self._agg_fn(self.tree[left_child], self.tree[right_child])

    def _query(
        self, node: int, start: int, end: int, left: int, right: int
    ) -> float:
        """Recursively query the segment tree.

        Args:
            node: Current tree node index (1-based).
            start: Left boundary of current segment.
            end: Right boundary of current segment.
            left: Query left boundary.
            right: Query right boundary.

        Returns:
            Aggregated value for the queried range.
        """
        # No overlap
        if right < start or end < left:
            return self._identity

        # Total overlap
        if left <= start and end <= right:
            return self.tree[node]

        # Partial overlap
        mid = (start + end) // 2
        left_val = self._query(2 * node, start, mid, left, right)
        right_val = self._query(2 * node + 1, mid + 1, end, left, right)
        return self._agg_fn(left_val, right_val)

    def _update(
        self, node: int, start: int, end: int, index: int, value: float
    ) -> None:
        """Recursively update a value and propagate changes.

        Args:
            node: Current tree node index (1-based).
            start: Left boundary of current segment.
            end: Right boundary of current segment.
            index: Target index to update.
            value: New value.
        """
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)

        self.tree[node] = self._agg_fn(
            self.tree[2 * node], self.tree[2 * node + 1]
        )

    def _validate_range(self, left: int, right: int) -> None:
        """Validate query range indices.

        Args:
            left: Left index.
            right: Right index.

        Raises:
            ValueError: If range is invalid.
        """
        if left > right:
            raise ValueError(
                f"left ({left}) must be <= right ({right})."
            )
        if left < 0 or right >= self.n:
            raise ValueError(
                f"Range [{left}, {right}] out of bounds for size {self.n}."
            )

    @staticmethod
    def _get_agg_fn(agg_type: AggregationType) -> Callable[[float, float], float]:
        """Return the aggregation function for the given type.

        Args:
            agg_type: Aggregation type enum.

        Returns:
            Binary aggregation function.
        """
        mapping: dict[AggregationType, Callable[[float, float], float]] = {
            AggregationType.MAX: max,
            AggregationType.MIN: min,
            AggregationType.SUM: lambda a, b: a + b,
        }
        return mapping[agg_type]

    @staticmethod
    def _get_identity(agg_type: AggregationType) -> float:
        """Return the identity element for the aggregation type.

        The identity element is returned when a query range has no overlap
        with the current segment (base case for recursion).

        Args:
            agg_type: Aggregation type enum.

        Returns:
            Identity float value.
        """
        mapping: dict[AggregationType, float] = {
            AggregationType.MAX: float("-inf"),
            AggregationType.MIN: float("inf"),
            AggregationType.SUM: 0.0,
        }
        return mapping[agg_type]


def build_price_trees(
    highs: List[float], lows: List[float]
) -> tuple[SegmentTree, SegmentTree]:
    """Build a MAX tree for highs and MIN tree for lows from OHLCV data.

    This is the primary entry point for the backtesting engine to create
    support/resistance detection structures.

    Args:
        highs: List of daily high prices.
        lows: List of daily low prices.

    Returns:
        Tuple of (max_tree for highs, min_tree for lows).

    Raises:
        ValueError: If highs and lows have different lengths or are empty.

    Example:
        >>> highs = [105.0, 110.0, 108.0, 115.0]
        >>> lows  = [100.0, 104.0, 102.0, 109.0]
        >>> max_tree, min_tree = build_price_trees(highs, lows)
        >>> max_tree.query(0, 3)  # highest high
        115.0
        >>> min_tree.query(0, 3)  # lowest low
        100.0
    """
    if len(highs) != len(lows):
        raise ValueError(
            f"highs and lows must have equal length: {len(highs)} != {len(lows)}"
        )
    if not highs:
        raise ValueError("highs and lows must not be empty.")

    max_tree = SegmentTree(highs, AggregationType.MAX)
    min_tree = SegmentTree(lows, AggregationType.MIN)
    logger.info(
        "Price trees built | n=%d | max_tree + min_tree ready", len(highs)
    )
    return max_tree, min_tree
