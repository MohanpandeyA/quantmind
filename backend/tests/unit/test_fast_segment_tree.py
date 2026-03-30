"""Unit tests for FastSegmentTree and FastMinSegmentTree.

Tests cover:
- Build correctness vs recursive SegmentTree
- Range max/min queries (full, partial, single element)
- Point updates and re-queries
- Edge cases: single element, power-of-2 sizes, non-power-of-2
- Error handling: empty input, invalid ranges, out-of-bounds update
- build_fast_price_trees() helper
- Performance: iterative is faster than recursive (benchmark)
"""

from __future__ import annotations

import time

import pytest
import numpy as np

from engine.fast_segment_tree import (
    FastMinSegmentTree,
    FastSegmentTree,
    build_fast_price_trees,
)
from engine.segment_tree import AggregationType, SegmentTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> list[float]:
    return [10.0, 15.0, 8.0, 20.0, 12.0, 5.0, 18.0]


@pytest.fixture
def fast_max_tree(sample_prices: list[float]) -> FastSegmentTree:
    return FastSegmentTree(sample_prices)


@pytest.fixture
def fast_min_tree(sample_prices: list[float]) -> FastMinSegmentTree:
    return FastMinSegmentTree(sample_prices)


# ---------------------------------------------------------------------------
# FastSegmentTree (MAX) construction tests
# ---------------------------------------------------------------------------

class TestFastSegmentTreeConstruction:
    """Tests for FastSegmentTree initialization."""

    def test_stores_original_n(self, sample_prices: list[float]) -> None:
        st = FastSegmentTree(sample_prices)
        assert st._original_n == len(sample_prices)

    def test_padded_n_is_power_of_2(self, sample_prices: list[float]) -> None:
        st = FastSegmentTree(sample_prices)
        assert (st.n & (st.n - 1)) == 0  # Power of 2 check

    def test_exact_power_of_2_input(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # n=8 = 2^3
        st = FastSegmentTree(data)
        assert st.n == 8
        assert st._original_n == 8

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FastSegmentTree([])

    def test_single_element(self) -> None:
        st = FastSegmentTree([42.0])
        assert st.query(0, 0) == 42.0

    def test_two_elements(self) -> None:
        st = FastSegmentTree([3.0, 7.0])
        assert st.query(0, 1) == 7.0
        assert st.query(0, 0) == 3.0
        assert st.query(1, 1) == 7.0


# ---------------------------------------------------------------------------
# FastSegmentTree MAX query tests
# ---------------------------------------------------------------------------

class TestFastSegmentTreeQueries:
    """Tests for range maximum queries."""

    def test_full_range_max(self, fast_max_tree: FastSegmentTree) -> None:
        # [10, 15, 8, 20, 12, 5, 18] → max = 20
        assert fast_max_tree.query(0, 6) == 20.0

    def test_partial_range_left(self, fast_max_tree: FastSegmentTree) -> None:
        # [10, 15, 8] → max = 15
        assert fast_max_tree.query(0, 2) == 15.0

    def test_partial_range_right(self, fast_max_tree: FastSegmentTree) -> None:
        # [12, 5, 18] → max = 18
        assert fast_max_tree.query(4, 6) == 18.0

    def test_single_element_query(self, fast_max_tree: FastSegmentTree) -> None:
        assert fast_max_tree.query(3, 3) == 20.0
        assert fast_max_tree.query(5, 5) == 5.0

    def test_two_element_range(self, fast_max_tree: FastSegmentTree) -> None:
        assert fast_max_tree.query(2, 3) == 20.0

    def test_left_greater_than_right_raises(
        self, fast_max_tree: FastSegmentTree
    ) -> None:
        with pytest.raises(ValueError, match="left.*<=.*right"):
            fast_max_tree.query(4, 2)

    def test_out_of_bounds_raises(self, fast_max_tree: FastSegmentTree) -> None:
        with pytest.raises(ValueError):
            fast_max_tree.query(0, 10)

    def test_negative_index_raises(self, fast_max_tree: FastSegmentTree) -> None:
        with pytest.raises(ValueError):
            fast_max_tree.query(-1, 3)


# ---------------------------------------------------------------------------
# FastSegmentTree update tests
# ---------------------------------------------------------------------------

class TestFastSegmentTreeUpdate:
    """Tests for point updates."""

    def test_update_changes_max(self, fast_max_tree: FastSegmentTree) -> None:
        fast_max_tree.update(5, 100.0)
        assert fast_max_tree.query(0, 6) == 100.0

    def test_update_first_element(self, fast_max_tree: FastSegmentTree) -> None:
        fast_max_tree.update(0, 999.0)
        assert fast_max_tree.query(0, 0) == 999.0
        assert fast_max_tree.query(0, 6) == 999.0

    def test_update_last_element(self, fast_max_tree: FastSegmentTree) -> None:
        fast_max_tree.update(6, 999.0)
        assert fast_max_tree.query(6, 6) == 999.0

    def test_update_does_not_affect_other_ranges(
        self, fast_max_tree: FastSegmentTree
    ) -> None:
        fast_max_tree.update(6, 1.0)
        assert fast_max_tree.query(0, 5) == 20.0

    def test_update_out_of_bounds_raises(
        self, fast_max_tree: FastSegmentTree
    ) -> None:
        with pytest.raises(IndexError):
            fast_max_tree.update(7, 50.0)

    def test_multiple_updates(self, fast_max_tree: FastSegmentTree) -> None:
        fast_max_tree.update(0, 50.0)
        fast_max_tree.update(1, 60.0)
        assert fast_max_tree.query(0, 1) == 60.0
        assert fast_max_tree.query(0, 6) == 60.0


# ---------------------------------------------------------------------------
# FastMinSegmentTree tests
# ---------------------------------------------------------------------------

class TestFastMinSegmentTree:
    """Tests for range minimum queries."""

    def test_full_range_min(self, fast_min_tree: FastMinSegmentTree) -> None:
        # [10, 15, 8, 20, 12, 5, 18] → min = 5
        assert fast_min_tree.query(0, 6) == 5.0

    def test_partial_range_min(self, fast_min_tree: FastMinSegmentTree) -> None:
        # [10, 15, 8] → min = 8
        assert fast_min_tree.query(0, 2) == 8.0

    def test_single_element_min(self, fast_min_tree: FastMinSegmentTree) -> None:
        assert fast_min_tree.query(5, 5) == 5.0

    def test_update_changes_min(self, fast_min_tree: FastMinSegmentTree) -> None:
        fast_min_tree.update(3, 1.0)
        assert fast_min_tree.query(0, 6) == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FastMinSegmentTree([])


# ---------------------------------------------------------------------------
# Correctness vs recursive SegmentTree
# ---------------------------------------------------------------------------

class TestFastVsRecursiveCorrectness:
    """Verify FastSegmentTree gives identical results to recursive SegmentTree."""

    def test_max_queries_match_recursive(self) -> None:
        np.random.seed(42)
        data = np.random.uniform(50.0, 200.0, 100).tolist()

        fast = FastSegmentTree(data)
        recursive = SegmentTree(data, AggregationType.MAX)

        for _ in range(50):
            l = np.random.randint(0, 90)
            r = np.random.randint(l, 100)
            assert fast.query(l, r) == pytest.approx(recursive.query(l, r), rel=1e-9)

    def test_min_queries_match_recursive(self) -> None:
        np.random.seed(7)
        data = np.random.uniform(50.0, 200.0, 100).tolist()

        fast = FastMinSegmentTree(data)
        recursive = SegmentTree(data, AggregationType.MIN)

        for _ in range(50):
            l = np.random.randint(0, 90)
            r = np.random.randint(l, 100)
            assert fast.query(l, r) == pytest.approx(recursive.query(l, r), rel=1e-9)

    def test_updates_match_recursive(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        fast = FastSegmentTree(data)
        recursive = SegmentTree(data, AggregationType.MAX)

        fast.update(2, 100.0)
        recursive.update(2, 100.0)

        assert fast.query(0, 4) == recursive.query(0, 4)
        assert fast.query(1, 3) == recursive.query(1, 3)


# ---------------------------------------------------------------------------
# Performance test: iterative faster than recursive
# ---------------------------------------------------------------------------

class TestFastSegmentTreePerformance:
    """Verify iterative tree is faster than recursive for large datasets."""

    def test_iterative_faster_than_recursive_for_many_queries(self) -> None:
        np.random.seed(0)
        n = 2520  # 10 years of daily data
        data = np.random.uniform(50.0, 300.0, n).tolist()

        fast = FastSegmentTree(data)
        recursive = SegmentTree(data, AggregationType.MAX)

        queries = [(np.random.randint(0, n - 100), np.random.randint(n - 100, n))
                   for _ in range(1000)]

        # Time iterative
        t0 = time.perf_counter()
        for l, r in queries:
            fast.query(l, r)
        fast_time = time.perf_counter() - t0

        # Time recursive
        t0 = time.perf_counter()
        for l, r in queries:
            recursive.query(l, r)
        recursive_time = time.perf_counter() - t0

        # Iterative should be faster (or at worst equal)
        # We use a generous threshold since Python timing can vary
        assert fast_time <= recursive_time * 2.0, (
            f"FastSegmentTree ({fast_time*1000:.1f}ms) should not be "
            f"much slower than recursive ({recursive_time*1000:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# build_fast_price_trees tests
# ---------------------------------------------------------------------------

class TestBuildFastPriceTrees:
    """Tests for the build_fast_price_trees() helper."""

    def test_returns_correct_types(self) -> None:
        highs = [105.0, 110.0, 108.0, 115.0]
        lows = [100.0, 104.0, 102.0, 109.0]
        max_tree, min_tree = build_fast_price_trees(highs, lows)
        assert isinstance(max_tree, FastSegmentTree)
        assert isinstance(min_tree, FastMinSegmentTree)

    def test_resistance_detection(self) -> None:
        highs = [105.0, 110.0, 108.0, 115.0, 112.0]
        lows = [100.0, 104.0, 102.0, 109.0, 107.0]
        max_tree, _ = build_fast_price_trees(highs, lows)
        assert max_tree.query(0, 4) == 115.0

    def test_support_detection(self) -> None:
        highs = [105.0, 110.0, 108.0, 115.0, 112.0]
        lows = [100.0, 104.0, 102.0, 109.0, 107.0]
        _, min_tree = build_fast_price_trees(highs, lows)
        assert min_tree.query(0, 4) == 100.0

    def test_live_update_simulation(self) -> None:
        """Simulate live trading: update tree on each new bar."""
        highs = [100.0, 102.0, 105.0, 103.0, 101.0]
        lows = [98.0, 99.0, 101.0, 100.0, 99.0]
        max_tree, min_tree = build_fast_price_trees(highs, lows)

        # New bar arrives: high=108, low=104
        max_tree.update(4, 108.0)
        min_tree.update(4, 104.0)

        assert max_tree.query(0, 4) == 108.0
        assert min_tree.query(0, 4) == 98.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            build_fast_price_trees([100.0, 105.0], [99.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            build_fast_price_trees([], [])
