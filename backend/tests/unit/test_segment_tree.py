"""Unit tests for the SegmentTree DSA implementation.

Tests cover:
- Build correctness for MAX, MIN, SUM trees
- Range query correctness (full range, partial, single element)
- Point update and re-query
- Edge cases: single element, two elements, all same values
- Error handling: empty input, invalid ranges, out-of-bounds update
- build_price_trees() helper function
"""

import pytest
import numpy as np

from engine.segment_tree import AggregationType, SegmentTree, build_price_trees


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> list[float]:
    """Standard price list for testing."""
    return [10.0, 15.0, 8.0, 20.0, 12.0, 5.0, 18.0]


@pytest.fixture
def max_tree(sample_prices: list[float]) -> SegmentTree:
    return SegmentTree(sample_prices, AggregationType.MAX)


@pytest.fixture
def min_tree(sample_prices: list[float]) -> SegmentTree:
    return SegmentTree(sample_prices, AggregationType.MIN)


@pytest.fixture
def sum_tree(sample_prices: list[float]) -> SegmentTree:
    return SegmentTree(sample_prices, AggregationType.SUM)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestSegmentTreeConstruction:
    """Tests for SegmentTree initialization and build."""

    def test_build_max_tree_stores_correct_size(self, sample_prices: list[float]) -> None:
        st = SegmentTree(sample_prices, AggregationType.MAX)
        assert st.n == len(sample_prices)

    def test_build_min_tree_stores_correct_size(self, sample_prices: list[float]) -> None:
        st = SegmentTree(sample_prices, AggregationType.MIN)
        assert st.n == len(sample_prices)

    def test_build_sum_tree_stores_correct_size(self, sample_prices: list[float]) -> None:
        st = SegmentTree(sample_prices, AggregationType.SUM)
        assert st.n == len(sample_prices)

    def test_empty_data_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SegmentTree([], AggregationType.MAX)

    def test_single_element_tree(self) -> None:
        st = SegmentTree([42.0], AggregationType.MAX)
        assert st.n == 1
        assert st.query(0, 0) == 42.0

    def test_two_element_tree(self) -> None:
        st = SegmentTree([3.0, 7.0], AggregationType.MAX)
        assert st.query(0, 1) == 7.0
        assert st.query(0, 0) == 3.0
        assert st.query(1, 1) == 7.0

    def test_agg_type_stored_correctly(self, sample_prices: list[float]) -> None:
        st = SegmentTree(sample_prices, AggregationType.MIN)
        assert st.agg_type == AggregationType.MIN


# ---------------------------------------------------------------------------
# MAX tree query tests
# ---------------------------------------------------------------------------

class TestMaxTreeQueries:
    """Tests for range maximum queries."""

    def test_full_range_max(self, max_tree: SegmentTree) -> None:
        # [10, 15, 8, 20, 12, 5, 18] → max = 20
        assert max_tree.query(0, 6) == 20.0

    def test_partial_range_max_left(self, max_tree: SegmentTree) -> None:
        # [10, 15, 8] → max = 15
        assert max_tree.query(0, 2) == 15.0

    def test_partial_range_max_right(self, max_tree: SegmentTree) -> None:
        # [12, 5, 18] → max = 18
        assert max_tree.query(4, 6) == 18.0

    def test_single_element_query(self, max_tree: SegmentTree) -> None:
        assert max_tree.query(3, 3) == 20.0
        assert max_tree.query(5, 5) == 5.0

    def test_two_element_range(self, max_tree: SegmentTree) -> None:
        # [8, 20] → max = 20
        assert max_tree.query(2, 3) == 20.0

    def test_range_max_convenience_method(self, max_tree: SegmentTree) -> None:
        assert max_tree.range_max(0, 6) == 20.0

    def test_range_max_wrong_agg_type_raises(self, min_tree: SegmentTree) -> None:
        with pytest.raises(ValueError, match="AggregationType.MAX"):
            min_tree.range_max(0, 6)


# ---------------------------------------------------------------------------
# MIN tree query tests
# ---------------------------------------------------------------------------

class TestMinTreeQueries:
    """Tests for range minimum queries."""

    def test_full_range_min(self, min_tree: SegmentTree) -> None:
        # [10, 15, 8, 20, 12, 5, 18] → min = 5
        assert min_tree.query(0, 6) == 5.0

    def test_partial_range_min_left(self, min_tree: SegmentTree) -> None:
        # [10, 15, 8] → min = 8
        assert min_tree.query(0, 2) == 8.0

    def test_partial_range_min_right(self, min_tree: SegmentTree) -> None:
        # [12, 5, 18] → min = 5
        assert min_tree.query(4, 6) == 5.0

    def test_single_element_min(self, min_tree: SegmentTree) -> None:
        assert min_tree.query(0, 0) == 10.0
        assert min_tree.query(5, 5) == 5.0

    def test_range_min_convenience_method(self, min_tree: SegmentTree) -> None:
        assert min_tree.range_min(0, 6) == 5.0

    def test_range_min_wrong_agg_type_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(ValueError, match="AggregationType.MIN"):
            max_tree.range_min(0, 6)


# ---------------------------------------------------------------------------
# SUM tree query tests
# ---------------------------------------------------------------------------

class TestSumTreeQueries:
    """Tests for range sum queries."""

    def test_full_range_sum(self, sum_tree: SegmentTree) -> None:
        # 10+15+8+20+12+5+18 = 88
        assert sum_tree.query(0, 6) == pytest.approx(88.0)

    def test_partial_range_sum(self, sum_tree: SegmentTree) -> None:
        # 10+15+8 = 33
        assert sum_tree.query(0, 2) == pytest.approx(33.0)

    def test_single_element_sum(self, sum_tree: SegmentTree) -> None:
        assert sum_tree.query(3, 3) == pytest.approx(20.0)

    def test_two_element_sum(self, sum_tree: SegmentTree) -> None:
        # 5+18 = 23
        assert sum_tree.query(5, 6) == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------

class TestSegmentTreeUpdate:
    """Tests for point updates."""

    def test_update_changes_max(self, max_tree: SegmentTree) -> None:
        # Update index 5 (value 5.0) to 100.0 → new max = 100
        max_tree.update(5, 100.0)
        assert max_tree.query(0, 6) == 100.0

    def test_update_changes_min(self, min_tree: SegmentTree) -> None:
        # Update index 3 (value 20.0) to 1.0 → new min = 1
        min_tree.update(3, 1.0)
        assert min_tree.query(0, 6) == 1.0

    def test_update_changes_sum(self, sum_tree: SegmentTree) -> None:
        # Original sum = 88. Update index 0 (10.0) to 20.0 → new sum = 98
        sum_tree.update(0, 20.0)
        assert sum_tree.query(0, 6) == pytest.approx(98.0)

    def test_update_first_element(self, max_tree: SegmentTree) -> None:
        max_tree.update(0, 999.0)
        assert max_tree.query(0, 0) == 999.0
        assert max_tree.query(0, 6) == 999.0

    def test_update_last_element(self, max_tree: SegmentTree) -> None:
        max_tree.update(6, 999.0)
        assert max_tree.query(6, 6) == 999.0

    def test_update_does_not_affect_other_ranges(self, max_tree: SegmentTree) -> None:
        # Update index 6 (18.0) to 1.0 — should not affect [0,5]
        max_tree.update(6, 1.0)
        assert max_tree.query(0, 5) == 20.0  # max of [10,15,8,20,12,5]

    def test_update_out_of_bounds_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(IndexError):
            max_tree.update(7, 50.0)

    def test_update_negative_index_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(IndexError):
            max_tree.update(-1, 50.0)


# ---------------------------------------------------------------------------
# Range validation tests
# ---------------------------------------------------------------------------

class TestRangeValidation:
    """Tests for invalid range handling."""

    def test_left_greater_than_right_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(ValueError, match="left.*<=.*right"):
            max_tree.query(4, 2)

    def test_negative_left_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(ValueError):
            max_tree.query(-1, 3)

    def test_right_out_of_bounds_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(ValueError):
            max_tree.query(0, 10)

    def test_both_out_of_bounds_raises(self, max_tree: SegmentTree) -> None:
        with pytest.raises(ValueError):
            max_tree.query(10, 20)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_all_same_values_max(self) -> None:
        data = [5.0] * 10
        st = SegmentTree(data, AggregationType.MAX)
        assert st.query(0, 9) == 5.0
        assert st.query(3, 7) == 5.0

    def test_all_same_values_min(self) -> None:
        data = [5.0] * 10
        st = SegmentTree(data, AggregationType.MIN)
        assert st.query(0, 9) == 5.0

    def test_negative_values_max(self) -> None:
        data = [-5.0, -3.0, -10.0, -1.0, -8.0]
        st = SegmentTree(data, AggregationType.MAX)
        assert st.query(0, 4) == -1.0

    def test_negative_values_min(self) -> None:
        data = [-5.0, -3.0, -10.0, -1.0, -8.0]
        st = SegmentTree(data, AggregationType.MIN)
        assert st.query(0, 4) == -10.0

    def test_large_dataset_correctness(self) -> None:
        """Verify O(log n) tree gives same result as brute force on large data."""
        np.random.seed(42)
        data = np.random.uniform(50.0, 200.0, 1000).tolist()
        st = SegmentTree(data, AggregationType.MAX)

        # Random range queries
        for _ in range(50):
            l = np.random.randint(0, 900)
            r = np.random.randint(l, 1000)
            expected = max(data[l : r + 1])
            assert st.query(l, r) == pytest.approx(expected, rel=1e-9)

    def test_floating_point_precision(self) -> None:
        data = [1.0000001, 1.0000002, 1.0000003]
        st = SegmentTree(data, AggregationType.MAX)
        assert st.query(0, 2) == pytest.approx(1.0000003)


# ---------------------------------------------------------------------------
# build_price_trees helper tests
# ---------------------------------------------------------------------------

class TestBuildPriceTrees:
    """Tests for the build_price_trees() convenience function."""

    def test_returns_two_trees(self) -> None:
        highs = [105.0, 110.0, 108.0, 115.0]
        lows = [100.0, 104.0, 102.0, 109.0]
        max_tree, min_tree = build_price_trees(highs, lows)
        assert isinstance(max_tree, SegmentTree)
        assert isinstance(min_tree, SegmentTree)

    def test_max_tree_queries_highs(self) -> None:
        highs = [105.0, 110.0, 108.0, 115.0]
        lows = [100.0, 104.0, 102.0, 109.0]
        max_tree, _ = build_price_trees(highs, lows)
        assert max_tree.query(0, 3) == 115.0

    def test_min_tree_queries_lows(self) -> None:
        highs = [105.0, 110.0, 108.0, 115.0]
        lows = [100.0, 104.0, 102.0, 109.0]
        _, min_tree = build_price_trees(highs, lows)
        assert min_tree.query(0, 3) == 100.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            build_price_trees([100.0, 105.0], [99.0])

    def test_empty_inputs_raises(self) -> None:
        with pytest.raises(ValueError):
            build_price_trees([], [])

    def test_resistance_level_detection(self) -> None:
        """Simulate finding resistance in a 5-day window."""
        highs = [100.0, 102.0, 105.0, 103.0, 101.0, 108.0, 106.0]
        lows  = [98.0,  99.0,  101.0, 100.0, 99.0,  104.0, 103.0]
        max_tree, min_tree = build_price_trees(highs, lows)

        # Resistance in last 5 days (indices 2-6)
        assert max_tree.query(2, 6) == 108.0
        # Support in first 3 days (indices 0-2)
        assert min_tree.query(0, 2) == 98.0
