"""Unit tests for sliding window DSA utilities.

Tests cover:
- rolling_mean: correctness, NaN padding, edge cases
- rolling_std: correctness, ddof handling
- rolling_sharpe: annualization, zero-vol handling
- exponential_moving_average: alpha computation, convergence
- rolling_max_drawdown: peak-to-trough correctness
- compute_returns: simple returns
- compute_log_returns: log returns, non-positive price guard
- Input validation: empty, window > n, window < 1
"""

import pytest
import numpy as np

from engine.sliding_window import (
    compute_log_returns,
    compute_returns,
    exponential_moving_average,
    rolling_max_drawdown,
    rolling_mean,
    rolling_sharpe,
    rolling_std,
)


# ---------------------------------------------------------------------------
# rolling_mean tests
# ---------------------------------------------------------------------------

class TestRollingMean:
    """Tests for Simple Moving Average computation."""

    def test_basic_correctness(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_mean(data, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_window_1_returns_original(self) -> None:
        data = [10.0, 20.0, 30.0]
        result = rolling_mean(data, window=1)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])

    def test_window_equals_length(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        result = rolling_mean(data, window=4)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert result[3] == pytest.approx(2.5)

    def test_nan_count_equals_window_minus_1(self) -> None:
        data = list(range(1, 11))  # 10 elements
        result = rolling_mean(data, window=5)
        nan_count = int(np.sum(np.isnan(result)))
        assert nan_count == 4  # window - 1

    def test_numpy_array_input(self) -> None:
        data = np.array([2.0, 4.0, 6.0, 8.0])
        result = rolling_mean(data, window=2)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(5.0)
        assert result[3] == pytest.approx(7.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            rolling_mean([], window=3)

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_mean([1.0, 2.0, 3.0], window=0)

    def test_window_exceeds_length_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            rolling_mean([1.0, 2.0], window=5)

    def test_constant_series(self) -> None:
        data = [5.0] * 10
        result = rolling_mean(data, window=3)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, [5.0] * 8)


# ---------------------------------------------------------------------------
# rolling_std tests
# ---------------------------------------------------------------------------

class TestRollingStd:
    """Tests for rolling standard deviation."""

    def test_basic_correctness(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_std(data, window=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(1.0)

    def test_constant_series_std_is_zero(self) -> None:
        data = [7.0] * 10
        result = rolling_std(data, window=3)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, [0.0] * 8)

    def test_ddof_0_vs_1(self) -> None:
        data = [1.0, 3.0, 5.0]
        result_ddof1 = rolling_std(data, window=3, ddof=1)
        result_ddof0 = rolling_std(data, window=3, ddof=0)
        # ddof=1 gives sample std, ddof=0 gives population std
        assert result_ddof1[2] > result_ddof0[2]

    def test_window_1_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            rolling_std([1.0, 2.0, 3.0], window=1)

    def test_nan_count(self) -> None:
        data = list(range(1, 8))
        result = rolling_std(data, window=4)
        nan_count = int(np.sum(np.isnan(result)))
        assert nan_count == 3  # window - 1


# ---------------------------------------------------------------------------
# rolling_sharpe tests
# ---------------------------------------------------------------------------

class TestRollingSharpe:
    """Tests for rolling Sharpe ratio."""

    def test_positive_returns_positive_sharpe(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 200)
        result = rolling_sharpe(returns, window=60)
        valid = result[~np.isnan(result)]
        # With positive mean return, Sharpe should be positive on average
        assert np.mean(valid) > 0

    def test_zero_volatility_returns_nan(self) -> None:
        # Constant returns → zero std → Sharpe should be NaN (not inf)
        returns = np.full(100, 0.001)
        result = rolling_sharpe(returns, window=30)
        valid = result[~np.isnan(result)]
        # All valid values should be NaN due to zero std
        assert len(valid) == 0

    def test_annualization_scales_result(self) -> None:
        np.random.seed(0)
        returns = np.random.normal(0.001, 0.01, 200)
        ann = rolling_sharpe(returns, window=60, annualize=True)
        non_ann = rolling_sharpe(returns, window=60, annualize=False)
        # Annualized should be sqrt(252) ≈ 15.87x larger
        valid_ann = ann[~np.isnan(ann)]
        valid_non = non_ann[~np.isnan(non_ann)]
        ratio = valid_ann / valid_non
        np.testing.assert_array_almost_equal(ratio, np.full(len(ratio), np.sqrt(252)), decimal=5)

    def test_nan_count_equals_window_minus_1(self) -> None:
        returns = np.random.normal(0.001, 0.01, 100)
        result = rolling_sharpe(returns, window=30)
        nan_count = int(np.sum(np.isnan(result)))
        assert nan_count == 29


# ---------------------------------------------------------------------------
# exponential_moving_average tests
# ---------------------------------------------------------------------------

class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_first_value_equals_input(self) -> None:
        data = [100.0, 105.0, 102.0]
        result = exponential_moving_average(data, span=3)
        assert result[0] == pytest.approx(100.0)

    def test_ema_converges_toward_constant(self) -> None:
        # If all values are the same, EMA should equal that value
        data = [50.0] * 20
        result = exponential_moving_average(data, span=5)
        np.testing.assert_array_almost_equal(result, [50.0] * 20)

    def test_ema_responds_to_price_increase(self) -> None:
        # Rising prices → EMA should be below current price (lagging)
        data = list(range(1, 21))  # 1, 2, ..., 20
        result = exponential_moving_average(data, span=5)
        # EMA lags, so last EMA < last price
        assert result[-1] < data[-1]

    def test_span_1_equals_input(self) -> None:
        # span=1 → alpha=1 → EMA = current price
        data = [10.0, 20.0, 30.0]
        result = exponential_moving_average(data, span=1)
        np.testing.assert_array_almost_equal(result, data)

    def test_output_length_equals_input(self) -> None:
        data = [1.0] * 50
        result = exponential_moving_average(data, span=10)
        assert len(result) == 50

    def test_span_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="span must be >= 1"):
            exponential_moving_average([1.0, 2.0], span=0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            exponential_moving_average([], span=5)


# ---------------------------------------------------------------------------
# rolling_max_drawdown tests
# ---------------------------------------------------------------------------

class TestRollingMaxDrawdown:
    """Tests for rolling maximum drawdown."""

    def test_no_drawdown_returns_zero(self) -> None:
        # Monotonically increasing equity → no drawdown
        equity = [100.0, 110.0, 120.0, 130.0, 140.0]
        result = rolling_max_drawdown(equity, window=3)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, [0.0] * len(valid))

    def test_known_drawdown(self) -> None:
        # Peak=120, trough=90 → drawdown = (120-90)/120 = 0.25
        equity = [100.0, 120.0, 90.0, 95.0, 100.0]
        result = rolling_max_drawdown(equity, window=5)
        assert result[4] == pytest.approx(0.25, rel=1e-3)

    def test_nan_count(self) -> None:
        equity = [100.0] * 10
        result = rolling_max_drawdown(equity, window=4)
        nan_count = int(np.sum(np.isnan(result)))
        assert nan_count == 3  # window - 1

    def test_values_between_0_and_1(self) -> None:
        np.random.seed(7)
        equity = np.cumprod(1 + np.random.normal(0.001, 0.02, 200)) * 100
        result = rolling_max_drawdown(equity.tolist(), window=30)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)


# ---------------------------------------------------------------------------
# compute_returns tests
# ---------------------------------------------------------------------------

class TestComputeReturns:
    """Tests for simple period returns."""

    def test_basic_correctness(self) -> None:
        prices = [100.0, 105.0, 102.0, 108.0]
        returns = compute_returns(prices)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.05)
        assert returns[1] == pytest.approx(-0.02857, rel=1e-3)
        assert returns[2] == pytest.approx(0.05882, rel=1e-3)

    def test_flat_prices_zero_returns(self) -> None:
        prices = [50.0] * 5
        returns = compute_returns(prices)
        np.testing.assert_array_almost_equal(returns, [0.0] * 4)

    def test_output_length(self) -> None:
        prices = list(range(1, 11))
        returns = compute_returns(prices)
        assert len(returns) == 9

    def test_single_price_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_returns([100.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_returns([])


# ---------------------------------------------------------------------------
# compute_log_returns tests
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    """Tests for log returns."""

    def test_basic_correctness(self) -> None:
        prices = [100.0, 105.0]
        log_ret = compute_log_returns(prices)
        expected = np.log(105.0 / 100.0)
        assert log_ret[0] == pytest.approx(expected)

    def test_output_length(self) -> None:
        prices = [10.0, 11.0, 12.0, 13.0]
        log_ret = compute_log_returns(prices)
        assert len(log_ret) == 3

    def test_non_positive_price_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_log_returns([100.0, 0.0, 105.0])

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_log_returns([100.0, -5.0])

    def test_single_price_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compute_log_returns([100.0])

    def test_log_returns_sum_to_total_log_return(self) -> None:
        prices = [100.0, 110.0, 121.0, 133.1]
        log_rets = compute_log_returns(prices)
        total = np.sum(log_rets)
        expected = np.log(133.1 / 100.0)
        assert total == pytest.approx(expected, rel=1e-6)
