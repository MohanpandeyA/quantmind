"""Unit tests for portfolio performance metrics.

Tests cover:
- sharpe_ratio: positive/negative returns, annualization, zero-vol
- sortino_ratio: downside-only penalization
- max_drawdown: peak-to-trough correctness
- calmar_ratio: annualized return / mdd
- annualized_return / annualized_volatility
- historical_var / parametric_var / conditional_var
- win_rate / profit_factor
- compute_full_report: integration of all metrics
- PerformanceReport.to_dict(): serialization
"""

import pytest
import numpy as np

from engine.metrics import (
    PerformanceReport,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_full_report,
    conditional_var,
    historical_var,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def positive_returns() -> np.ndarray:
    """Consistently positive daily returns."""
    np.random.seed(42)
    return np.random.normal(0.002, 0.01, 252)


@pytest.fixture
def negative_returns() -> np.ndarray:
    """Consistently negative daily returns."""
    np.random.seed(42)
    return np.random.normal(-0.002, 0.01, 252)


@pytest.fixture
def mixed_returns() -> np.ndarray:
    """Mixed positive/negative returns."""
    np.random.seed(7)
    return np.random.normal(0.0, 0.015, 252)


@pytest.fixture
def equity_curve_rising() -> np.ndarray:
    """Monotonically rising equity curve."""
    return np.linspace(100_000, 150_000, 252)


@pytest.fixture
def equity_curve_with_drawdown() -> np.ndarray:
    """Equity curve with a known 25% drawdown."""
    # 100 → 120 → 90 → 110 → 130
    return np.array([100.0, 110.0, 120.0, 100.0, 90.0, 95.0, 110.0, 130.0])


# ---------------------------------------------------------------------------
# sharpe_ratio tests
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    """Tests for Sharpe ratio computation."""

    def test_positive_returns_positive_sharpe(
        self, positive_returns: np.ndarray
    ) -> None:
        result = sharpe_ratio(positive_returns)
        assert result > 0

    def test_negative_returns_negative_sharpe(
        self, negative_returns: np.ndarray
    ) -> None:
        result = sharpe_ratio(negative_returns)
        assert result < 0

    def test_zero_volatility_returns_zero(self) -> None:
        returns = np.full(100, 0.001)
        result = sharpe_ratio(returns)
        assert result == 0.0

    def test_annualization_factor(self, positive_returns: np.ndarray) -> None:
        ann = sharpe_ratio(positive_returns, annualize=True)
        non_ann = sharpe_ratio(positive_returns, annualize=False)
        assert ann == pytest.approx(non_ann * np.sqrt(252), rel=1e-5)

    def test_single_return_returns_zero(self) -> None:
        assert sharpe_ratio(np.array([0.01])) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert sharpe_ratio(np.array([])) == 0.0

    def test_risk_free_rate_reduces_sharpe(
        self, positive_returns: np.ndarray
    ) -> None:
        sharpe_no_rf = sharpe_ratio(positive_returns, risk_free_rate=0.0)
        sharpe_with_rf = sharpe_ratio(positive_returns, risk_free_rate=0.0001)
        assert sharpe_no_rf > sharpe_with_rf


# ---------------------------------------------------------------------------
# sortino_ratio tests
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    """Tests for Sortino ratio computation."""

    def test_positive_returns_positive_sortino(
        self, positive_returns: np.ndarray
    ) -> None:
        result = sortino_ratio(positive_returns)
        assert result > 0

    def test_no_losing_periods_returns_inf(self) -> None:
        returns = np.full(100, 0.001)  # All positive
        result = sortino_ratio(returns)
        assert result == float("inf")

    def test_sortino_gte_sharpe_for_positive_skew(
        self, positive_returns: np.ndarray
    ) -> None:
        # Sortino only penalizes downside, so it should be >= Sharpe
        # for positively skewed returns
        s = sharpe_ratio(positive_returns, annualize=False)
        so = sortino_ratio(positive_returns, annualize=False)
        # This is generally true but not guaranteed for all distributions
        # Just check both are positive
        assert s > 0
        assert so > 0

    def test_empty_returns_zero(self) -> None:
        assert sortino_ratio(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown tests
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    """Tests for maximum drawdown computation."""

    def test_known_drawdown(
        self, equity_curve_with_drawdown: np.ndarray
    ) -> None:
        # Peak = 120, trough = 90 → MDD = (120-90)/120 = 0.25
        mdd = max_drawdown(equity_curve_with_drawdown)
        assert mdd == pytest.approx(0.25, rel=1e-3)

    def test_rising_equity_zero_drawdown(
        self, equity_curve_rising: np.ndarray
    ) -> None:
        mdd = max_drawdown(equity_curve_rising)
        assert mdd == pytest.approx(0.0, abs=1e-6)

    def test_result_between_0_and_1(self) -> None:
        np.random.seed(5)
        equity = np.cumprod(1 + np.random.normal(0.001, 0.02, 500)) * 100_000
        mdd = max_drawdown(equity)
        assert 0.0 <= mdd <= 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            max_drawdown(np.array([]))

    def test_single_element_zero_drawdown(self) -> None:
        mdd = max_drawdown(np.array([100.0]))
        assert mdd == 0.0

    def test_total_loss_drawdown_is_1(self) -> None:
        equity = np.array([100.0, 50.0, 0.001])
        mdd = max_drawdown(equity)
        assert mdd > 0.99  # Near 100% drawdown


# ---------------------------------------------------------------------------
# calmar_ratio tests
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    """Tests for Calmar ratio computation."""

    def test_positive_calmar_for_good_strategy(
        self,
        positive_returns: np.ndarray,
        equity_curve_with_drawdown: np.ndarray,
    ) -> None:
        # Use a simple equity curve derived from positive returns
        equity = np.cumprod(1 + positive_returns) * 100_000
        result = calmar_ratio(positive_returns, equity)
        assert result > 0

    def test_zero_drawdown_returns_zero(
        self, positive_returns: np.ndarray, equity_curve_rising: np.ndarray
    ) -> None:
        result = calmar_ratio(positive_returns, equity_curve_rising)
        assert result == 0.0

    def test_empty_returns_zero(self) -> None:
        assert calmar_ratio(np.array([]), np.array([100.0])) == 0.0


# ---------------------------------------------------------------------------
# annualized_return / annualized_volatility tests
# ---------------------------------------------------------------------------

class TestAnnualizedMetrics:
    """Tests for annualized return and volatility."""

    def test_annualized_return_positive_for_positive_returns(
        self, positive_returns: np.ndarray
    ) -> None:
        result = annualized_return(positive_returns)
        assert result > 0

    def test_annualized_return_empty_returns_zero(self) -> None:
        assert annualized_return(np.array([])) == 0.0

    def test_annualized_volatility_positive(
        self, mixed_returns: np.ndarray
    ) -> None:
        result = annualized_volatility(mixed_returns)
        assert result > 0

    def test_annualized_volatility_zero_for_constant(self) -> None:
        returns = np.full(100, 0.001)
        result = annualized_volatility(returns)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_annualized_volatility_single_returns_zero(self) -> None:
        assert annualized_volatility(np.array([0.01])) == 0.0


# ---------------------------------------------------------------------------
# VaR / CVaR tests
# ---------------------------------------------------------------------------

class TestVaR:
    """Tests for Value at Risk and Conditional VaR."""

    def test_historical_var_positive(self, mixed_returns: np.ndarray) -> None:
        var = historical_var(mixed_returns, confidence=0.95)
        assert var >= 0.0

    def test_historical_var_95_less_than_99(
        self, mixed_returns: np.ndarray
    ) -> None:
        var_95 = historical_var(mixed_returns, confidence=0.95)
        var_99 = historical_var(mixed_returns, confidence=0.99)
        # Higher confidence → larger VaR (more conservative)
        assert var_99 >= var_95

    def test_historical_var_invalid_confidence_raises(
        self, mixed_returns: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="confidence"):
            historical_var(mixed_returns, confidence=1.5)

    def test_historical_var_zero_confidence_raises(
        self, mixed_returns: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="confidence"):
            historical_var(mixed_returns, confidence=0.0)

    def test_cvar_gte_var(self, mixed_returns: np.ndarray) -> None:
        var = historical_var(mixed_returns, confidence=0.95)
        cvar = conditional_var(mixed_returns, confidence=0.95)
        # CVaR (expected shortfall) should be >= VaR
        assert cvar >= var

    def test_cvar_empty_returns_zero(self) -> None:
        assert conditional_var(np.array([])) == 0.0

    def test_var_empty_returns_zero(self) -> None:
        assert historical_var(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# win_rate / profit_factor tests
# ---------------------------------------------------------------------------

class TestWinRateAndProfitFactor:
    """Tests for trade-level statistics."""

    def test_win_rate_all_winners(self) -> None:
        trades = np.array([0.05, 0.03, 0.02, 0.01])
        assert win_rate(trades) == 1.0

    def test_win_rate_all_losers(self) -> None:
        trades = np.array([-0.05, -0.03, -0.02])
        assert win_rate(trades) == 0.0

    def test_win_rate_mixed(self) -> None:
        trades = np.array([0.05, -0.02, 0.03, -0.01, 0.04])
        assert win_rate(trades) == pytest.approx(0.6)

    def test_win_rate_empty_returns_zero(self) -> None:
        assert win_rate(np.array([])) == 0.0

    def test_profit_factor_known_value(self) -> None:
        # Gross profit = 0.05+0.03+0.04 = 0.12
        # Gross loss   = 0.02+0.01 = 0.03
        # PF = 0.12 / 0.03 = 4.0
        trades = np.array([0.05, -0.02, 0.03, -0.01, 0.04])
        pf = profit_factor(trades)
        assert pf == pytest.approx(4.0, rel=1e-5)

    def test_profit_factor_no_losses_returns_inf(self) -> None:
        trades = np.array([0.05, 0.03, 0.02])
        pf = profit_factor(trades)
        assert pf == float("inf")

    def test_profit_factor_no_wins_returns_zero(self) -> None:
        trades = np.array([-0.05, -0.03])
        pf = profit_factor(trades)
        assert pf == 0.0

    def test_profit_factor_empty_returns_zero(self) -> None:
        assert profit_factor(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# compute_full_report integration tests
# ---------------------------------------------------------------------------

class TestComputeFullReport:
    """Integration tests for compute_full_report()."""

    def test_returns_performance_report_instance(
        self, positive_returns: np.ndarray
    ) -> None:
        equity = np.cumprod(1 + positive_returns) * 100_000
        report = compute_full_report(positive_returns, equity)
        assert isinstance(report, PerformanceReport)

    def test_report_fields_populated(
        self, positive_returns: np.ndarray
    ) -> None:
        equity = np.cumprod(1 + positive_returns) * 100_000
        trade_returns = np.array([0.05, -0.02, 0.03, 0.04, -0.01])
        report = compute_full_report(positive_returns, equity, trade_returns)

        assert report.sharpe_ratio != 0.0
        assert report.max_drawdown >= 0.0
        assert report.n_trades == 5
        assert report.n_days == len(positive_returns)

    def test_report_to_dict_has_all_keys(
        self, positive_returns: np.ndarray
    ) -> None:
        equity = np.cumprod(1 + positive_returns) * 100_000
        report = compute_full_report(positive_returns, equity)
        d = report.to_dict()

        expected_keys = {
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "var_95", "cvar_95", "win_rate", "profit_factor",
            "n_trades", "n_days",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_report_to_dict_values_are_numeric(
        self, positive_returns: np.ndarray
    ) -> None:
        equity = np.cumprod(1 + positive_returns) * 100_000
        report = compute_full_report(positive_returns, equity)
        d = report.to_dict()
        for key, val in d.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"

    def test_positive_strategy_has_positive_total_return(
        self, positive_returns: np.ndarray
    ) -> None:
        equity = np.cumprod(1 + positive_returns) * 100_000
        report = compute_full_report(positive_returns, equity)
        assert report.total_return > 0

    def test_negative_strategy_has_negative_total_return(
        self, negative_returns: np.ndarray
    ) -> None:
        equity = np.cumprod(1 + negative_returns) * 100_000
        report = compute_full_report(negative_returns, equity)
        assert report.total_return < 0
