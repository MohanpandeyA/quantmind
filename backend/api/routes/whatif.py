"""What-If Simulator route — POST /whatif."""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
import yfinance as yf
from fastapi import APIRouter
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/whatif", tags=["What-If Simulator"])


class WhatIfRequest(BaseModel):
    ticker: str
    question: str
    scenario_type: str = "price_shock"
    price_shock_pct: Optional[float] = None
    historical_start: Optional[str] = None
    historical_end: Optional[str] = None
    strategy: Optional[str] = "momentum"
    current_signal: Optional[str] = None
    current_sharpe: Optional[float] = None


class WhatIfResponse(BaseModel):
    ticker: str
    question: str
    scenario_type: str
    scenario_description: str
    signal_change: Optional[str]
    original_signal: Optional[str]
    new_signal: Optional[str]
    price_impact: Optional[float]
    sharpe_impact: Optional[float]
    original_sharpe: Optional[float]
    new_sharpe: Optional[float]
    historical_occurrences: int
    avg_recovery_days: Optional[float]
    ai_explanation: str
    processing_time_ms: float


HISTORICAL_SCENARIOS = {
    "covid": ("2020-02-01", "2020-04-30", "COVID-19 crash (Feb-Apr 2020)"),
    "2022_bear": ("2022-01-01", "2022-12-31", "2022 bear market"),
    "2008": ("2008-09-01", "2009-03-31", "2008 financial crisis"),
}


def _detect_scenario(question: str):
    q = question.lower()
    shock_match = re.search(r"(drop|fall|decline|crash|down)\s+(\d+(?:\.\d+)?)\s*%", q)
    rise_match = re.search(
        r"(rise|gain|up|rally|jump|increase)\s+(\d+(?:\.\d+)?)\s*%", q
    )
    if shock_match:
        return "price_shock", -float(shock_match.group(2)), None, None
    if rise_match:
        return "price_shock", float(rise_match.group(2)), None, None
    if "covid" in q or "pandemic" in q:
        s, e, _ = HISTORICAL_SCENARIOS["covid"]
        return "historical_period", None, s, e
    if "2022" in q and ("bear" in q or "crash" in q):
        s, e, _ = HISTORICAL_SCENARIOS["2022_bear"]
        return "historical_period", None, s, e
    if "2008" in q or "financial crisis" in q:
        s, e, _ = HISTORICAL_SCENARIOS["2008"]
        return "historical_period", None, s, e
    return "price_shock", -10.0, None, None


def _run_mini_backtest(ticker: str, start: str, end: str, strategy: str) -> dict:
    try:
        from engine.backtester import BacktestConfig, Backtester
        from engine.strategies.base_strategy import StrategyConfig
        from engine.strategies.macd_strategy import MACDStrategy
        from engine.strategies.mean_reversion import MeanReversionStrategy
        from engine.strategies.momentum import MomentumStrategy
        from engine.strategies.rsi_strategy import RSIStrategy

        strat_config = StrategyConfig(initial_capital=100_000)
        strat_map = {
            "momentum": MomentumStrategy(strat_config),
            "mean_reversion": MeanReversionStrategy(strat_config),
            "rsi": RSIStrategy(strat_config),
            "macd": MACDStrategy(strat_config),
        }
        strat = strat_map.get(strategy, MomentumStrategy(strat_config))
        bt_config = BacktestConfig(ticker=ticker, start_date=start, end_date=end)
        bt = Backtester(config=bt_config, strategy=strat)
        result, report = bt.run()
        sig = (
            "BUY"
            if report.total_return > 0.05
            else "SELL" if report.total_return < -0.05 else "HOLD"
        )
        return {
            "sharpe": report.sharpe_ratio,
            "total_return": report.total_return,
            "signal": sig,
        }
    except Exception as e:
        logger.warning("WhatIf backtest failed: %s", e)
        return {"sharpe": 0.0, "total_return": 0.0, "signal": "HOLD"}


def _count_occurrences(ticker: str, shock_pct: float):
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty:
            return 0, 0.0
        closes = df["Close"].squeeze()
        daily_returns = closes.pct_change().dropna()
        threshold = shock_pct / 100
        if threshold < 0:
            count = int((daily_returns < threshold).sum())
        else:
            count = int((daily_returns > threshold).sum())
        recovery_days = []
        closes_list = closes.tolist()
        returns_list = daily_returns.tolist()
        for i, ret in enumerate(returns_list):
            if (threshold < 0 and ret < threshold) or (
                threshold > 0 and ret > threshold
            ):
                pre = closes_list[i]
                for j in range(i + 1, min(i + 90, len(closes_list))):
                    if threshold < 0 and closes_list[j] >= pre:
                        recovery_days.append(j - i)
                        break
                    elif threshold > 0 and closes_list[j] <= pre:
                        recovery_days.append(j - i)
                        break
        avg = float(np.mean(recovery_days)) if recovery_days else 0.0
        return count, avg
    except Exception:
        return 0, 0.0


def _explain(
    ticker,
    question,
    scenario_desc,
    orig_sig,
    new_sig,
    price_impact,
    new_sharpe,
    occurrences,
    avg_recovery,
    strategy,
):
    try:
        if not settings.groq_api_key:
            raise ValueError("no key")
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        prompt = (
            f"You are QuantMind AI. Answer this what-if question in 3-4 sentences with specific numbers.\n\n"
            f"Ticker: {ticker}\nQuestion: {question}\nScenario: {scenario_desc}\n"
            f"Strategy: {strategy}\nOriginal signal: {orig_sig} -> New signal: {new_sig}\n"
            f"Price impact: {price_impact:+.1f}%\nNew Sharpe: {new_sharpe:.2f}\n"
            f"Historical occurrences (5yr): {occurrences}\nAvg recovery: {avg_recovery:.0f} days\n\n"
            f"Explain: 1) signal change 2) financial impact 3) historical context 4) recommendation."
        )
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        sig_txt = (
            f"signal changes from {orig_sig} to {new_sig}"
            if orig_sig != new_sig
            else f"signal remains {new_sig}"
        )
        rec_txt = (
            f"Historically recovered in ~{avg_recovery:.0f} days."
            if avg_recovery > 0
            else ""
        )
        occ_txt = (
            f"This occurred {occurrences} times in 5 years." if occurrences > 0 else ""
        )
        return f"Under this scenario, the {strategy} strategy {sig_txt}. Price impact: {price_impact:+.1f}%, Sharpe: {new_sharpe:.2f}. {occ_txt} {rec_txt}".strip()


@router.post("", response_model=WhatIfResponse, summary="Run a what-if scenario")
async def run_whatif(req: WhatIfRequest) -> WhatIfResponse:
    t0 = time.perf_counter()
    ticker = req.ticker.upper()
    logger.info("WhatIf | ticker=%s | question=%r", ticker, req.question[:60])

    stype = req.scenario_type
    shock = req.price_shock_pct
    hstart = req.historical_start
    hend = req.historical_end

    if stype == "price_shock" and shock is None:
        stype, shock, hstart, hend = _detect_scenario(req.question)

    if stype == "price_shock" and shock is not None:
        desc = f"{shock:+.1f}% price {'drop' if shock < 0 else 'rise'} in {ticker}"
        hstart = hstart or "2022-01-01"
        hend = hend or "2024-12-31"
    elif stype == "historical_period" and hstart and hend:
        desc = f"{ticker} during {hstart} to {hend}"
    else:
        stype = "price_shock"
        shock = -10.0
        desc = f"-10% price drop in {ticker}"
        hstart = "2022-01-01"
        hend = "2024-12-31"

    bt = _run_mini_backtest(
        ticker, hstart or "2022-01-01", hend or "2024-12-31", req.strategy or "momentum"
    )
    orig_sig = req.current_signal or "HOLD"
    new_sig = bt["signal"]
    orig_sharpe = req.current_sharpe or 0.5
    new_sharpe = bt["sharpe"]
    price_impact = shock if shock is not None else bt["total_return"] * 100
    occurrences, avg_recovery = _count_occurrences(
        ticker, shock if shock is not None else -5.0
    )
    explanation = _explain(
        ticker,
        req.question,
        desc,
        orig_sig,
        new_sig,
        price_impact,
        new_sharpe,
        occurrences,
        avg_recovery,
        req.strategy or "momentum",
    )

    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "WhatIf | complete | ticker=%s | signal=%s->%s | time=%.0fms",
        ticker,
        orig_sig,
        new_sig,
        ms,
    )

    return WhatIfResponse(
        ticker=ticker,
        question=req.question,
        scenario_type=stype,
        scenario_description=desc,
        signal_change=f"{orig_sig} -> {new_sig}" if orig_sig != new_sig else None,
        original_signal=orig_sig,
        new_signal=new_sig,
        price_impact=round(price_impact, 2),
        sharpe_impact=round(new_sharpe - orig_sharpe, 3),
        original_sharpe=round(orig_sharpe, 3),
        new_sharpe=round(new_sharpe, 3),
        historical_occurrences=occurrences,
        avg_recovery_days=round(avg_recovery, 1) if avg_recovery > 0 else None,
        ai_explanation=explanation,
        processing_time_ms=round(ms, 1),
    )
