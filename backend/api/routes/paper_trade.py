"""Paper Trading routes — execute BUY/SELL signals via Alpaca paper trading API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/paper-trade", tags=["Paper Trading"])


def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "Content-Type": "application/json",
    }


def _check_keys() -> None:
    if not settings.alpaca_api_key or not settings.alpaca_secret_key or settings.alpaca_api_key == "your_alpaca_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="Alpaca API keys not configured. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to backend/.env",
        )


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class PlaceOrderRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    side: str = Field(..., description="'buy' or 'sell'")
    qty: float = Field(..., gt=0, description="Number of shares")
    signal: Optional[str] = Field(None, description="BUY/SELL/HOLD signal from QuantMind")
    strategy: Optional[str] = Field(None, description="Strategy that generated the signal")
    note: Optional[str] = Field(None, description="Optional note")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/order", summary="Place a paper trade order")
async def place_order(req: PlaceOrderRequest) -> Dict[str, Any]:
    """Place a market order on Alpaca paper trading.

    Args:
        req: Order details (ticker, side, qty).

    Returns:
        Alpaca order response with order_id, status, filled_qty.
    """
    _check_keys()

    payload = {
        "symbol": req.ticker.upper(),
        "qty": str(req.qty),
        "side": req.side.lower(),
        "type": "market",
        "time_in_force": "day",
    }

    logger.info(
        "PaperTrade | placing order | %s %s x%.0f | signal=%s",
        req.side.upper(), req.ticker, req.qty, req.signal,
    )

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{settings.alpaca_base_url}/v2/orders",
            json=payload,
            headers=_alpaca_headers(),
        )

    if resp.status_code not in (200, 201):
        logger.error("PaperTrade | order failed | %d | %s", resp.status_code, resp.text)
        raise HTTPException(status_code=resp.status_code, detail=resp.json().get("message", resp.text))

    order = resp.json()
    logger.info(
        "PaperTrade | order placed | id=%s | status=%s",
        order.get("id"), order.get("status"),
    )

    return {
        "order_id": order.get("id"),
        "ticker": order.get("symbol"),
        "side": order.get("side"),
        "qty": order.get("qty"),
        "status": order.get("status"),
        "filled_qty": order.get("filled_qty"),
        "filled_avg_price": order.get("filled_avg_price"),
        "created_at": order.get("created_at"),
        "signal": req.signal,
        "strategy": req.strategy,
        "note": req.note,
    }


@router.get("/portfolio", summary="Get paper trading portfolio")
async def get_paper_portfolio() -> Dict[str, Any]:
    """Get current paper trading account positions and P&L.

    Returns:
        Account equity, cash, positions with unrealized P&L.
    """
    _check_keys()

    async with httpx.AsyncClient(timeout=15) as client:
        account_resp, positions_resp = await asyncio.gather(
            client.get(f"{settings.alpaca_base_url}/v2/account", headers=_alpaca_headers()),
            client.get(f"{settings.alpaca_base_url}/v2/positions", headers=_alpaca_headers()),
        )

    if account_resp.status_code != 200:
        raise HTTPException(status_code=account_resp.status_code, detail="Failed to fetch Alpaca account")

    account = account_resp.json()
    positions = positions_resp.json() if positions_resp.status_code == 200 else []

    formatted_positions = [
        {
            "ticker": p.get("symbol"),
            "qty": float(p.get("qty", 0)),
            "avg_entry_price": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "market_value": float(p.get("market_value", 0)),
            "cost_basis": float(p.get("cost_basis", 0)),
            "unrealized_pl": float(p.get("unrealized_pl", 0)),
            "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
            "side": p.get("side"),
        }
        for p in positions
    ]

    return {
        "equity": float(account.get("equity", 0)),
        "cash": float(account.get("cash", 0)),
        "buying_power": float(account.get("buying_power", 0)),
        "portfolio_value": float(account.get("portfolio_value", 0)),
        "initial_capital": 100000.0,
        "total_pnl": float(account.get("equity", 0)) - 100000.0,
        "total_pnl_pct": (float(account.get("equity", 0)) - 100000.0) / 100000.0,
        "position_count": len(formatted_positions),
        "positions": formatted_positions,
        "account_status": account.get("status"),
    }


@router.get("/orders", summary="Get recent paper trade orders")
async def get_orders(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent paper trading order history.

    Args:
        limit: Max orders to return (default 20).

    Returns:
        List of recent orders with status and fill details.
    """
    _check_keys()

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{settings.alpaca_base_url}/v2/orders",
            params={"limit": limit, "status": "all"},
            headers=_alpaca_headers(),
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch orders")

    orders = resp.json()
    return [
        {
            "order_id": o.get("id"),
            "ticker": o.get("symbol"),
            "side": o.get("side"),
            "qty": o.get("qty"),
            "filled_qty": o.get("filled_qty"),
            "filled_avg_price": o.get("filled_avg_price"),
            "status": o.get("status"),
            "created_at": o.get("created_at"),
            "filled_at": o.get("filled_at"),
        }
        for o in orders
    ]


@router.delete("/order/{order_id}", summary="Cancel a pending order")
async def cancel_order(order_id: str) -> Dict[str, str]:
    """Cancel a pending paper trade order.

    Args:
        order_id: Alpaca order ID.

    Returns:
        Cancellation confirmation.
    """
    _check_keys()

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.delete(
            f"{settings.alpaca_base_url}/v2/orders/{order_id}",
            headers=_alpaca_headers(),
        )

    if resp.status_code == 204:
        return {"status": "cancelled", "order_id": order_id}
    raise HTTPException(status_code=resp.status_code, detail="Failed to cancel order")


# asyncio needed for gather
import asyncio  # noqa: E402
