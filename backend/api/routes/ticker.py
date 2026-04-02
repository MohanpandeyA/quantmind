"""Ticker search endpoint — hybrid Yahoo Finance + curated fallback.

Strategy:
    1. Try Yahoo Finance query2 endpoint (more permissive than query1)
    2. If Yahoo returns 429/error, fall back to built-in curated list
    3. Curated list covers 200+ popular US + Indian tickers

Endpoint:
    GET /ticker/search?q=reliance  → [{symbol, name, exchange, type}, ...]
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ticker", tags=["Ticker"])

# Yahoo Finance query2 is less rate-limited than query1
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
    "Cookie": "B=abc123; GUC=AQEBCAFmXxBmXxBmXxBm",
}

# ---------------------------------------------------------------------------
# Curated ticker list — 200+ popular US + Indian stocks
# Format: (symbol, name, exchange, type)
# ---------------------------------------------------------------------------
CURATED_TICKERS: list[tuple[str, str, str, str]] = [
    # ── US Large Cap ──────────────────────────────────────────────────────
    ("AAPL",  "Apple Inc.",                          "NASDAQ", "EQUITY"),
    ("MSFT",  "Microsoft Corporation",               "NASDAQ", "EQUITY"),
    ("GOOGL", "Alphabet Inc. (Google)",              "NASDAQ", "EQUITY"),
    ("GOOG",  "Alphabet Inc. Class C",               "NASDAQ", "EQUITY"),
    ("AMZN",  "Amazon.com Inc.",                     "NASDAQ", "EQUITY"),
    ("NVDA",  "NVIDIA Corporation",                  "NASDAQ", "EQUITY"),
    ("META",  "Meta Platforms Inc.",                 "NASDAQ", "EQUITY"),
    ("TSLA",  "Tesla Inc.",                          "NASDAQ", "EQUITY"),
    ("BRK-B", "Berkshire Hathaway Inc.",             "NYSE",   "EQUITY"),
    ("JPM",   "JPMorgan Chase & Co.",                "NYSE",   "EQUITY"),
    ("V",     "Visa Inc.",                           "NYSE",   "EQUITY"),
    ("MA",    "Mastercard Inc.",                     "NYSE",   "EQUITY"),
    ("UNH",   "UnitedHealth Group Inc.",             "NYSE",   "EQUITY"),
    ("JNJ",   "Johnson & Johnson",                   "NYSE",   "EQUITY"),
    ("WMT",   "Walmart Inc.",                        "NYSE",   "EQUITY"),
    ("XOM",   "Exxon Mobil Corporation",             "NYSE",   "EQUITY"),
    ("PG",    "Procter & Gamble Co.",                "NYSE",   "EQUITY"),
    ("LLY",   "Eli Lilly and Company",               "NYSE",   "EQUITY"),
    ("HD",    "The Home Depot Inc.",                 "NYSE",   "EQUITY"),
    ("CVX",   "Chevron Corporation",                 "NYSE",   "EQUITY"),
    ("MRK",   "Merck & Co. Inc.",                    "NYSE",   "EQUITY"),
    ("ABBV",  "AbbVie Inc.",                         "NYSE",   "EQUITY"),
    ("KO",    "The Coca-Cola Company",               "NYSE",   "EQUITY"),
    ("PEP",   "PepsiCo Inc.",                        "NASDAQ", "EQUITY"),
    ("COST",  "Costco Wholesale Corporation",        "NASDAQ", "EQUITY"),
    ("AVGO",  "Broadcom Inc.",                       "NASDAQ", "EQUITY"),
    ("NFLX",  "Netflix Inc.",                        "NASDAQ", "EQUITY"),
    ("AMD",   "Advanced Micro Devices Inc.",         "NASDAQ", "EQUITY"),
    ("INTC",  "Intel Corporation",                   "NASDAQ", "EQUITY"),
    ("CSCO",  "Cisco Systems Inc.",                  "NASDAQ", "EQUITY"),
    ("ADBE",  "Adobe Inc.",                          "NASDAQ", "EQUITY"),
    ("CRM",   "Salesforce Inc.",                     "NYSE",   "EQUITY"),
    ("ORCL",  "Oracle Corporation",                  "NYSE",   "EQUITY"),
    ("IBM",   "International Business Machines",     "NYSE",   "EQUITY"),
    ("QCOM",  "Qualcomm Inc.",                       "NASDAQ", "EQUITY"),
    ("TXN",   "Texas Instruments Inc.",              "NASDAQ", "EQUITY"),
    ("AMAT",  "Applied Materials Inc.",              "NASDAQ", "EQUITY"),
    ("MU",    "Micron Technology Inc.",              "NASDAQ", "EQUITY"),
    ("LRCX",  "Lam Research Corporation",            "NASDAQ", "EQUITY"),
    ("NOW",   "ServiceNow Inc.",                     "NYSE",   "EQUITY"),
    ("UBER",  "Uber Technologies Inc.",              "NYSE",   "EQUITY"),
    ("LYFT",  "Lyft Inc.",                           "NASDAQ", "EQUITY"),
    ("ABNB",  "Airbnb Inc.",                         "NASDAQ", "EQUITY"),
    ("SNAP",  "Snap Inc.",                           "NYSE",   "EQUITY"),
    ("TWTR",  "Twitter / X Corp.",                   "NYSE",   "EQUITY"),
    ("PYPL",  "PayPal Holdings Inc.",                "NASDAQ", "EQUITY"),
    ("SQ",    "Block Inc. (Square)",                 "NYSE",   "EQUITY"),
    ("COIN",  "Coinbase Global Inc.",                "NASDAQ", "EQUITY"),
    ("HOOD",  "Robinhood Markets Inc.",              "NASDAQ", "EQUITY"),
    ("PLTR",  "Palantir Technologies Inc.",          "NYSE",   "EQUITY"),
    ("SNOW",  "Snowflake Inc.",                      "NYSE",   "EQUITY"),
    ("DDOG",  "Datadog Inc.",                        "NASDAQ", "EQUITY"),
    ("ZS",    "Zscaler Inc.",                        "NASDAQ", "EQUITY"),
    ("CRWD",  "CrowdStrike Holdings Inc.",           "NASDAQ", "EQUITY"),
    ("NET",   "Cloudflare Inc.",                     "NYSE",   "EQUITY"),
    ("SHOP",  "Shopify Inc.",                        "NYSE",   "EQUITY"),
    ("SPOT",  "Spotify Technology S.A.",             "NYSE",   "EQUITY"),
    ("RBLX",  "Roblox Corporation",                  "NYSE",   "EQUITY"),
    ("U",     "Unity Software Inc.",                 "NYSE",   "EQUITY"),
    ("RIVN",  "Rivian Automotive Inc.",              "NASDAQ", "EQUITY"),
    ("LCID",  "Lucid Group Inc.",                    "NASDAQ", "EQUITY"),
    ("NIO",   "NIO Inc.",                            "NYSE",   "EQUITY"),
    ("BABA",  "Alibaba Group Holding Ltd.",          "NYSE",   "EQUITY"),
    ("JD",    "JD.com Inc.",                         "NASDAQ", "EQUITY"),
    ("PDD",   "PDD Holdings Inc. (Temu/Pinduoduo)",  "NASDAQ", "EQUITY"),
    ("BIDU",  "Baidu Inc.",                          "NASDAQ", "EQUITY"),
    ("TSM",   "Taiwan Semiconductor Mfg. Co.",       "NYSE",   "EQUITY"),
    ("ASML",  "ASML Holding N.V.",                   "NASDAQ", "EQUITY"),
    ("SAP",   "SAP SE",                              "NYSE",   "EQUITY"),
    ("TM",    "Toyota Motor Corporation",            "NYSE",   "EQUITY"),
    ("SONY",  "Sony Group Corporation",              "NYSE",   "EQUITY"),
    ("BAC",   "Bank of America Corp.",               "NYSE",   "EQUITY"),
    ("WFC",   "Wells Fargo & Company",               "NYSE",   "EQUITY"),
    ("GS",    "The Goldman Sachs Group Inc.",        "NYSE",   "EQUITY"),
    ("MS",    "Morgan Stanley",                      "NYSE",   "EQUITY"),
    ("C",     "Citigroup Inc.",                      "NYSE",   "EQUITY"),
    ("AXP",   "American Express Company",            "NYSE",   "EQUITY"),
    ("BLK",   "BlackRock Inc.",                      "NYSE",   "EQUITY"),
    ("SCHW",  "Charles Schwab Corporation",          "NYSE",   "EQUITY"),
    ("SPY",   "SPDR S&P 500 ETF Trust",              "NYSE",   "ETF"),
    ("QQQ",   "Invesco QQQ Trust (NASDAQ-100)",      "NASDAQ", "ETF"),
    ("VTI",   "Vanguard Total Stock Market ETF",     "NYSE",   "ETF"),
    ("IWM",   "iShares Russell 2000 ETF",            "NYSE",   "ETF"),
    ("GLD",   "SPDR Gold Shares ETF",                "NYSE",   "ETF"),
    ("SLV",   "iShares Silver Trust ETF",            "NYSE",   "ETF"),
    ("GME",   "GameStop Corp.",                      "NYSE",   "EQUITY"),
    ("AMC",   "AMC Entertainment Holdings",          "NYSE",   "EQUITY"),
    ("BB",    "BlackBerry Limited",                  "NYSE",   "EQUITY"),
    # ── Indian Stocks (NSE) ───────────────────────────────────────────────
    ("RELIANCE.NS",   "Reliance Industries Limited",         "NSI", "EQUITY"),
    ("TCS.NS",        "Tata Consultancy Services Ltd.",       "NSI", "EQUITY"),
    ("HDFCBANK.NS",   "HDFC Bank Limited",                   "NSI", "EQUITY"),
    ("INFY.NS",       "Infosys Limited",                     "NSI", "EQUITY"),
    ("ICICIBANK.NS",  "ICICI Bank Limited",                  "NSI", "EQUITY"),
    ("HINDUNILVR.NS", "Hindustan Unilever Limited",          "NSI", "EQUITY"),
    ("ITC.NS",        "ITC Limited",                         "NSI", "EQUITY"),
    ("SBIN.NS",       "State Bank of India",                 "NSI", "EQUITY"),
    ("BHARTIARTL.NS", "Bharti Airtel Limited",               "NSI", "EQUITY"),
    ("KOTAKBANK.NS",  "Kotak Mahindra Bank Limited",         "NSI", "EQUITY"),
    ("LT.NS",         "Larsen & Toubro Limited",             "NSI", "EQUITY"),
    ("AXISBANK.NS",   "Axis Bank Limited",                   "NSI", "EQUITY"),
    ("ASIANPAINT.NS", "Asian Paints Limited",                "NSI", "EQUITY"),
    ("MARUTI.NS",     "Maruti Suzuki India Limited",         "NSI", "EQUITY"),
    ("SUNPHARMA.NS",  "Sun Pharmaceutical Industries",       "NSI", "EQUITY"),
    ("TITAN.NS",      "Titan Company Limited",               "NSI", "EQUITY"),
    ("BAJFINANCE.NS", "Bajaj Finance Limited",               "NSI", "EQUITY"),
    ("WIPRO.NS",      "Wipro Limited",                       "NSI", "EQUITY"),
    ("HCLTECH.NS",    "HCL Technologies Limited",            "NSI", "EQUITY"),
    ("ULTRACEMCO.NS", "UltraTech Cement Limited",            "NSI", "EQUITY"),
    ("NESTLEIND.NS",  "Nestle India Limited",                "NSI", "EQUITY"),
    ("POWERGRID.NS",  "Power Grid Corporation of India",     "NSI", "EQUITY"),
    ("NTPC.NS",       "NTPC Limited",                        "NSI", "EQUITY"),
    ("ONGC.NS",       "Oil and Natural Gas Corporation",     "NSI", "EQUITY"),
    ("TATAMOTORS.NS", "Tata Motors Limited",                 "NSI", "EQUITY"),
    ("TATASTEEL.NS",  "Tata Steel Limited",                  "NSI", "EQUITY"),
    ("ADANIENT.NS",   "Adani Enterprises Limited",           "NSI", "EQUITY"),
    ("ADANIPORTS.NS", "Adani Ports and SEZ Limited",         "NSI", "EQUITY"),
    ("ADANIGREEN.NS", "Adani Green Energy Limited",          "NSI", "EQUITY"),
    ("JSWSTEEL.NS",   "JSW Steel Limited",                   "NSI", "EQUITY"),
    ("HINDALCO.NS",   "Hindalco Industries Limited",         "NSI", "EQUITY"),
    ("CIPLA.NS",      "Cipla Limited",                       "NSI", "EQUITY"),
    ("DRREDDY.NS",    "Dr. Reddy's Laboratories",            "NSI", "EQUITY"),
    ("DIVISLAB.NS",   "Divi's Laboratories Limited",         "NSI", "EQUITY"),
    ("BAJAJFINSV.NS", "Bajaj Finserv Limited",               "NSI", "EQUITY"),
    ("EICHERMOT.NS",  "Eicher Motors Limited",               "NSI", "EQUITY"),
    ("HEROMOTOCO.NS", "Hero MotoCorp Limited",               "NSI", "EQUITY"),
    ("BPCL.NS",       "Bharat Petroleum Corporation",        "NSI", "EQUITY"),
    ("COALINDIA.NS",  "Coal India Limited",                  "NSI", "EQUITY"),
    ("GRASIM.NS",     "Grasim Industries Limited",           "NSI", "EQUITY"),
    ("TECHM.NS",      "Tech Mahindra Limited",               "NSI", "EQUITY"),
    ("INDUSINDBK.NS", "IndusInd Bank Limited",               "NSI", "EQUITY"),
    ("BRITANNIA.NS",  "Britannia Industries Limited",        "NSI", "EQUITY"),
    ("APOLLOHOSP.NS", "Apollo Hospitals Enterprise",         "NSI", "EQUITY"),
    ("TATACONSUM.NS", "Tata Consumer Products Limited",      "NSI", "EQUITY"),
    ("PIDILITIND.NS", "Pidilite Industries Limited",         "NSI", "EQUITY"),
    ("HAVELLS.NS",    "Havells India Limited",               "NSI", "EQUITY"),
    ("DMART.NS",      "Avenue Supermarts Limited (DMart)",   "NSI", "EQUITY"),
    ("ZOMATO.NS",     "Zomato Limited",                      "NSI", "EQUITY"),
    ("PAYTM.NS",      "One97 Communications (Paytm)",        "NSI", "EQUITY"),
    ("NYKAA.NS",      "FSN E-Commerce Ventures (Nykaa)",     "NSI", "EQUITY"),
    ("POLICYBZR.NS",  "PB Fintech (PolicyBazaar)",           "NSI", "EQUITY"),
    ("IRCTC.NS",      "Indian Railway Catering & Tourism",   "NSI", "EQUITY"),
    ("HAL.NS",        "Hindustan Aeronautics Limited",       "NSI", "EQUITY"),
    ("BEL.NS",        "Bharat Electronics Limited",          "NSI", "EQUITY"),
    ("MUTHOOTFIN.NS", "Muthoot Finance Limited",             "NSI", "EQUITY"),
    ("CHOLAFIN.NS",   "Cholamandalam Investment & Finance",  "NSI", "EQUITY"),
    ("SBILIFE.NS",    "SBI Life Insurance Company",          "NSI", "EQUITY"),
    ("HDFCLIFE.NS",   "HDFC Life Insurance Company",         "NSI", "EQUITY"),
    ("ICICIGI.NS",    "ICICI Lombard General Insurance",     "NSI", "EQUITY"),
    ("INFY",          "Infosys Limited (NYSE ADR)",          "NYSE", "EQUITY"),
    ("WIT",           "Wipro Limited (NYSE ADR)",            "NYSE", "EQUITY"),
    ("HDB",           "HDFC Bank Limited (NYSE ADR)",        "NYSE", "EQUITY"),
    ("IBN",           "ICICI Bank Limited (NYSE ADR)",       "NYSE", "EQUITY"),
]

# Pre-build lowercase search index for fast matching
_SEARCH_INDEX: list[dict[str, Any]] = [
    {
        "symbol": sym,
        "name": name,
        "exchange": exch,
        "type": typ,
        "type_display": {"EQUITY": "Equity", "ETF": "ETF", "MUTUALFUND": "Fund"}.get(typ, typ),
        "_search": f"{sym} {name}".lower(),
    }
    for sym, name, exch, typ in CURATED_TICKERS
]


def _search_curated(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Search the curated ticker list by symbol or company name."""
    q = query.lower().strip()
    results = []

    # Exact symbol prefix match first (highest priority)
    for item in _SEARCH_INDEX:
        if item["symbol"].lower().startswith(q):
            results.append(item)

    # Then name contains match
    for item in _SEARCH_INDEX:
        if item not in results and q in item["_search"]:
            results.append(item)

    # Return clean dicts without internal _search key
    return [
        {k: v for k, v in r.items() if k != "_search"}
        for r in results[:limit]
    ]


@router.get("/search")
async def search_ticker(
    q: str = Query(..., min_length=1, max_length=50, description="Search query"),
) -> JSONResponse:
    """Search for ticker symbols by company name or partial symbol.

    Tries Yahoo Finance first; falls back to curated list on rate limit/error.

    Args:
        q: Search query (e.g., "reliance", "apple", "AAPL")

    Returns:
        JSON: {"results": [{symbol, name, exchange, type, type_display}, ...]}
    """
    logger.info("Ticker search | q=%r", q)

    # Try Yahoo Finance first
    try:
        async with httpx.AsyncClient(timeout=4.0, headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(
                YAHOO_SEARCH_URL,
                params={
                    "q": q,
                    "quotesCount": 8,
                    "newsCount": 0,
                    "listsCount": 0,
                    "enableFuzzyQuery": False,
                    "quotesQueryId": "tss_match_phrase_query",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        quotes = data.get("quotes", [])
        results = []
        for quote in quotes:
            symbol = quote.get("symbol", "")
            name = quote.get("longname") or quote.get("shortname") or symbol
            exchange = quote.get("exchange", "")
            q_type = quote.get("quoteType", "EQUITY")

            if q_type in ("OPTION", "FUTURE", "CURRENCY", "CRYPTOCURRENCY"):
                continue

            type_display = {
                "EQUITY": "Equity", "ETF": "ETF",
                "MUTUALFUND": "Fund", "INDEX": "Index",
            }.get(q_type, q_type)

            results.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "type": q_type,
                "type_display": type_display,
            })

        if results:
            logger.info("Ticker search | yahoo | q=%r | results=%d", q, len(results))
            return JSONResponse(content={"results": results, "source": "yahoo"})

    except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
        logger.warning("Ticker search | yahoo failed | q=%r | %s — using curated list", q, type(e).__name__)

    # Fallback: curated list
    results = _search_curated(q)
    logger.info("Ticker search | curated | q=%r | results=%d", q, len(results))
    return JSONResponse(content={"results": results, "source": "curated"})
