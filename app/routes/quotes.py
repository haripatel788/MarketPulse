import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import APIRouter

router = APIRouter(prefix="/quotes", tags=["quotes"])

TAPE_SYMBOLS = [
    "SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "AMZN",
    "GOOGL", "META", "TSLA", "JPM", "AMD", "NFLX",
]
CACHE_TTL_SEC = 60
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

_cache: dict[str, object] = {"at": 0.0, "quotes": []}
_lock = threading.Lock()


def _fetch_quote(symbol: str) -> dict | None:
    try:
        r = requests.get(
            YAHOO_URL.format(symbol=symbol),
            params={"range": "1d", "interval": "1d"},
            headers=YAHOO_HEADERS,
            timeout=6,
        )
        r.raise_for_status()
        meta = r.json()["chart"]["result"][0]["meta"]
        price = float(meta["regularMarketPrice"])
        prev = float(meta["chartPreviousClose"])
    except Exception:
        return None
    change = price - prev
    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "change_pct": round(change / prev * 100.0, 2) if prev else 0.0,
        "as_of": int(meta.get("regularMarketTime") or 0),
    }


@router.get("")
def quotes() -> dict:
    """Last price and change vs previous close for the header tape, cached briefly."""
    with _lock:
        if time.time() - float(_cache["at"]) < CACHE_TTL_SEC and _cache["quotes"]:
            return {"quotes": _cache["quotes"], "cached": True}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [q for q in pool.map(_fetch_quote, TAPE_SYMBOLS) if q is not None]
    if results:
        with _lock:
            _cache["at"] = time.time()
            _cache["quotes"] = results
    return {"quotes": results, "cached": False}
