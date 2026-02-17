# ============================================================
# VELLA_v10_BASE — LONG ONLY (EMA9↑EMA14 / EXIT=EMA4)
# - EXECUTION CORE: based on v9 proven trade plumbing (lotSize/qty/order/reduceOnly/closed-bar loop)
# - ENTRY: EMA9 crosses ABOVE EMA14 (Golden Cross)
# - EXIT: close < EMA4 (CLOSE_LT_EMA mode default)
# - TIME AXIS: REST closed-bar only (kline[-2])
# ============================================================

import os
import sys
import time
import signal
import logging
import requests
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# ============================================================
# CFG (ALL CONTROL HERE)
# ============================================================
# 20250210_BASE : 완전 단순 EMA3선 교차 실험 모드
# 20260217_LONG : EMA9↑EMA14 확정 / EXIT=EMA4 / SHORT/stack 제거

CFG = {
    # -------------------------
    # BASIC
    # -------------------------
    "01_TRADE_SYMBOL": "CCUSDT",
    "02_INTERVAL": "5m",
    "03_CAPITAL_BASE_USDT": 30.0,
    "04_LEVERAGE": 1,

    # -------------------------
    # ENTRY (LONG ONLY)
    # - trigger: EMA_FAST crosses ABOVE EMA_MID
    # -------------------------
    "10_EMA_FAST": 9,
    "11_EMA_MID": 14,

    # -------------------------
    # ENTRY MANAGEMENT
    # -------------------------
    "20_ENTRY_COOLDOWN_BARS": 0,

    # -------------------------
    # EXIT
    # -------------------------
    "30_EXIT_EMA": 4,
    "31_EXIT_MODE": "CLOSE_LT_EMA",  # "CLOSE_LT_EMA" / "CROSSUNDER"

    # -------------------------
    # EXIT OPTIONS (plug-in slots; default OFF)
    # -------------------------
    "40_SL_ENABLE": False,
    "41_SL_PCT": 2.0,

    "50_TIMEOUT_EXIT_ENABLE": False,
    "51_TIMEOUT_BARS": 60,

    # -------------------------
    # ENGINE
    # -------------------------
    "90_KLINE_LIMIT": 1500,
    "91_POLL_SEC": 7,
    "92_LOG_LEVEL": "INFO",
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CFG["92_LOG_LEVEL"], logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VELLA_v10_BASE")

# ============================================================
# BINANCE (v9 style)
# ============================================================

try:
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
except Exception:
    Client = None
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    ORDER_TYPE_MARKET = "MARKET"

BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"

def init_client() -> "Client":
    if Client is None:
        raise RuntimeError("python-binance missing. pip install python-binance")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET env vars.")
    return Client(api_key, api_secret)

def set_leverage(client: "Client", symbol: str, leverage: int) -> None:
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        log.error(f"set_leverage failed: {e}")

def fetch_klines_futures(symbol: str, interval: str, limit: int) -> Optional[List[Any]]:
    try:
        r = requests.get(
            BINANCE_FUTURES_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_klines_futures: {e}")
        return None

def get_futures_lot_size(client: "Client", symbol: str) -> Optional[Dict[str, Decimal]]:
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return {
                            "stepSize": Decimal(f["stepSize"]),
                            "minQty": Decimal(f["minQty"]),
                            "maxQty": Decimal(f["maxQty"]),
                        }
        return None
    except Exception as e:
        log.error(f"get_futures_lot_size: {e}")
        return None

def calculate_quantity(qty_raw: float, lot: Dict[str, Decimal]) -> Optional[float]:
    if lot is None:
        return None
    qty_decimal = Decimal(str(qty_raw))
    step = lot["stepSize"]
    qty = (qty_decimal / step).quantize(Decimal("1"), rounding=ROUND_DOWN) * step
    if qty < lot["minQty"]:
        return None
    if qty > lot["maxQty"]:
        qty = lot["maxQty"]
    precision = abs(step.as_tuple().exponent)
    return float(qty.quantize(Decimal(10) ** -precision))

# ============================================================
# INDICATORS
# ============================================================

def ema_series(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    if len(values) < period:
        return [values[0]] * len(values)
    k = 2 / (period + 1)
    out = [values[0]] * len(values)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = (values[i] * k) + (prev * (1 - k))
        out[i] = prev
    for i in range(period - 1):
        out[i] = out[period - 1]
    return out

# ============================================================
# STATE
# ============================================================

@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    entry_bar: int

@dataclass
class EngineState:
    bar: int = 0
    last_open_time: Optional[int] = None
    cooldown_until_bar: int = 0
    position: Optional[Position] = None
    close_history: List[float] = field(default_factory=list)

# ============================================================
# ENTRY (LONG ONLY — EMA9 crosses ABOVE EMA14)
# ============================================================

def base_entry_signal(closes: List[float]) -> Optional[str]:
    if len(closes) < 60:
        return None

    ema_fast_s = ema_series(closes, CFG["10_EMA_FAST"])
    ema_mid_s  = ema_series(closes, CFG["11_EMA_MID"])

    cross_up = (
        (ema_fast_s[-2] <= ema_mid_s[-2]) and
        (ema_fast_s[-1] > ema_mid_s[-1])
    )

    if cross_up:
        return "LONG"

    return None

# ============================================================
# EXIT (EMA4 based + SL + TIMEOUT)
# ============================================================

def exit_option_sl(close: float, entry_price: float, side: str) -> bool:
    if not CFG["40_SL_ENABLE"]:
        return False
    sl = float(CFG["41_SL_PCT"]) / 100.0
    if side == "LONG":
        return close <= entry_price * (1.0 - sl)
    else:
        return close >= entry_price * (1.0 + sl)

def exit_option_timeout(current_bar: int, entry_bar: int) -> bool:
    if not CFG["50_TIMEOUT_EXIT_ENABLE"]:
        return False
    return (current_bar - entry_bar) >= int(CFG["51_TIMEOUT_BARS"])

def exit_signal(state: EngineState) -> bool:
    pos = state.position
    if pos is None:
        return False

    close_now = state.close_history[-1]

    if exit_option_sl(close_now, pos.entry_price, pos.side):
        return True
    if exit_option_timeout(state.bar, pos.entry_bar):
        return True

    ema_exit_s   = ema_series(state.close_history, int(CFG["30_EXIT_EMA"]))
    ema_exit_now = ema_exit_s[-1]
    mode         = CFG["31_EXIT_MODE"]

    if mode == "CLOSE_LT_EMA":
        if close_now < ema_exit_now:
            return True

    elif mode == "CROSSUNDER":
        close_prev    = state.close_history[-2]
        ema_exit_prev = ema_exit_s[-2]
        crossunder = (close_prev >= ema_exit_prev) and (close_now < ema_exit_now)
        if crossunder:
            return True

    return False

# ============================================================
# EXECUTION (v9-style order plumbing)
# ============================================================

def place_long_entry(client: "Client", symbol: str, capital_usdt: float, lot: Dict[str, Decimal]) -> Optional[Dict[str, Any]]:
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        leverage = int(CFG["04_LEVERAGE"])
        notional = float(capital_usdt) * float(leverage)
        qty_raw = notional / price
        qty = calculate_quantity(qty_raw, lot)
        if qty is None:
            log.error("entry: qty calculation failed (minQty/stepSize)")
            return None
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=qty,
        )
        return {"entry_price": price, "qty": qty}
    except Exception as e:
        log.error(f"place_long_entry: {e}")
        return None

def place_long_exit(client: "Client", symbol: str, qty: float, lot: Dict[str, Decimal]) -> bool:
    try:
        qty_rounded = calculate_quantity(qty, lot)
        if qty_rounded is None:
            log.error("exit: qty too small (minQty) — cannot close")
            return False
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=qty_rounded,
            reduceOnly=True
        )
        return True
    except Exception as e:
        log.error(f"place_long_exit: {e}")
        return False

# ============================================================
# ENGINE LOOP (closed bar only)
# ============================================================

STOP = False
def _sig_handler(_sig, _frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

def engine():
    client = init_client()
    symbol = CFG["01_TRADE_SYMBOL"]
    interval = CFG["02_INTERVAL"]
    capital = float(CFG["03_CAPITAL_BASE_USDT"])

    set_leverage(client, symbol, int(CFG["04_LEVERAGE"]))

    lot = get_futures_lot_size(client, symbol)
    if lot is None:
        raise RuntimeError("lot_size retrieval failed")

    st = EngineState()

    log.info(f"START v10_BASE (EMA9↑EMA14 / EXIT=EMA4) | symbol={symbol} interval={interval} capital={capital} lev={CFG['04_LEVERAGE']}")

    while not STOP:
        try:
            kl = fetch_klines_futures(symbol, interval, int(CFG["90_KLINE_LIMIT"]))
            if not kl:
                time.sleep(CFG["91_POLL_SEC"])
                continue

            completed = kl[-2]
            open_time = int(completed[0])

            if st.last_open_time == open_time:
                time.sleep(CFG["91_POLL_SEC"])
                continue

            if not st.close_history:
                for k in kl[:-1]:
                    st.close_history.append(float(k[4]))
                st.bar = len(st.close_history)
                st.last_open_time = int(kl[-2][0])
                log.info(f"COLD START: loaded {st.bar} bars")
                continue

            st.last_open_time = open_time
            st.bar += 1

            close = float(completed[4])
            st.close_history.append(close)

            if len(st.close_history) > 2000:
                st.close_history = st.close_history[-2000:]

            if st.position is None:
                if st.bar < st.cooldown_until_bar:
                    continue

                direction = base_entry_signal(st.close_history)

                if direction == "LONG":
                    order = place_long_entry(client, symbol, capital, lot)
                    if order:
                        st.position = Position(
                            side="LONG",
                            entry_price=float(order["entry_price"]),
                            qty=float(order["qty"]),
                            entry_bar=st.bar,
                        )
                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd
                        log.info(f"[ENTRY] LONG qty={st.position.qty} entry={st.position.entry_price} bar={st.bar}")
                    else:
                        log.error("[ENTRY_FAIL] order failed")

            else:
                if st.position.entry_bar == st.bar:
                    continue

                if exit_signal(st):
                    ok = place_long_exit(client, symbol, st.position.qty, lot)
                    if ok:
                        log.info(f"[EXIT] LONG close={close} entry={st.position.entry_price} bar={st.bar}")
                        st.position = None
                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd
                    else:
                        log.error("[EXIT_FAIL] order failed (kept position)")

        except Exception as e:
            log.error(f"engine loop error: {e}")
            time.sleep(CFG["91_POLL_SEC"])

    log.info("STOP v10_BASE")

if __name__ == "__main__":
    engine()