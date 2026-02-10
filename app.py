# ============================================================
# VELLA_v10 — LONG ENGINE (Binance Futures)
# - EXECUTION CORE: based on v9 proven trade plumbing (lotSize/qty/order/reduceOnly/closed-bar loop)
# - ENTRY: Claude v11 philosophy (EMA10/15/20 stack + spread + slope + peak + deep pullback within N bars + 1-shot per trend cycle)
# - EXIT: Bella rule (close < avg(prev N closed closes))  [default N=2, CFG adjustable]
# - TIME AXIS: REST closed-bar only (kline[-2])
# ============================================================

import os
import sys
import time
import signal
import logging
import requests
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# ============================================================
# CFG (ALL CONTROL HERE)
# ============================================================
# 20260210_1730 : 클롣드 엔트리 + 벨라 n봉 엑시트 기본 매매확인후 필터 추가

CFG = {
    # -------------------------
    # BASIC
    # -------------------------
    "01_TRADE_SYMBOL": "CCUSDT",
    "02_INTERVAL": "5m",
    "03_CAPITAL_BASE_USDT": 30.0,
    "04_LEVERAGE": 1,

    # -------------------------
    # ENTRY (Claude v11)
    # -------------------------
    "10_EMA_FAST": 10,
    "11_EMA_MID": 15,
    "12_EMA_SLOW": 20,

    "13_PULLBACK_N": 8,          # 눌림 허용 봉수
    "14_SLOPE_BARS": 5,          # EMA10 기울기 비교 봉수
    "15_SPREAD_MIN": 0.0004,     # EMA10-EMA20 최소 확장값 (절대값)
    "16_PEAK_BARS": 30,          # EMA10 고점(최근 N봉 최고) 확인

    # -------------------------
    # ENTRY MANAGEMENT FILTERS (plug-in slots)
    # -------------------------
    "20_ENTRY_COOLDOWN_BARS": 0,     # 엔트리/엑시트 후 재진입 쿨다운(봉수)
    "21_MAX_ENTRY_PER_TREND": 1,     # 추세 사이클당 1회(기본 1; Claude 철학)

    # -------------------------
    # EXIT (Bella)
    # -------------------------
    "30_EXIT_AVG_N": 2,              # 기본값 2, CFG로 조정
    "31_EXIT_USE_PREV_N_ONLY": True, # True: avg=prev N closes(현재봉 제외). False면 포함(권장X)

    # -------------------------
    # EXIT OPTIONS (plug-in slots; default OFF)
    # -------------------------
    "40_SL_ENABLE": False,
    "41_SL_PCT": 2.0,                # % (롱: close <= entry*(1-sl))

    "50_TIMEOUT_EXIT_ENABLE": False,
    "51_TIMEOUT_BARS": 60,           # 포지션 보유 제한(봉수)

    # -------------------------
    # ENGINE
    # -------------------------
    "90_KLINE_LIMIT": 240,            # 충분히 크게(EMA/Peak 위해)
    "91_POLL_SEC": 7,                 # 완료봉 갱신 체크 주기
    "92_LOG_LEVEL": "INFO",           # INFO/ERROR
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CFG["92_LOG_LEVEL"], logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VELLA_v10_LONG")

# ============================================================
# BINANCE (v9 style)
# ============================================================

try:
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL, FUTURE_ORDER_TYPE_MARKET
except Exception:
    Client = None
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    FUTURE_ORDER_TYPE_MARKET = "MARKET"

BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"

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
        # leverage set can fail if already set or symbol restrictions; do not crash engine
        log.error(f"set_leverage failed: {e}")

def fetch_klines_spot(symbol: str, interval: str, limit: int) -> Optional[List[Any]]:
    # NOTE: v9 used spot klines endpoint for signal time-axis (closed bars).
    # For Futures, spot klines are fine as a deterministic bar feed for many USDT symbols.
    try:
        r = requests.get(
            BINANCE_SPOT_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_klines_spot: {e}")
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
    """
    v9-style qty rounding: floor to stepSize, enforce minQty/maxQty.
    """
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
    """
    Deterministic EMA series, aligned 1:1 with input values.
    Seed = SMA(period) at index period-1, then EMA forward.
    For indexes < period-1, we fill with the first value (safe warm-up).
    """
    if not values:
        return []
    if len(values) < period:
        # not enough, return flat warmup series
        return [values[0]] * len(values)

    k = 2 / (period + 1)
    out = [values[0]] * len(values)

    sma = sum(values[:period]) / period
    out[period - 1] = sma
    prev = sma
    for i in range(period, len(values)):
        prev = (values[i] * k) + (prev * (1 - k))
        out[i] = prev

    # fill warm-up head with first computed SMA (more stable than raw)
    for i in range(period - 1):
        out[i] = out[period - 1]
    return out

# ============================================================
# STATE
# ============================================================

@dataclass
class Position:
    side: str               # "LONG"
    entry_price: float
    qty: float
    entry_bar: int

@dataclass
class ClaudeEntryState:
    entry_fired: bool = False
    deep_seen: bool = False
    bars_since_deep: int = 999

@dataclass
class EngineState:
    bar: int = 0
    last_open_time: Optional[int] = None
    cooldown_until_bar: int = 0
    entry_state: ClaudeEntryState = ClaudeEntryState()
    position: Optional[Position] = None
    close_history: List[float] = None
    low_history: List[float] = None

    def __post_init__(self):
        self.close_history = []
        self.low_history = []

# ============================================================
# ENTRY (Claude v11)
# ============================================================

def claude_entry_signal(
    closes: List[float],
    lows: List[float],
    st: ClaudeEntryState
) -> bool:
    """
    Uses last closed bar only (already supplied as histories of closed bars).
    Claude v11 philosophy:
      - stack: EMA10 > EMA15 > EMA20
      - spread: EMA10-EMA20 >= SPREAD_MIN
      - slope: EMA10_now > EMA10[past slope_bars]
      - peak: EMA10_now >= highest EMA10 over PEAK_BARS
      - deep pullback happened: low <= EMA20, then entry within PULLBACK_N bars
      - cycle entry 1-shot: entry_fired blocks until stack breaks
      - final trigger: close > EMA10
    """
    if len(closes) < max(CFG["16_PEAK_BARS"], 60):
        return False

    ema10_s = ema_series(closes, CFG["10_EMA_FAST"])
    ema15_s = ema_series(closes, CFG["11_EMA_MID"])
    ema20_s = ema_series(closes, CFG["12_EMA_SLOW"])

    ema10 = ema10_s[-1]
    ema15 = ema15_s[-1]
    ema20 = ema20_s[-1]
    close = closes[-1]
    low = lows[-1]

    stack_now = (ema10 > ema15) and (ema15 > ema20)
    if not stack_now:
        # reset cycle
        st.entry_fired = False
        st.deep_seen = False
        st.bars_since_deep = 999
        return False

    spread_ratio = (ema10 - ema20) / ema20
    spread_ok = spread_ratio >= float(CFG["15_SPREAD_MIN"])

    slope_bars = int(CFG["14_SLOPE_BARS"])
    if len(ema10_s) < slope_bars + 1:
        return False
    slope_ok = ema10 > ema10_s[-1 - slope_bars]

    peak_bars = int(CFG["16_PEAK_BARS"])
    ema10_peak = max(ema10_s[-peak_bars:])
    peak_ok = ema10 >= ema10_peak

    # Deep pullback tracking (low <= ema20)
    if low <= ema20:
        st.bars_since_deep = 0
        st.deep_seen = True
    else:
        st.bars_since_deep += 1

    recent_deep_ok = st.deep_seen and (st.bars_since_deep <= int(CFG["13_PULLBACK_N"]))

    # 1-shot per trend cycle
    if int(CFG["21_MAX_ENTRY_PER_TREND"]) <= 0:
        # safety: no entries
        return False

    if st.entry_fired:
        return False

    entry_raw = stack_now and spread_ok and slope_ok and peak_ok and recent_deep_ok and (close > ema10)
    return bool(entry_raw)

def on_entry_fired(st: ClaudeEntryState) -> None:
    st.entry_fired = True
    st.deep_seen = False
    st.bars_since_deep = 999

# ============================================================
# EXIT (Bella core + options)
# ============================================================

def bella_exit_core_avg_break(closes: List[float], n: int) -> bool:
    """
    Bella core:
      - avg = mean(prev N closed closes)
      - exit if current_close < avg
    """
    n = int(n)
    if n <= 0:
        return False
    if len(closes) < n + 1:
        return False

    current = closes[-1]
    if CFG["31_EXIT_USE_PREV_N_ONLY"]:
        prev = closes[-(n + 1):-1]
        avg = sum(prev) / n
    else:
        # not recommended, but allowed
        avg = sum(closes[-n:]) / n

    return current < avg

def exit_option_sl(close: float, entry_price: float) -> bool:
    if not CFG["40_SL_ENABLE"]:
        return False
    sl = float(CFG["41_SL_PCT"]) / 100.0
    return close <= entry_price * (1.0 - sl)

def exit_option_timeout(current_bar: int, entry_bar: int) -> bool:
    if not CFG["50_TIMEOUT_EXIT_ENABLE"]:
        return False
    return (current_bar - entry_bar) >= int(CFG["51_TIMEOUT_BARS"])

def exit_signal(state: EngineState) -> bool:
    """
    Exit priority (clean, extendable):
      1) SL (optional)
      2) TIMEOUT (optional)
      3) Bella core avg-break (always ON)
    """
    pos = state.position
    if pos is None:
        return False

    close = state.close_history[-1]

    if exit_option_sl(close, pos.entry_price):
        return True
    if exit_option_timeout(state.bar, pos.entry_bar):
        return True

    n = int(CFG["30_EXIT_AVG_N"])
    if bella_exit_core_avg_break(state.close_history, n):
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
            side=SIDE_BUY,  # LONG entry
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=qty,
        )

        return {"entry_price": price, "qty": qty}
    except Exception as e:
        log.error(f"place_long_entry: {e}")
        return None

def place_long_exit(client: "Client", symbol: str, qty: float, lot: Dict[str, Decimal]) -> bool:
    """
    Long close = SELL with reduceOnly=True
    """
    try:
        # Re-round qty defensively
        qty_rounded = calculate_quantity(qty, lot)
        if qty_rounded is None:
            log.error("exit: qty too small (minQty) — cannot close")
            return False

        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL,  # LONG exit
            type=FUTURE_ORDER_TYPE_MARKET,
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

    log.info(f"START v10 LONG | symbol={symbol} interval={interval} capital={capital} lev={CFG['04_LEVERAGE']}")

    while not STOP:
        try:
            kl = fetch_klines_spot(symbol, interval, int(CFG["90_KLINE_LIMIT"]))
            if not kl:
                time.sleep(CFG["91_POLL_SEC"])
                continue

            completed = kl[-2]                  # closed bar
            open_time = int(completed[0])       # ms

            if st.last_open_time == open_time:
                time.sleep(CFG["91_POLL_SEC"])
                continue

            st.last_open_time = open_time
            st.bar += 1

            close = float(completed[4])
            low = float(completed[3])

            st.close_history.append(close)
            st.low_history.append(low)

            # keep history bounded
            if len(st.close_history) > 2000:
                st.close_history = st.close_history[-2000:]
                st.low_history = st.low_history[-2000:]

            # -------------------------
            # ENTRY MANAGEMENT FILTERS (cooldown)
            # -------------------------
            if st.bar < st.cooldown_until_bar:
                # still update Claude state via claude_entry_signal call? -> No: avoid state drifting during cooldown.
                # We still must reset cycle if stack breaks; handled when cooldown ends.
                pass

            # -------------------------
            # POSITION LOGIC
            # -------------------------
            if st.position is None:
                # cooldown check
                if st.bar < st.cooldown_until_bar:
                    log.info(f"[WAIT] cooldown bars remaining={st.cooldown_until_bar - st.bar}")
                    continue

                # ENTRY SIGNAL
                sig_entry = claude_entry_signal(st.close_history, st.low_history, st.entry_state)
                if sig_entry:
                    order = place_long_entry(client, symbol, capital, lot)
                    if order:
                        st.position = Position(
                            side="LONG",
                            entry_price=float(order["entry_price"]),
                            qty=float(order["qty"]),
                            entry_bar=st.bar,
                        )
                        on_entry_fired(st.entry_state)

                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd

                        log.info(f"[ENTRY] LONG qty={st.position.qty} entry={st.position.entry_price} bar={st.bar}")
                    else:
                        log.error("[ENTRY_FAIL] order failed")
                else:
                    # optional: lightweight trace
                    log.info(f"[NO_ENTRY] bar={st.bar} close={close}")
            else:
                # avoid same-bar entry-exit
                if st.position.entry_bar == st.bar:
                    continue

                if exit_signal(st):
                    ok = place_long_exit(client, symbol, st.position.qty, lot)
                    if ok:
                        log.info(f"[EXIT] LONG close={close} entry={st.position.entry_price} bar={st.bar}")
                        st.position = None

                        # cooldown after exit
                        cd = int(CFG["20_ENTRY_COOLDOWN_BARS"])
                        if cd > 0:
                            st.cooldown_until_bar = st.bar + cd
                    else:
                        log.error("[EXIT_FAIL] order failed (kept position)")
                else:
                    log.info(f"[HOLD] bar={st.bar} close={close} entry={st.position.entry_price}")

        except Exception as e:
            log.error(f"engine loop error: {e}")
            time.sleep(CFG["91_POLL_SEC"])

    log.info("STOP v10 LONG")

if __name__ == "__main__":
    engine()
