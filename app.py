"""
============================================================
VELLA RANGE LONG LADDER v10.7 (BR10 기준선 — 실전 투입 최종)
============================================================
BR10 v10.1 기준선 유지 + 거미줄 작전 철학 완전 반영.

[v10.2 패치 내역]
1. HARD SL → 10단 체결 완료 후에만 avg 기준 -7% 발동 (1~9단 완전 비활성)
2. TIMEOUT → 사실상 제거 (DEEP_FILL_STAGE=99, TIMEOUT_BARS=99999)
3. LADDER INVALIDATION → 완전 비활성화 (주석 처리)
4. TARGET PROFIT 단계별 상향:
   - 4~5단: 0.5% → 0.8%
   - 6~7단: 0.3% → 0.6%
   - 8~9단: 0.1% → 0.4%
   - 10단:  -0.08% → +0.3%
5. LADDER_GAP_PCT: 3% → 5%
6. SIZE_WEIGHTS: 중간 봉우리형 재설계
   [0.5, 0.7, 1.0, 1.4, 1.8, 1.4, 1.0, 0.8, 0.6, 0.5]

거미줄 작전 핵심 철학:
  - 1~9단: 무조건 버팀. SL/TIMEOUT/INVALIDATION 없음.
  - 10단 체결 후: avg 기준 -7% 최후 방어선 1개만 존재.
  - 평단이 완성될수록 작은 반등으로도 수익 탈출 가능.

EXIT 우선순위:
  1. HARD SL (10단 체결 완료 후에만)
  2. TIMEOUT (사실상 비활성)
  3. TP1 1.0% → 50% 부분청산 후 트레일링 전환
  4. TRAIL EXIT: 고점 추적 → -0.5% 반락 시 전량 청산
  ※ TP1 전: 지정가 EXIT 병행
  ※ TP1 후: 트레일링 EXIT 전용

상태 머신:
  WATCHING       — 포지션 없음. 4H 필터 + 5M 트리거 대기.
  LADDER_ACTIVE  — 거미줄 배치 완료. 체결 감시 (무효화 없음).
  POSITION_HOLD  — 포지션 존재. EXIT 동기화 및 최후 방어 관리.
  COOLDOWN       — 청산 완료 후 재진입 금지 대기.

재시작 sync:
  A: 포지션 있음 (amt > 0)        → POSITION_HOLD, tp1_done=True, trail_high=None
  B: 포지션 없음 + BUY 주문 존재  → LADDER_ACTIVE, entry_price_base=최고가(1차 기준)
  C: 포지션 없음 + 주문 없음      → WATCHING
============================================================
"""

import time
import logging
import os
from decimal import Decimal, ROUND_DOWN
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
except Exception:
    Client = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception

ClientError = (BinanceAPIException, BinanceOrderException)

# ============================================================
# CFG
# ============================================================
CFG = {
    # ── 10번대: 심볼 / 시간축 ──────────────────────────────
    "SYMBOL":              "BTCUSDT",
    "INTERVAL_TRIGGER":    "5m",
    "INTERVAL_EXEC":       "5m",
    "INTERVAL_FILTER_HTF": "4h",

    # ── 20번대: 트리거 / HTF 필터 ─────────────────────────
    "EMA_TRIGGER_LEN":    15,
    "HTF_FILTER_EMA_LEN": 15,
    "HTF_FILTER_ENABLE":  True,

    # ── 30번대: 자본 / 레버리지 / 마진 ───────────────────
    "TOTAL_CAPITAL_USDT": 6000.0,
    "LEVERAGE":           3,
    "MARGIN_TYPE":        "CROSS",   # CROSS / ISOLATED
    "MAX_CAPITAL_RATIO":  0.95,

    # ── 40번대: 거미줄 구조 ───────────────────────────────
    "LADDER_COUNT":   10,
    "LADDER_GAP_PCT": 0.05,          # v10.2: 3% → 5%
    "SIZE_WEIGHTS": [
        0.5, 0.7, 1.0, 1.4, 1.8,    # v10.2: 중간 봉우리형
        1.4, 1.0, 0.8, 0.6, 0.5
    ],
    "LADDER_INVALIDATION_MULT":    2.0,   # 비활성화됨 (코드에서 차단)
    "LADDER_NO_FILL_TIMEOUT_BARS": 99999, # v10.3: INVALIDATION 우회 방지 — 사실상 제거

    # ── 50번대: TP / 트레일링 ─────────────────────────────
    "TP1_PROFIT_PCT":       0.01,
    "TP1_PARTIAL_RATIO":    0.5,
    "TRAILING_REBOUND_PCT": 0.005,

    # ── 60번대: EXIT 가격 구조 ────────────────────────────
    "FEE_PCT_ONEWAY":            0.0004,
    "TARGET_PROFIT_STAGE_1_3":   0.012,   # 1~3단: 1.2% 유지
    "TARGET_PROFIT_STAGE_4_5":   0.008,   # v10.2: 0.5% → 0.8%
    "TARGET_PROFIT_STAGE_6_7":   0.006,   # v10.2: 0.3% → 0.6%
    "TARGET_PROFIT_STAGE_8_9":   0.004,   # v10.2: 0.1% → 0.4%
    "TARGET_PROFIT_STAGE_10":    0.003,   # v10.2: -0.08% → +0.3%
    "EXIT_REPRICE_THRESHOLD_PCT": 0.006,   # v10.3: GAP 5% 구조 과민 방지

    # ── 70번대: 리스크 / 타임아웃 ────────────────────────
    "HARD_SL_PCT":             0.07,   # v10.2: 5% → 7% / 10단 체결 후에만 발동
    "DEEP_FILL_STAGE":         99,     # v10.2: TIMEOUT 사실상 제거
    "TIMEOUT_BARS_AFTER_DEEP": 99999,  # v10.2: TIMEOUT 사실상 제거

    # ── 80번대: 운영 / 루프 ───────────────────────────────
    "REENTRY_COOLDOWN_BARS":      8,
    "POLL_INTERVAL_SEC":          10,
    "BAR_CHECK_MIN_INTERVAL_SEC": 40,
    "LOG_LEVEL": "INFO",
}

# ============================================================
# 로거
# ============================================================
logging.basicConfig(
    level=getattr(logging, CFG["LOG_LEVEL"]),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vella_range_long_v10.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("VELLA_RL10")

# ============================================================
# 클라이언트
# ============================================================
API_KEY    = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

if Client is None:
    raise RuntimeError("python-binance missing")


class BinanceFuturesCompat:
    def __init__(self, key: str, secret: str):
        self._client = Client(key, secret)

    def exchange_info(self):
        return self._client.futures_exchange_info()

    def klines(self, symbol: str, interval: str, limit: int = 500):
        return self._client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

    def get_position_risk(self, symbol: str):
        return self._client.futures_position_information(symbol=symbol)

    def get_orders(self, symbol: str):
        return self._client.futures_get_open_orders(symbol=symbol)

    def cancel_order(self, symbol: str, orderId: int):
        return self._client.futures_cancel_order(symbol=symbol, orderId=orderId)

    def cancel_open_orders(self, symbol: str):
        return self._client.futures_cancel_all_open_orders(symbol=symbol)

    def query_order(self, symbol: str, orderId: int):
        return self._client.futures_get_order(symbol=symbol, orderId=orderId)

    def new_order(self, **kwargs):
        if "reduceOnly" in kwargs and isinstance(kwargs["reduceOnly"], str):
            kwargs["reduceOnly"] = kwargs["reduceOnly"].lower() == "true"
        return self._client.futures_create_order(**kwargs)

    def change_leverage(self, symbol: str, leverage: int):
        return self._client.futures_change_leverage(symbol=symbol, leverage=leverage)

    def change_margin_type(self, symbol: str, marginType: str):
        return self._client.futures_change_margin_type(symbol=symbol, marginType=marginType)

    def ticker_price(self, symbol: str):
        return self._client.futures_symbol_ticker(symbol=symbol)


client = BinanceFuturesCompat(API_KEY, API_SECRET)

# ============================================================
# 심볼 필터 캐시
# ============================================================
_SYM_FILTERS: dict = {}


def load_symbol_filters(symbol: str) -> dict:
    global _SYM_FILTERS
    if symbol in _SYM_FILTERS:
        return _SYM_FILTERS[symbol]
    info = client.exchange_info()
    for s in info["symbols"]:
        if s["symbol"] != symbol:
            continue
        result = {
            "price_prec":   s["pricePrecision"],
            "qty_prec":     s["quantityPrecision"],
            "tick_size":    None,
            "step_size":    None,
            "min_qty":      None,
            "min_notional": None,
        }
        for f in s["filters"]:
            ft = f["filterType"]
            if ft == "PRICE_FILTER":
                result["tick_size"] = f["tickSize"]
            elif ft == "LOT_SIZE":
                result["step_size"] = f["stepSize"]
                result["min_qty"]   = float(f["minQty"])
            elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                result["min_notional"] = float(f.get("notional", f.get("minNotional", 5.0)))
        _SYM_FILTERS[symbol] = result
        log.info(
            f"필터 로드: tick={result['tick_size']} step={result['step_size']} "
            f"minQty={result['min_qty']} minNotional={result['min_notional']}"
        )
        return result
    raise RuntimeError(f"심볼 {symbol} 필터 없음")

# ============================================================
# 수치 유틸
# ============================================================


def _quantize(value: float, unit_str: str, prec: int) -> str:
    d_val   = Decimal(str(value))
    d_unit  = Decimal(unit_str)
    floored = (d_val / d_unit).to_integral_value(rounding=ROUND_DOWN) * d_unit
    quant   = Decimal("0." + "0" * prec) if prec > 0 else Decimal("1")
    return str(floored.quantize(quant))


def fmt_price(price: float, sym: str) -> str:
    f = _SYM_FILTERS[sym]
    if f["tick_size"]:
        return _quantize(price, f["tick_size"], f["price_prec"])
    return f"{round(price, f['price_prec']):.{f['price_prec']}f}"


def fmt_qty(qty: float, sym: str) -> str:
    f = _SYM_FILTERS[sym]
    if f["step_size"]:
        return _quantize(qty, f["step_size"], f["qty_prec"])
    return f"{round(qty, f['qty_prec']):.{f['qty_prec']}f}"


def is_order_valid(price: float, qty: float, sym: str) -> bool:
    f = _SYM_FILTERS[sym]
    if f["min_qty"] and qty < f["min_qty"]:
        log.warning(f"주문 스킵: qty {qty} < minQty {f['min_qty']}")
        return False
    if f["min_notional"] and price * qty < f["min_notional"]:
        log.warning(f"주문 스킵: notional {price*qty:.2f} < minNotional {f['min_notional']}")
        return False
    return True

# ============================================================
# EMA
# ============================================================


def calc_ema(values: list, period: int) -> list:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    series = [e]
    for v in values[period:]:
        e = float(v) * k + e * (1 - k)
        series.append(e)
    return series

# ============================================================
# 캔들 조회
# ============================================================


def get_closed_bar_ts_with_closes(symbol: str, interval: str, limit: int = 60):
    raw    = client.klines(symbol, interval, limit=limit + 1)
    closed = raw[:-1]
    closes = [float(k[4]) for k in closed]
    ts     = int(closed[-1][0]) if closed else 0
    return closes, ts


def get_closed_bar_open_ts(symbol: str, interval: str) -> int:
    raw = client.klines(symbol, interval, limit=2)
    return int(raw[-2][0])

# ============================================================
# BarCache
# ============================================================


class BarCache:
    def __init__(self, min_interval_sec: float = 0):
        self._last_ts: int         = 0
        self._cached_result        = None
        self._last_api_time: float = 0.0
        self._min_interval         = min_interval_sec

    def query(self, fetch_fn, compute_fn):
        now = time.time()
        if self._cached_result is not None and \
                (now - self._last_api_time) < self._min_interval:
            return self._cached_result, self._last_ts
        data, ts            = fetch_fn()
        self._last_api_time = now
        if ts != self._last_ts or self._cached_result is None:
            self._cached_result = compute_fn(data)
            self._last_ts       = ts
        return self._cached_result, ts

# ============================================================
# 4시간 필터 (LONG: close > EMA15)
# ============================================================


def _compute_4h_filter(closes: list) -> bool:
    period = CFG["HTF_FILTER_EMA_LEN"]
    if len(closes) < period + 1:
        log.warning("HTF 데이터 부족 → 필터 차단")
        return False
    ema_s = calc_ema(closes, period)
    ok    = closes[-1] > ema_s[-1]
    label = "PASS" if ok else "BLOCK"
    log.info(f"[HTF FILTER {label}] 4H close {closes[-1]:.4f} {'>' if ok else '<='} EMA{period} {ema_s[-1]:.4f}")
    return ok


def check_4h_long_filter(symbol: str, cache: BarCache) -> bool:
    if not CFG["HTF_FILTER_ENABLE"]:
        return True
    period = CFG["HTF_FILTER_EMA_LEN"]

    def fetch():
        closes, ts = get_closed_bar_ts_with_closes(
            symbol, CFG["INTERVAL_FILTER_HTF"], limit=period + 10
        )
        return closes, ts

    result, _ = cache.query(fetch_fn=fetch, compute_fn=_compute_4h_filter)
    return result

# ============================================================
# 5M EMA15 상향 역전 트리거 (LONG)
# 조건:
#   cond1 = closes[-1] > ema15[-1]   — EMA15 상향 돌파
#   cond2 = lows[-2]   < ema15[-2]   — 직전봉 저가가 EMA15 아래
#   cond3 = closes[-1] > closes[-2]  — 상승 확정 1봉
# ============================================================


def _compute_5m_trigger(closes: list, lows: list) -> bool:
    period = CFG["EMA_TRIGGER_LEN"]
    if len(closes) < period + 2 or len(lows) < period + 2:
        return False
    ema_s = calc_ema(closes, period)
    cond1 = closes[-1] > ema_s[-1]
    cond2 = lows[-2]   < ema_s[-2]
    cond3 = closes[-1] > closes[-2]
    triggered = cond1 and cond2 and cond3
    if triggered:
        log.info(
            f"[5M TRIGGER] EMA15 상향 역전 확정: "
            f"close={closes[-1]:.4f}>ema={ema_s[-1]:.4f} | "
            f"low[-2]={lows[-2]:.4f}<ema[-2]={ema_s[-2]:.4f} | "
            f"close[-1]={closes[-1]:.4f}>close[-2]={closes[-2]:.4f}"
        )
    return triggered


def _fetch_5m_trigger_inputs(symbol: str, limit: int):
    raw    = client.klines(symbol, CFG["INTERVAL_TRIGGER"], limit=limit + 1)
    closed = raw[:-1]
    closes = [float(k[4]) for k in closed]
    lows   = [float(k[3]) for k in closed]
    ts     = int(closed[-1][0]) if closed else 0
    return closes, lows, ts


def calc_ema15_trigger(symbol: str, cache: BarCache) -> tuple[bool, int]:
    period = CFG["EMA_TRIGGER_LEN"]
    limit  = period + 10

    def fetch():
        closes, lows, ts = _fetch_5m_trigger_inputs(symbol, limit)
        return (closes, lows), ts

    def compute(data):
        closes, lows = data
        return _compute_5m_trigger(closes, lows)

    result, ts = cache.query(fetch_fn=fetch, compute_fn=compute)
    return result, ts

# ============================================================
# 포지션
# ============================================================


def get_position(symbol: str) -> dict:
    for p in client.get_position_risk(symbol=symbol):
        if p["symbol"] == symbol:
            return {"amt": float(p["positionAmt"]), "avg_price": float(p["entryPrice"])}
    return {"amt": 0.0, "avg_price": 0.0}


def has_long_position(pos: dict) -> bool:
    return pos["amt"] > 0.0001

# ============================================================
# 주문 유틸
# ============================================================


def get_open_orders(symbol: str) -> list:
    try:
        return client.get_orders(symbol=symbol)
    except ClientError as e:
        log.error(f"주문 조회 실패: {e}")
        return []


def cancel_order(symbol: str, order_id: int) -> bool:
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
        log.info(f"주문 취소: {order_id}")
        return True
    except ClientError as e:
        log.warning(f"주문 취소 실패 ({order_id}): {e}")
        return False


def cancel_all_orders(symbol: str):
    try:
        client.cancel_open_orders(symbol=symbol)
        log.info("미체결 전체 취소")
    except ClientError as e:
        log.warning(f"전체 취소 실패: {e}")


def query_order_status(symbol: str, order_id: int) -> str:
    try:
        return client.query_order(symbol=symbol, orderId=order_id).get("status", "UNKNOWN")
    except ClientError as e:
        log.warning(f"query_order 실패 ({order_id}): {e}")
        return "UNKNOWN"


def place_limit_long(symbol: str, price: float, qty: float) -> dict | None:
    if not is_order_valid(price, qty, symbol):
        return None
    try:
        order = client.new_order(
            symbol=symbol, side="BUY", type="LIMIT", timeInForce="GTC",
            price=fmt_price(price, symbol), quantity=fmt_qty(qty, symbol),
        )
        log.info(f"롱 지정가: {fmt_price(price, symbol)} × {fmt_qty(qty, symbol)}")
        return order
    except ClientError as e:
        log.error(f"롱 주문 실패: {e}")
        return None


def place_market_long(symbol: str, qty: float) -> dict | None:
    q_str = fmt_qty(abs(qty), symbol)
    if float(q_str) <= 0:
        log.warning(f"시장가 롱 스킵: qty={q_str}")
        return None
    try:
        order = client.new_order(
            symbol=symbol, side="BUY", type="MARKET",
            quantity=q_str,
        )
        log.info(f"시장가 롱 진입: {q_str}")
        return order
    except ClientError as e:
        log.error(f"시장가 롱 실패: {e}")
        return None


def place_limit_exit(symbol: str, price: float, qty: float) -> dict | None:
    if not is_order_valid(price, qty, symbol):
        return None
    try:
        order = client.new_order(
            symbol=symbol, side="SELL", type="LIMIT", timeInForce="GTC",
            price=fmt_price(price, symbol), quantity=fmt_qty(qty, symbol),
            reduceOnly="true",
        )
        log.info(f"청산 지정가: {fmt_price(price, symbol)} × {fmt_qty(qty, symbol)}")
        return order
    except ClientError as e:
        log.error(f"청산 주문 실패: {e}")
        return None


def market_close_long(symbol: str, qty: float) -> bool:
    q_str = fmt_qty(abs(qty), symbol)
    if float(q_str) <= 0:
        log.warning(f"시장가 청산 스킵: qty={q_str}")
        return False
    try:
        client.new_order(
            symbol=symbol, side="SELL", type="MARKET",
            quantity=q_str, reduceOnly="true",
        )
        log.info(f"시장가 청산: {q_str}")
        return True
    except ClientError as e:
        log.error(f"시장가 청산 실패: {e}")
        return False


def set_leverage(symbol: str, leverage: int):
    try:
        client.change_leverage(symbol=symbol, leverage=leverage)
        log.info(f"레버리지 {leverage}x 설정")
    except ClientError as e:
        log.warning(f"레버리지 설정 오류: {e}")


def set_margin_type(symbol: str, margin_type: str):
    try:
        mt = "CROSSED" if margin_type.upper() == "CROSS" else "ISOLATED"
        client.change_margin_type(symbol=symbol, marginType=mt)
        log.info(f"마진 타입 {mt} 설정 완료")
    except ClientError as e:
        msg = str(e)
        if "No need to change margin type" in msg:
            log.info(f"마진 타입 이미 {mt} 상태")
        elif "cannot be changed if there exists position" in msg or "cannot be changed if open orders exist" in msg:
            log.warning(f"마진 타입 변경 불가: 포지션/주문 존재 | 요청={mt} | {e}")
        else:
            log.error(f"마진 타입 설정 실패: 요청={mt} | {e}")
            raise

# ============================================================
# 사이즈 / 가격 계산
# ============================================================


def normalize_weights(weights: list, count: int) -> list:
    w = weights[:count]
    t = sum(w)
    return [x / t for x in w]


def build_ladder_prices(entry_price: float, count: int, gap_pct: float) -> list:
    # 롱: 1차는 현재가, 2~10차는 아래로 내려가는 지정가
    return [entry_price * (1 - gap_pct * i) for i in range(count)]


def calc_ladder_quantities(total_capital: float, leverage: float,
                           weights: list, entry_price: float) -> list:
    effective = total_capital * CFG["MAX_CAPITAL_RATIO"] * leverage
    return [effective * w / entry_price for w in weights]


def get_stage_target_pct(stage: int) -> float:
    if stage <= 3: return CFG["TARGET_PROFIT_STAGE_1_3"]
    if stage <= 5: return CFG["TARGET_PROFIT_STAGE_4_5"]
    if stage <= 7: return CFG["TARGET_PROFIT_STAGE_6_7"]
    if stage <= 9: return CFG["TARGET_PROFIT_STAGE_8_9"]
    return CFG["TARGET_PROFIT_STAGE_10"]


def calc_exit_price(avg_price: float, stage: int) -> float:
    # 롱: 위로 청산
    return avg_price * (1 + CFG["FEE_PCT_ONEWAY"] * 2 + get_stage_target_pct(stage))

# ============================================================
# 5분 완료봉 감지
# ============================================================


class BarTracker:
    def __init__(self, symbol: str, interval: str):
        self.symbol        = symbol
        self.interval      = interval
        self.last_ts       = None
        self._cached_ts    = None
        self._last_checked = 0.0

    def new_bar_closed(self) -> bool:
        now = time.time()
        if now - self._last_checked >= CFG["BAR_CHECK_MIN_INTERVAL_SEC"]:
            self._cached_ts    = get_closed_bar_open_ts(self.symbol, self.interval)
            self._last_checked = now
        ts = self._cached_ts
        if ts is None:
            return False
        if self.last_ts is None:
            self.last_ts = ts
            return False
        if ts > self.last_ts:
            self.last_ts = ts
            return True
        return False

# ============================================================
# 상태 머신
# ============================================================


class RangeLongEngine:
    def __init__(self):
        self.state  = "WATCHING"
        self.symbol = CFG["SYMBOL"]

        self.ladder_orders: list[dict] = []
        self.entry_price_base = None

        self.max_filled_stage = 0
        self.exit_order_ids: list[int] = []
        self.last_exit_qty   = 0.0
        self.last_exit_price = 0.0
        self.last_stage      = 0

        self.tp1_done:   bool         = False
        self.trail_high: float | None = None

        self._filled_order_ids:   set[int] = set()
        self._canceled_order_ids: set[int] = set()
        self._last_position_amt            = 0.0

        self._closing_in_progress: bool = False
        self._last_filled_check_ts: int  = 0

        self.bars_after_deep  = 0
        self.cooldown_bars    = 0
        self.no_fill_bars     = 0

        self.last_trigger_bar_ts: int = 0

        self.bar_tracker = BarTracker(self.symbol, CFG["INTERVAL_EXEC"])

        min_iv = CFG["BAR_CHECK_MIN_INTERVAL_SEC"]
        self._htf_cache     = BarCache(min_interval_sec=min_iv)
        self._trigger_cache = BarCache(min_interval_sec=min_iv)

        load_symbol_filters(self.symbol)
        set_margin_type(self.symbol, CFG["MARGIN_TYPE"])
        set_leverage(self.symbol, CFG["LEVERAGE"])

    # --------------------------------------------------------
    # 안전 취소
    # --------------------------------------------------------
    def _safe_cancel(self, order_id: int):
        if order_id in self._filled_order_ids:
            return
        if order_id in self._canceled_order_ids:
            return
        success = cancel_order(self.symbol, order_id)
        if success:
            self._canceled_order_ids.add(order_id)

    def _cancel_ladder_orders(self):
        for o in self.ladder_orders:
            self._safe_cancel(o["order_id"])

    def cancel_sell_exit_orders(self, exit_order_ids: list):
        for oid in exit_order_ids:
            self._safe_cancel(oid)

    # --------------------------------------------------------
    # FILLED 캐시 기반 체결 단계 카운트
    # --------------------------------------------------------
    def _count_filled_stages(self) -> int:
        for o in self.ladder_orders:
            oid = o["order_id"]
            if oid in self._filled_order_ids:
                continue
            if query_order_status(self.symbol, oid) == "FILLED":
                self._filled_order_ids.add(oid)
        return sum(1 for o in self.ladder_orders
                   if o["order_id"] in self._filled_order_ids)

    # --------------------------------------------------------
    # 재시작 동기화
    # --------------------------------------------------------
    def _sync_on_start(self):
        pos         = get_position(self.symbol)
        open_orders = get_open_orders(self.symbol)
        buy_orders  = [o for o in open_orders if o["side"] == "BUY" and o["status"] == "NEW"]
        buy_sorted  = sorted(buy_orders, key=lambda x: float(x["price"]), reverse=True)

        if has_long_position(pos):
            log.info("[SYNC] 롱 포지션 감지 → POSITION_HOLD 복구")
            self.state = "POSITION_HOLD"

            for i, o in enumerate(buy_sorted):
                self.ladder_orders.append({
                    "stage":    i + 1,
                    "order_id": int(o["orderId"]),
                    "price":    float(o["price"]),
                    "qty":      float(o["origQty"]),
                })
            self.entry_price_base   = pos["avg_price"]
            self._last_position_amt = pos["amt"]

            sell_orders = [o for o in open_orders if o["side"] == "SELL" and o["status"] == "NEW"]
            self.exit_order_ids = [int(o["orderId"]) for o in sell_orders]

            self.max_filled_stage = self._count_filled_stages()
            self.last_stage       = self.max_filled_stage

            self.tp1_done   = True
            self.trail_high = None
            log.info(
                f"[SYNC] 복구 완료 | avg={pos['avg_price']} | "
                f"BUY {len(buy_sorted)}개 | SELL exit {len(sell_orders)}개 | "
                f"max_filled_stage={self.max_filled_stage} | "
                f"tp1_done=True(보수적) trail_high=None"
            )

        elif buy_sorted:
            log.info("[SYNC] 포지션 없음 + BUY 주문 존재 → LADDER_ACTIVE 복구")
            self.state = "LADDER_ACTIVE"
            for i, o in enumerate(buy_sorted):
                self.ladder_orders.append({
                    "stage":    i + 1,
                    "order_id": int(o["orderId"]),
                    "price":    float(o["price"]),
                    "qty":      float(o["origQty"]),
                })
            self.entry_price_base = float(buy_sorted[0]["price"])
            log.info(f"[SYNC] entry_price_base = {self.entry_price_base:.4f} (max BUY price)")

        else:
            log.info("[SYNC] 포지션 없음 + 주문 없음 → WATCHING 시작")
            self.state = "WATCHING"

    # --------------------------------------------------------
    # 메인 루프
    # --------------------------------------------------------
    def run(self):
        log.info("=" * 60)
        log.info("VELLA RANGE LONG LADDER v10.7 시작")
        log.info(f"심볼: {self.symbol} | 자본: {CFG['TOTAL_CAPITAL_USDT']} USDT | 레버: {CFG['LEVERAGE']}x")
        log.info(f"GAP: {CFG['LADDER_GAP_PCT']*100:.0f}% | HARD_SL: {CFG['HARD_SL_PCT']*100:.0f}%(10단 후)")
        log.info("=" * 60)
        self._sync_on_start()
        while True:
            try:
                self._tick()
            except Exception as e:
                log.error(f"루프 오류: {e}", exc_info=True)
            time.sleep(CFG["POLL_INTERVAL_SEC"])

    # --------------------------------------------------------
    # 틱
    # --------------------------------------------------------
    def _tick(self):
        symbol = self.symbol
        ticker = client.ticker_price(symbol=symbol)
        current_price = float(ticker["price"])

        pos     = get_position(symbol)
        has_pos = has_long_position(pos)
        new_bar = self.bar_tracker.new_bar_closed()

        # ── COOLDOWN ──
        if self.state == "COOLDOWN":
            if new_bar:
                self.cooldown_bars -= 1
                log.info(f"쿨다운: 남은 봉 {self.cooldown_bars}")
            if self.cooldown_bars <= 0:
                self.state = "WATCHING"
                log.info("쿨다운 종료 → WATCHING")
            return

        # ── WATCHING ──
        if self.state == "WATCHING":
            if has_pos:
                log.warning("외부 포지션 감지 → POSITION_HOLD")
                self.state = "POSITION_HOLD"
                return

            if not check_4h_long_filter(symbol, self._htf_cache):
                return

            triggered, bar_ts = calc_ema15_trigger(symbol, self._trigger_cache)

            if triggered and bar_ts == self.last_trigger_bar_ts:
                log.debug(f"동일 5M 봉 재트리거 차단: ts={bar_ts}")
                return

            if triggered:
                self.last_trigger_bar_ts = bar_ts
                self._deploy_ladder(current_price)
            return

        # ── LADDER_ACTIVE ──
        if self.state == "LADDER_ACTIVE":
            if has_pos:
                log.info("포지션 체결 감지 → POSITION_HOLD")
                self.state              = "POSITION_HOLD"
                self.bars_after_deep    = 0
                self.no_fill_bars       = 0
                self._last_position_amt = pos["amt"]
                return

            if new_bar:
                self.no_fill_bars += 1
                log.info(f"거미줄 미체결 대기: {self.no_fill_bars}/{CFG['LADDER_NO_FILL_TIMEOUT_BARS']}봉")
            if self.no_fill_bars >= CFG["LADDER_NO_FILL_TIMEOUT_BARS"]:
                log.warning(f"거미줄 미체결 타임아웃 ({self.no_fill_bars}봉) → 철거 후 WATCHING")
                self._cancel_ladder_orders()
                self._reset_ladder()
                self.state = "WATCHING"
                return

            # v10.2: LADDER INVALIDATION 완전 비활성화
            # (거미줄 작전 철학 — 10단 끝까지 버팀)

            log.info(f"거미줄 대기 | 현재가: {current_price:.4f}")
            return

        # ── POSITION_HOLD ──
        if self.state == "POSITION_HOLD":
            if not has_pos:
                log.info("포지션 청산 감지 → 쿨다운")
                self.cancel_sell_exit_orders(self.exit_order_ids)
                self.exit_order_ids = []
                self._cancel_ladder_orders()
                self._start_cooldown()
                return

            avg_price    = pos["avg_price"]
            position_qty = pos["amt"]

            amt_changed = abs(position_qty - self._last_position_amt) > 0.0001
            cur_bar_ts  = self.bar_tracker.last_ts or 0
            need_check  = (
                amt_changed
                or (new_bar and cur_bar_ts != self._last_filled_check_ts)
                or self.max_filled_stage == 0
            )
            if need_check:
                filled = self._count_filled_stages()
                if filled > self.max_filled_stage:
                    log.info(f"체결 단계 갱신: {self.max_filled_stage} → {filled}")
                    self.max_filled_stage = filled
                self._last_position_amt    = position_qty
                self._last_filled_check_ts = cur_bar_ts

            log.info(
                f"HOLD | avg={avg_price:.4f} | price={current_price:.4f} | "
                f"stage={self.max_filled_stage} | qty={position_qty:.4f} | "
                f"tp1={self.tp1_done} | trail_high={self.trail_high} | "
                f"closing={self._closing_in_progress}"
            )

            pnl_pct = (current_price - avg_price) / avg_price

            # 1. HARD SL — v10.2: 10단 체결 완료 후에만 발동
            if (self.max_filled_stage >= CFG["LADDER_COUNT"]
                    and pnl_pct < -CFG["HARD_SL_PCT"]):
                log.warning(
                    f"HARD SL 발동 | 10단 완료 후 avg 기준 손실 {pnl_pct*100:.2f}%"
                )
                self._final_close(symbol, position_qty, "HARD_SL")
                return

            # 2. TIMEOUT — v10.2: 사실상 비활성 (DEEP_FILL_STAGE=99)
            if self.max_filled_stage >= CFG["DEEP_FILL_STAGE"]:
                if new_bar:
                    self.bars_after_deep += 1
                if self.bars_after_deep >= CFG["TIMEOUT_BARS_AFTER_DEEP"]:
                    log.warning(f"TIMEOUT 발동 | {self.bars_after_deep}봉")
                    self._final_close(symbol, position_qty, "TIMEOUT")
                    return

            # 3. TP1
            if not self.tp1_done and pnl_pct >= CFG["TP1_PROFIT_PCT"]:
                self._handle_tp1(symbol, position_qty, current_price)
                return

            # 4. 트레일링
            if self.tp1_done:
                if self.trail_high is None:
                    self.trail_high = current_price
                    log.info(f"trail_high 초기화: {self.trail_high:.4f}")

                self.trail_high = max(self.trail_high, current_price)

                if current_price <= self.trail_high * (1 - CFG["TRAILING_REBOUND_PCT"]):
                    log.info(
                        f"[TRAIL EXIT] 고점={self.trail_high:.4f} 대비 -0.5% 반락 "
                        f"(current={current_price:.4f})"
                    )
                    self._final_close(symbol, position_qty, "TRAIL")
                return

            # 5. 지정가 EXIT 동기화
            # v10.5: 2단 이상 체결 후에만 EXIT 생성
            # → 1단 체결 직후 조기 exit 배치 방지
            if not self._closing_in_progress and self.max_filled_stage >= 2:
                self._sync_exit_order(symbol, avg_price, position_qty)

    # --------------------------------------------------------
    # TP1 처리
    # --------------------------------------------------------
    def _handle_tp1(self, symbol: str, position_qty: float, current_price: float):
        partial_qty = abs(position_qty) * CFG["TP1_PARTIAL_RATIO"]
        log.info(f"[TP1] 수익 도달 → 50% 부분청산 시도 qty={partial_qty:.4f}")

        success = market_close_long(symbol, partial_qty)

        if success:
            time.sleep(0.2)
            pos = get_position(symbol)

            self.cancel_sell_exit_orders(self.exit_order_ids)
            self.exit_order_ids = []

            self._cancel_ladder_orders()
            self.ladder_orders     = []
            self._filled_order_ids = set()
            # v10.3: max_filled_stage 유지 — TP1 후에도 stage 기반 EXIT 정확도 보존
            # (0으로 리셋하면 이후 EXIT가 항상 1단 기준으로 계산되는 버그 방지)

            self._last_position_amt = pos["amt"]
            self.tp1_done   = True
            self.trail_high = None
            log.info(
                f"[TP1] 부분청산 성공 → tp1_done=True | "
                f"잔량={pos['amt']:.4f} | stage={self.max_filled_stage} 유지 | trail_high=None"
            )
        else:
            log.error("[TP1] 부분청산 실패 → 기존 주문 유지, 다음 tick 재시도")

    # --------------------------------------------------------
    # 공용 종료 헬퍼
    # --------------------------------------------------------
    def _final_close(self, symbol: str, position_qty: float, reason: str):
        log.info(f"[FINAL CLOSE] 사유={reason} | qty={position_qty:.4f}")
        self._closing_in_progress = True

        self.cancel_sell_exit_orders(self.exit_order_ids)
        self.exit_order_ids = []

        self._cancel_ladder_orders()

        success = market_close_long(symbol, abs(position_qty))

        if success:
            self._closing_in_progress = False
            self._start_cooldown()
        else:
            log.error(
                f"[FINAL CLOSE] 청산 실패 → POSITION_HOLD 유지, 다음 tick 재시도 "
                f"(사유={reason})"
            )

    # --------------------------------------------------------
    # 거미줄 배치 — 1차 시장가 BUY, 2~10차 하단 LIMIT BUY
    # --------------------------------------------------------
    def _deploy_ladder(self, current_price: float):
        # v10.4: 중복 거미줄 차단 — WATCHING 상태에서만 진입 허용
        if self.state != "WATCHING":
            log.warning(f"_deploy_ladder 차단: state={self.state} (WATCHING 아님)")
            return

        symbol  = self.symbol
        count   = CFG["LADDER_COUNT"]
        gap     = CFG["LADDER_GAP_PCT"]
        weights = normalize_weights(CFG["SIZE_WEIGHTS"], count)
        prices  = build_ladder_prices(current_price, count, gap)
        qtys    = calc_ladder_quantities(
            CFG["TOTAL_CAPITAL_USDT"], CFG["LEVERAGE"], weights, current_price
        )

        cancel_all_orders(symbol)
        self._reset_ladder()
        self.entry_price_base = current_price

        log.info(f"거미줄 배치 | 기준가: {current_price:.4f} | {count}단계 | GAP={gap*100:.0f}%")
        success = 0

        # 1차: 시장가 즉시 진입
        order_1st = place_market_long(symbol, qtys[0])
        if order_1st:
            self.ladder_orders.append({
                "stage":    1,
                "order_id": int(order_1st["orderId"]),
                "price":    current_price,
                "qty":      qtys[0],
            })
            self._filled_order_ids.add(int(order_1st["orderId"]))
            self.max_filled_stage = 1
            success += 1
            log.info(f"1차 시장가 진입 완료: qty={fmt_qty(qtys[0], symbol)}")
        else:
            log.error("1차 시장가 진입 실패")

        # 2~10차: 하단 지정가 거미줄
        for i in range(1, count):
            order = place_limit_long(symbol, prices[i], qtys[i])
            if order:
                self.ladder_orders.append({
                    "stage":    i + 1,
                    "order_id": int(order["orderId"]),
                    "price":    prices[i],
                    "qty":      qtys[i],
                })
                success += 1
            time.sleep(0.15)

        if success == 0:
            log.error("거미줄 주문 0개 성공 → WATCHING 복귀")
            self.state = "WATCHING"
        else:
            log.info(f"거미줄 배치 완료: {success}/{count}개 → LADDER_ACTIVE")
            self.no_fill_bars = 0
            # v10.6: 1차 체결 여부 무관하게 LADDER_ACTIVE 고정
            # → POSITION_HOLD 직행 금지 / 반드시 LADDER_ACTIVE 거침
            self.state = "LADDER_ACTIVE"

    # --------------------------------------------------------
    # 지정가 EXIT 동기화
    # --------------------------------------------------------
    def _sync_exit_order(self, symbol: str, avg_price: float, position_qty: float):
        stage      = max(self.max_filled_stage, 1)
        exit_price = calc_exit_price(avg_price, stage)
        exit_qty   = abs(position_qty)
        threshold  = CFG["EXIT_REPRICE_THRESHOLD_PCT"]

        need_replace = (
            not self.exit_order_ids
            or stage != self.last_stage
            or (self.last_exit_price > 0
                and abs(exit_price - self.last_exit_price) > self.last_exit_price * threshold)
            or (self.last_exit_qty > 0
                and abs(exit_qty - self.last_exit_qty) > self.last_exit_qty * 0.05)
        )

        if not need_replace:
            return

        self.cancel_sell_exit_orders(self.exit_order_ids)
        self.exit_order_ids = []
        self.last_stage     = -1

        order = place_limit_exit(symbol, exit_price, exit_qty)
        if order:
            self.exit_order_ids  = [int(order["orderId"])]
            self.last_exit_price = exit_price
            self.last_exit_qty   = exit_qty
            self.last_stage      = stage
            log.info(
                f"청산 주문 동기화 | stage={stage} | "
                f"청산가={exit_price:.4f} | qty={exit_qty:.4f}"
            )

    # --------------------------------------------------------
    # 내부 리셋
    # --------------------------------------------------------
    def _reset_ladder(self):
        self.ladder_orders          = []
        self.entry_price_base       = None
        self.max_filled_stage       = 0
        self.exit_order_ids         = []
        self.last_exit_qty          = 0.0
        self.last_exit_price        = 0.0
        self.bars_after_deep        = 0
        self.no_fill_bars           = 0
        self.last_stage             = 0
        self._filled_order_ids      = set()
        self._canceled_order_ids    = set()
        self._last_position_amt     = 0.0
        self._closing_in_progress   = False
        self._last_filled_check_ts  = 0
        self.tp1_done               = False
        self.trail_high             = None

    def _start_cooldown(self):
        self._reset_ladder()
        self.state         = "COOLDOWN"
        self.cooldown_bars = CFG["REENTRY_COOLDOWN_BARS"]
        log.info(f"쿨다운 시작: {self.cooldown_bars}봉 (5m 기준)")


# ============================================================
# 엔트리포인트
# ============================================================
if __name__ == "__main__":
    engine = RangeLongEngine()
    engine.run()