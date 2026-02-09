# ============================================================
# VELLA_v10 — LONG ONLY AUTO TRADING
# AWS + BINANCE FUTURES
# 구조 : ema 5 10 15 + vwap (추세 추종)
# ============================================================

import os
import sys
import time
import signal
import requests
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

# ============================================================
# CFG
# ============================================================
# 2026_0209_1600 : 기본 ema 5 > 15 엔트리 5 > 10 엑시트 조건

CFG = {
    # BASIC
    "TRADE_SYMBOL": "RAREUSDT",
    "CAPITAL_USDT": 30,
    "ENGINE_ENABLE": True,
    "LEVERAGE": 1,

    #=======================================================
    # STAGE 1 [01~19] : PRE-ENTRY FILTER
    #========================================================
    "01_BTC_DAILY_ENABLE": False,
    "02_BTC_1H_ENABLE": False,
    "03_SYMBOL_1H_ENABLE": False,
    "04_SHOCK_ENABLE": False,
    "14_SHOCK_PCT": 99.0,
    "05_VOLUME_ENABLE": False,
    "15_VOLUME_K": 0.0,
    "06_SPREAD_ENABLE": False,
    "16_SPREAD_MAX": 99.0,
    "07_DEAD_MARKET_ENABLE": False,
    "17_ATR_MIN_PCT": 0.0,
    "08_MATH_GUARD_ENABLE": True,
    "09_FUNDING_ENABLE": False,
    
    #===================================================
    # [20~21] STAGE 2: ENTRY
    "21_ENTRY_EMA_FAST": 5,
    "22_ENTRY_EMA_SLOW": 15,

    #===================================================
    # STAGE 3 [30~49] : ENTRY MANAGEMENT
    #=====================================================
    "30_IK_VWAP_ENABLE": False,
    "31_IK_EMA_ENABLE": False,
    "32_IK_VOLUME_ENABLE": False,
    "33_BIG_EMA_ENABLE": False,
    "33_GRACE_BARS": 0,
    "34_BIG_DONCHIAN_ENABLE": False,
    "34_LOOKBACK": 1,
    "35_SMALL_PULLBACK_ENABLE": False,
    "36_SMALL_VOL_ENABLE": False,
    "37_SMALL_ESCGO_ENABLE": False,
    "38_SURVIVAL_ENABLE": False,

    #===================================================
    # STAGE 4 [50~69] : EXIT GATE
    #======================================================
    "50_SL_PCT": 5.0,
    "51_VWAP_BREAK_ENABLE": False,
    "54_SLOPE_LOOKBACK": 2,
    "54_SLOPE_EXIT_PCT": 999.0,
    "54_SLOPE_EMA": 5,
    "56_EXIT_EMA_FAST": 5,
    "56_EXIT_EMA_SLOW": 10,
    "57_DONCHIAN_LOW_ENABLE": False,
    "57_DONCHIAN_LOW_LOOKBACK": 10,
    "58_EXIT_AVG_ENABLE": False,
    "58_EXIT_AVG_N": 10,
}

# ============================================================
# UTIL
# ============================================================
def q(x, p=6):
    """Display-only formatting. Never use in calculation/logic."""
    return float(Decimal(str(x)).quantize(Decimal("1." + "0" * p), rounding=ROUND_DOWN))

def now_ts():
    return int(time.time())

def log(stage, priority, name, result):
    print(f"[{now_ts()}] S{stage} P{priority} {name}: {result}")

# ============================================================
# BINANCE CLIENT
# ============================================================
def init_binance_client():
    k = os.getenv("BINANCE_API_KEY")
    s = os.getenv("BINANCE_API_SECRET")
    if not k or not s:
        raise RuntimeError("API KEY missing")
    return Client(k, s)

def setup_futures_env(client, symbol, leverage):
    try:
        client.futures_change_position_mode(dualSidePosition=False)
        print("Position mode set to ONE_WAY")
    except Exception as e:
        print(f"Position mode already ONE_WAY or error: {e}")
    
    try:
        client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        print(f"Margin type set to ISOLATED for {symbol}")
    except Exception as e:
        print(f"Margin type already ISOLATED or error: {e}")
    
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        print(f"Leverage setup error: {e}")

class FX:
    def __init__(self, client, symbol):
        self.client = client
        self.symbol = symbol
        info = client.futures_exchange_info()
        sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
        lot = next(f for f in sym["filters"] if f["filterType"] == "LOT_SIZE")
        self.step = Decimal(lot["stepSize"])
        self.minq = Decimal(lot["minQty"])
        
        min_notional_filter = next((f for f in sym["filters"] if f["filterType"] == "MIN_NOTIONAL"), None)
        self.min_notional = Decimal(min_notional_filter["notional"]) if min_notional_filter else Decimal("0")
        print(f"MIN_NOTIONAL: {self.min_notional}")
    
    def _norm_qty(self, qty):
        qd = (Decimal(str(qty)) / self.step).to_integral_value(rounding=ROUND_DOWN) * self.step
        if qd < self.minq:
            return None
        d = len(str(self.step).split(".")[1].rstrip("0"))
        return f"{qd:.{d}f}"
    
    def order(self, side, qty, price=None, reduce_only=False):
        qs = self._norm_qty(qty)
        if qs is None:
            print(f"Order skipped: qty {qty} below minQty {self.minq}")
            return 0.0
        
        if price and Decimal(qs) * Decimal(str(price)) < self.min_notional:
            print(f"Order skipped: notional {Decimal(qs) * Decimal(str(price))} below MIN_NOTIONAL {self.min_notional}")
            return 0.0
        
        try:
            self.client.futures_create_order(
                symbol=self.symbol,
                side=SIDE_SELL if side == "SELL" else SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=qs,
                reduceOnly=reduce_only,
                recvWindow=5000
            )
            return float(qs)
        except Exception as e:
            print(f"Order failed: {e}")
            return 0.0

# ============================================================
# MARKET DATA
# ============================================================
BINANCE_API = "https://fapi.binance.com/fapi/v1/klines"

def calc_ema(prices, period):
    if len(prices) < period:
        return None
    alpha = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * alpha + ema * (1 - alpha)
    return ema

def fetch_klines(symbol, interval, limit):
    try:
        resp = requests.get(
            BINANCE_API,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=5
        )
        data = resp.json()
        
        if not isinstance(data, list):
            print(f"fetch_klines invalid response ({symbol} {interval}): {data}")
            return None
        
        return data
    
    except Exception as e:
        print(f"fetch_klines exception ({symbol} {interval}): {e}")
        return None

def collect_data(symbol):
    klines_15m = fetch_klines(symbol, "15m", 50)
    klines_1h = fetch_klines(symbol, "1h", 30)
    klines_1m = fetch_klines(symbol, "1m", 5)
    btc_klines_daily = fetch_klines("BTCUSDT", "1d", 2)
    btc_klines_1h = fetch_klines("BTCUSDT", "1h", 30)
    
    if klines_15m is None or klines_1h is None or klines_1m is None or btc_klines_daily is None or btc_klines_1h is None:
        return None
    
    k_current = klines_15m[-2]
    k_prev = klines_15m[-3]
    
    closes_15m = [float(k[4]) for k in klines_15m[:-1]]
    highs_15m = [float(k[2]) for k in klines_15m[:-1]]
    lows_15m = [float(k[3]) for k in klines_15m[:-1]]
    volumes_15m = [float(k[5]) for k in klines_15m[:-1]]
    
    closes_1h = [float(k[4]) for k in klines_1h[:-1]]
    closes_1m = [float(k[4]) for k in klines_1m[:-1]]
    btc_closes_1h = [float(k[4]) for k in btc_klines_1h[:-1]]
    
    current_open = float(k_current[1])
    current_high = float(k_current[2])
    current_low = float(k_current[3])
    current_close = float(k_current[4])
    current_volume = float(k_current[5])
    bar_ts = int(k_current[6])
    
    prev_close = float(k_prev[4])
    
    ema_periods = set()
    
    ema_periods.add(CFG["21_ENTRY_EMA_FAST"])
    ema_periods.add(CFG["22_ENTRY_EMA_SLOW"])
    
    if CFG.get("31_IK_EMA_ENABLE"):
        ema_periods.add(CFG["41_IK_EMA_FAST"])
        ema_periods.add(CFG["42_IK_EMA_SLOW"])
    
    if CFG.get("33_BIG_EMA_ENABLE"):
        ema_periods.add(CFG["43_BIG_EMA_FAST"])
        ema_periods.add(CFG["44_BIG_EMA_SLOW"])
    
    ema_periods.add(CFG["54_SLOPE_EMA"])
    ema_periods.add(CFG["56_EXIT_EMA_FAST"])
    ema_periods.add(CFG["56_EXIT_EMA_SLOW"])
    
    ema_values = {}
    ema_values_prev = {}
    for period in ema_periods:
        if period <= len(closes_15m):
            ema_values[period] = calc_ema(closes_15m[-period:] + [current_close], period)
            ema_values_prev[period] = calc_ema(closes_15m[-period-1:-1] + [prev_close], period)
    
    cum_pv = sum(float(k[4]) * float(k[5]) for k in klines_15m[-30:-1])
    cum_v = sum(float(k[5]) for k in klines_15m[-30:-1])
    vwap = cum_pv / cum_v if cum_v > 0 else current_close
    
    volume_avg = sum(volumes_15m[-20:]) / 20
    
    trs = []
    for i in range(1, min(16, len(closes_15m))):
        h = highs_15m[-i]
        l = lows_15m[-i]
        prev_c = closes_15m[-i-1] if i < len(closes_15m) else closes_15m[-i]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    atr15 = sum(trs) / len(trs) if trs else 0
    
    donchian_high = max(highs_15m[-CFG["34_LOOKBACK"]:]) if len(highs_15m) >= CFG["34_LOOKBACK"] else current_high
    donchian_low = min(lows_15m[-CFG["57_DONCHIAN_LOW_LOOKBACK"]:]) if len(lows_15m) >= CFG["57_DONCHIAN_LOW_LOOKBACK"] else current_low
    
    slope_ema_period = CFG["54_SLOPE_EMA"]
    slope_ema_hist = []
    for i in range(slope_ema_period, min(50, len(closes_15m))):
        slope_ema_hist.append(calc_ema(closes_15m[i-slope_ema_period:i], slope_ema_period))
    
    avg_close_n = sum(closes_15m[-CFG["58_EXIT_AVG_N"]:]) / CFG["58_EXIT_AVG_N"] if len(closes_15m) >= CFG["58_EXIT_AVG_N"] else current_close
    
    btc_daily_open = float(btc_klines_daily[-2][1])
    btc_price = float(btc_klines_daily[-1][4])
    btc_ema20_1h = calc_ema(btc_closes_1h[-20:], 20)
    
    symbol_ema20_1h = calc_ema(closes_1h[-20:], 20)
    
    price_1m_prev = closes_1m[-2] if len(closes_1m) >= 2 else current_close
    
    return {
        "bar_ts": bar_ts,
        "open": current_open,
        "high": current_high,
        "low": current_low,
        "close": current_close,
        "volume": current_volume,
        "ema_values": ema_values,
        "ema_values_prev": ema_values_prev,
        "vwap": vwap,
        "volume_avg": volume_avg,
        "atr15": atr15,
        "donchian_high": donchian_high,
        "donchian_low": donchian_low,
        "avg_close_n": avg_close_n,
        "slope_ema_hist": slope_ema_hist,
        "btc_daily_open": btc_daily_open,
        "btc_price": btc_price,
        "btc_ema20_1h": btc_ema20_1h,
        "symbol_ema20_1h": symbol_ema20_1h,
        "price_1m_prev": price_1m_prev,
    }

# ============================================================
# CONTEXT (STAGE 간 전달 전용)
# ============================================================
class Context:
    def __init__(self):
        self.position = None
        self.entry_price = None
        self.entry_volume_ref = None
        self.position_qty = 0.0
        self.entry_bar_ts = None
        self.big_ema_state = "NONE"
        self.big_ema_entry_bar_ts = None
        self.big_ema_bar_count = 0

# ============================================================
# STAGE 1 — PRE-ENTRY FILTER
# ============================================================
def s1_p1_btc_daily(data):
    if not CFG["01_BTC_DAILY_ENABLE"]:
        return True
    btc_return = (data["btc_price"] - data["btc_daily_open"]) / data["btc_daily_open"] * 100
    result = btc_return > 0
    if not result:
        log(1, 1, "BTC_DAILY", "FAIL")
    return result

def s1_p2_btc_1h(data):
    if not CFG["02_BTC_1H_ENABLE"]:
        return True
    result = data["btc_price"] >= data["btc_ema20_1h"]
    if not result:
        log(1, 2, "BTC_1H", "FAIL")
    return result

def s1_p3_symbol_1h(data):
    if not CFG["03_SYMBOL_1H_ENABLE"]:
        return True
    result = data["close"] >= data["symbol_ema20_1h"]
    if not result:
        log(1, 3, "SYMBOL_1H", "FAIL")
    return result

def s1_p4_shock(data):
    if not CFG["04_SHOCK_ENABLE"]:
        return True
    price_change = abs((data["close"] - data["price_1m_prev"]) / data["price_1m_prev"] * 100)
    result = price_change <= CFG["14_SHOCK_PCT"]
    if not result:
        log(1, 4, "SHOCK", "FAIL")
    return result

def s1_p5_volume(data):
    if not CFG["05_VOLUME_ENABLE"]:
        return True
    result = data["volume"] >= data["volume_avg"] * CFG["15_VOLUME_K"]
    if not result:
        log(1, 5, "VOLUME", "FAIL")
    return result

def s1_p6_spread():
    if not CFG["06_SPREAD_ENABLE"]:
        return True
    return True

def s1_p7_dead_market(data):
    if not CFG["07_DEAD_MARKET_ENABLE"]:
        return True
    atr_pct = data["atr15"] / data["close"] * 100
    result = atr_pct >= CFG["17_ATR_MIN_PCT"]
    if not result:
        log(1, 7, "DEAD_MARKET", "FAIL")
    return result

def s1_p8_math_guard(data):
    if not CFG["08_MATH_GUARD_ENABLE"]:
        return True
    ema_fast = data["ema_values"].get(CFG["21_ENTRY_EMA_FAST"])
    ema_slow = data["ema_values"].get(CFG["22_ENTRY_EMA_SLOW"])
    ema_fast_prev = data["ema_values_prev"].get(CFG["21_ENTRY_EMA_FAST"])
    ema_slow_prev = data["ema_values_prev"].get(CFG["22_ENTRY_EMA_SLOW"])
    result = (data["vwap"] is not None and ema_fast is not None and ema_slow is not None 
              and ema_fast_prev is not None and ema_slow_prev is not None)
    if not result:
        log(1, 8, "MATH_GUARD", "FAIL")
    return result

def s1_p9_funding():
    if not CFG["09_FUNDING_ENABLE"]:
        return True
    return True

def stage1_preentry(data):
    if not s1_p1_btc_daily(data):
        return False
    if not s1_p2_btc_1h(data):
        return False
    if not s1_p3_symbol_1h(data):
        return False
    if not s1_p4_shock(data):
        return False
    if not s1_p5_volume(data):
        return False
    if not s1_p6_spread():
        return False
    if not s1_p7_dead_market(data):
        return False
    if not s1_p8_math_guard(data):
        return False
    if not s1_p9_funding():
        return False
    return True

# ============================================================
# STAGE 2 — ENTRY (LONG)
# ============================================================
def stage2_entry(data):
    """
    LONG ENTRY 조건 (추세 추종):
    1. 가격이 VWAP 위 (정배열 추세 확인)
    2. EMA5가 EMA15를 아래에서 위로 상향 돌파
    
    의미:
    - 정배열 상승 추세 진입 ⭕
    - 저점 줍기 ❌
    - 완료봉 기준 크로스 감지 (intrabar 판단 ❌)
    """
    ema_fast = data["ema_values"].get(CFG["21_ENTRY_EMA_FAST"])
    ema_slow = data["ema_values"].get(CFG["22_ENTRY_EMA_SLOW"])
    ema_fast_prev = data["ema_values_prev"].get(CFG["21_ENTRY_EMA_FAST"])
    ema_slow_prev = data["ema_values_prev"].get(CFG["22_ENTRY_EMA_SLOW"])
    
    if ema_fast is None or ema_slow is None or ema_fast_prev is None or ema_slow_prev is None:
        return False
    
    price_above_vwap = data["close"] > data["vwap"]
    bullish_cross = ema_fast_prev < ema_slow_prev and ema_fast > ema_slow
    
    return price_above_vwap and bullish_cross

# ============================================================
# STAGE 3 — ENTRY MANAGEMENT (LONG)
# ============================================================
def s3_p1_ik_vwap(data):
    if not CFG["30_IK_VWAP_ENABLE"]:
        return False
    result = data["close"] < data["vwap"]
    if result:
        log(3, 30, "IK_VWAP", "EXIT")
    return result

def s3_p2_ik_ema(data):
    if not CFG["31_IK_EMA_ENABLE"]:
        return False
    ema_fast = data["ema_values"].get(CFG["41_IK_EMA_FAST"])
    ema_slow = data["ema_values"].get(CFG["42_IK_EMA_SLOW"])
    if ema_fast is None or ema_slow is None:
        return False
    result = ema_fast < ema_slow
    if result:
        log(3, 31, "IK_EMA", "EXIT")
    return result

def s3_p3_ik_volume(data, ctx):
    if not CFG["32_IK_VOLUME_ENABLE"]:
        return False
    if ctx.entry_volume_ref is None:
        return False
    result = data["volume"] > ctx.entry_volume_ref and data["close"] <= data["open"]
    if result:
        log(3, 32, "IK_VOLUME", "EXIT")
    return result

def s3_p4_big_ema(data, ctx):
    if not CFG["33_BIG_EMA_ENABLE"]:
        return False
    
    if ctx.big_ema_state == "NONE":
        ctx.big_ema_state = "PENDING"
        ctx.big_ema_entry_bar_ts = data["bar_ts"]
        ctx.big_ema_bar_count = 0
    
    if ctx.big_ema_entry_bar_ts != data["bar_ts"]:
        ctx.big_ema_bar_count += 1
        ctx.big_ema_entry_bar_ts = data["bar_ts"]
    
    if ctx.big_ema_state == "PENDING":
        ema_fast = data["ema_values"].get(CFG["43_BIG_EMA_FAST"])
        ema_slow = data["ema_values"].get(CFG["44_BIG_EMA_SLOW"])
        if ema_fast is None or ema_slow is None:
            return False
        
        # 현재 절대값 기준 (0.0001)
        # 향후 개선: 퍼센트 기준으로 변경 가능
        ema_distance = abs(ema_fast - ema_slow)
        ema_expanding = ema_distance > 0.0001
        
        if ema_expanding:
            ctx.big_ema_state = "PASS"
            return False
        
        if ctx.big_ema_bar_count > CFG["33_GRACE_BARS"]:
            ctx.big_ema_state = "FAIL"
            log(3, 33, "BIG_EMA", "EXIT")
            return True
    
    return False

def s3_p5_big_donchian(data):
    """
    BIG_DONCHIAN (LONG):
    - 최근 LOOKBACK 고점 돌파 시 EXIT
    - 의미: 추세 추종 EXIT ❌ / 고점 과열 수익 보호 ⭕
    - 현재 CFG: ENABLE=False (비활성 상태)
    """
    if not CFG["34_BIG_DONCHIAN_ENABLE"]:
        return False
    result = data["high"] > data["donchian_high"]
    if result:
        log(3, 34, "BIG_DONCHIAN", "EXIT")
    return result

def s3_p6_small_pullback():
    if not CFG["35_SMALL_PULLBACK_ENABLE"]:
        return False
    return False

def s3_p7_small_vol():
    if not CFG["36_SMALL_VOL_ENABLE"]:
        return False
    return False

def s3_p8_small_escgo():
    if not CFG["37_SMALL_ESCGO_ENABLE"]:
        return False
    return False

def stage3_management(data, ctx):
    if s3_p1_ik_vwap(data):
        return True
    if s3_p2_ik_ema(data):
        return True
    if s3_p3_ik_volume(data, ctx):
        return True
    if s3_p4_big_ema(data, ctx):
        return True
    if s3_p5_big_donchian(data):
        return True
    if s3_p6_small_pullback():
        return True
    if s3_p7_small_vol():
        return True
    if s3_p8_small_escgo():
        return True
    return False

# ============================================================
# STAGE 4 — EXIT GATE (LONG)
# ============================================================
def s4_p1_sl(data, ctx):
    """LONG SL: (entry - close) / entry * 100"""
    if ctx.entry_price is None:
        return False
    loss_pct = (ctx.entry_price - data["close"]) / ctx.entry_price * 100
    result = loss_pct > CFG["50_SL_PCT"]
    if result:
        log(4, 50, "SL", "EXIT")
    return result

def s4_p2_vwap_break(data):
    """VWAP 재이탈 EXIT: 가격이 VWAP 아래로 내려가면 추세 붕괴"""
    if not CFG["51_VWAP_BREAK_ENABLE"]:
        return False
    result = data["close"] < data["vwap"]
    if result:
        log(4, 51, "VWAP_BREAK_EXIT", "EXIT")
    return result

def s4_p3_ema_slope(data):
    """
    EMA SLOPE EXIT (LONG):
    - 현재 상태: 의도적 봉인 (54_SLOPE_EXIT_PCT=999.0)
    - 활성화 시 조건: slope < -threshold (하락 전환)
    - 현재는 실행되지 않음
    """
    slope_ema_period = CFG["54_SLOPE_EMA"]
    lookback = CFG["54_SLOPE_LOOKBACK"]
    
    if len(data["slope_ema_hist"]) < lookback + 1:
        return False
    
    current_ema = data["slope_ema_hist"][-1]
    prev_ema = data["slope_ema_hist"][-lookback - 1]
    
    if current_ema is None or prev_ema is None:
        return False
    
    slope_pct = (current_ema - prev_ema) / prev_ema * 100
    result = slope_pct < -CFG["54_SLOPE_EXIT_PCT"]
    if result:
        log(4, 54, "EMA_SLOPE", "EXIT")
    return result

def s4_p4_ema_cross(data):
    """EMA5 < EMA10 하향 돌파 시 EXIT"""
    ema_fast = data["ema_values"].get(CFG["56_EXIT_EMA_FAST"])
    ema_slow = data["ema_values"].get(CFG["56_EXIT_EMA_SLOW"])
    
    if ema_fast is None or ema_slow is None:
        return False
    
    result = ema_fast < ema_slow
    if result:
        log(4, 56, "EMA_CROSS", "EXIT")
    return result

def s4_p5_donchian_low(data):
    """Donchian LOW 이탈 EXIT: 최근 N봉 저점 붕괴 시 추세 실패"""
    if not CFG["57_DONCHIAN_LOW_ENABLE"]:
        return False
    result = data["close"] < data["donchian_low"]
    if result:
        log(4, 57, "DONCHIAN_LOW_EXIT", "EXIT")
    return result

def s4_p6_avg_break(data):
    """N봉 평균 이탈 EXIT: 단기 추세 평균 붕괴"""
    if not CFG["58_EXIT_AVG_ENABLE"]:
        return False
    result = data["close"] < data["avg_close_n"]
    if result:
        log(4, 58, "AVG_BREAK_EXIT", "EXIT")
    return result

def stage4_exit(data, ctx):
    if s4_p1_sl(data, ctx):
        return True
    if s4_p2_vwap_break(data):
        return True
    if s4_p3_ema_slope(data):
        return True
    if s4_p4_ema_cross(data):
        return True
    if s4_p5_donchian_low(data):
        return True
    if s4_p6_avg_break(data):
        return True
    return False

# ============================================================
# ENGINE — 완료봉 기준 단일 실행
# ============================================================
def emergency_close(fx, ctx):
    if ctx.position is not None and ctx.position_qty > 0:
        print(f"EMERGENCY CLOSE: Closing {ctx.position_qty} {ctx.position}")
        fx.order("SELL", ctx.position_qty, reduce_only=True)
        print("Position closed")

def run():
    if not CFG["ENGINE_ENABLE"]:
        print("ENGINE DISABLED")
        return
    
    client = init_binance_client()
    setup_futures_env(client, CFG["TRADE_SYMBOL"], CFG["LEVERAGE"])
    fx = FX(client, CFG["TRADE_SYMBOL"])
    ctx = Context()
    
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down...")
        emergency_close(fx, ctx)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    last_bar_ts = None
    
    print("ENGINE START")
    
    try:
        while True:
            data = collect_data(CFG["TRADE_SYMBOL"])
            
            if data is None:
                time.sleep(3)
                continue
            
            if data["bar_ts"] == last_bar_ts:
                time.sleep(3)
                continue
            
            last_bar_ts = data["bar_ts"]
            
            if ctx.position is None:
                s1_pass = stage1_preentry(data)
                if s1_pass:
                    s2_pass = stage2_entry(data)
                    if s2_pass:
                        qty = fx.order("BUY", (CFG["CAPITAL_USDT"] * 0.95) / data["close"], price=data["close"], reduce_only=False)
                        if qty > 0:
                            ctx.position = "LONG"
                            ctx.entry_price = data["close"]
                            ctx.entry_volume_ref = data["volume"]
                            ctx.position_qty = qty
                            ctx.entry_bar_ts = data["bar_ts"]
                            ctx.big_ema_state = "NONE"
                            ctx.big_ema_entry_bar_ts = None
                            ctx.big_ema_bar_count = 0
                            print(f"[ENTRY] LONG price={q(data['close'])} qty={qty}")
            else:
                if data["bar_ts"] == ctx.entry_bar_ts:
                    time.sleep(3)
                    continue
                
                ema5 = data["ema_values"].get(5)
                ema10 = data["ema_values"].get(10)
                ema15 = data["ema_values"].get(15)
                print(f"[POSITION] LONG | close={q(data['close'])} vwap={q(data['vwap'])} | ema5={q(ema5) if ema5 else 'N/A'} ema10={q(ema10) if ema10 else 'N/A'} ema15={q(ema15) if ema15 else 'N/A'}")
                
                s3_kill = stage3_management(data, ctx)
                if s3_kill:
                    fx.order("SELL", ctx.position_qty, reduce_only=True)
                    print(f"[EXIT] S3 price={q(data['close'])}")
                    ctx.__init__()
                else:
                    s4_kill = stage4_exit(data, ctx)
                    if s4_kill:
                        fx.order("SELL", ctx.position_qty, reduce_only=True)
                        print(f"[EXIT] S4 price={q(data['close'])}")
                        ctx.__init__()
            
            time.sleep(3)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected")
        emergency_close(fx, ctx)
    except Exception as e:
        print(f"Unexpected error: {e}")
        emergency_close(fx, ctx)
        raise

if __name__ == "__main__":
    run()