# ============================================================
# VELLA_v10_OPTIMIZER — EMA Combination Finder (v2.8.4)
# - PATCH: EMA warm-up 보존 / LONG E2 pullback 부호 / ATR 슬라이스 오염 제거
# ============================================================

import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ============================================================
# CFG
# ============================================================

CFG = {
    "100_LONG_SYMBOLS": [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "AVAXUSDT", "LINKUSDT",
        "ADAUSDT", "DOGEUSDT", "SUIUSDT"
    ],

    "101_SHORT_SYMBOLS": [
        "SEIUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT",
        "INJUSDT", "APTUSDT", "TIAUSDT",
        "RNDRUSDT", "ATOMUSDT", "NEARUSDT"
    ],

    "102_FAST_RANGE": [5, 10],
    "103_MID_RANGE": [12, 24],
    "104_EXIT_RANGE": [3, 6],

    "105_BACKTEST_DAYS": 4,
    "106_FETCH_EXTRA_DAYS": 1,
    "107_INTERVAL": "5m",

    "110_FEE_RATE": 0.0004,
    "111_SLIPPAGE_RATE": 0.00015,

    "113_OVERTRADE_PENALTY_PER_TRADE": 0.05,
    "114_OVERTRADE_BASELINE_PER_DAY": 5,
    "115_MDD_WEIGHT": 1.5,

    "120_SHOCK_ATR_MULTIPLIER": 2.2,
    "121_SHOCK_VOLUME_MULTIPLIER": 2.2,
    "122_SHOCK_4H_RANGE_PCT": 4.5,

    "130_EMAIL_TO": "jktiger486@gmail.com",
    "131_EMAIL_FROM": "jktiger486@gmail.com",
    "132_SMTP_HOST": "email-smtp.ap-northeast-2.amazonaws.com",
    "133_SMTP_PORT": 587,

    "141_LOG_LEVEL": "INFO",

    "150_ARENA_RANGE": [30, 50],
    "151_TOUCH_TOLERANCE_RANGE": [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003],
    "152_SLOPE_THRESHOLD_RANGE": [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003],
    "153_SWING_LOOKBACK_RANGE": [3, 10],
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CFG["141_LOG_LEVEL"], logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VELLA_OPTIMIZER")

# ============================================================
# BINANCE FUTURES REST
# ============================================================

BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUTURES_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"

def get_bars_per_day(interval: str) -> int:
    mapping = {
        "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
        "30m": 48, "1h": 24, "2h": 12, "4h": 6,
        "6h": 4, "12h": 2, "1d": 1
    }
    return mapping.get(interval, 288)

def fetch_klines(symbol: str, interval: str, days: int) -> Optional[List]:
    try:
        bars_per_day = get_bars_per_day(interval)
        limit = min(days * bars_per_day + 100, 1500)
        r = requests.get(
            BINANCE_FUTURES_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_klines {symbol}: {e}")
        return None

def fetch_klines_paged(symbol: str, interval: str, total_bars: int) -> Optional[List]:
    try:
        interval_ms = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
        }
        ms_per_bar = interval_ms.get(interval, 300_000)
        per_page   = 1500
        all_bars   = []
        end_time   = int(datetime.now(timezone.utc).timestamp() * 1000)

        remaining = total_bars
        while remaining > 0:
            fetch_count = min(remaining, per_page)
            start_time  = end_time - fetch_count * ms_per_bar

            r = requests.get(
                BINANCE_FUTURES_KLINES,
                params={
                    "symbol":    symbol,
                    "interval":  interval,
                    "startTime": start_time,
                    "endTime":   end_time,
                    "limit":     fetch_count,
                },
                timeout=10
            )
            r.raise_for_status()
            chunk = r.json()

            if not chunk:
                break

            all_bars  = chunk + all_bars
            end_time  = int(chunk[0][0]) - 1
            remaining -= len(chunk)

            if len(chunk) < fetch_count:
                break

        seen   = set()
        result = []
        for bar in all_bars:
            ts = bar[0]
            if ts not in seen:
                seen.add(ts)
                result.append(bar)
        result.sort(key=lambda x: x[0])
        return result if result else None

    except Exception as e:
        log.error(f"fetch_klines_paged {symbol}: {e}")
        return None

def fetch_24h_ticker(symbol: str) -> Optional[Dict]:
    try:
        r = requests.get(BINANCE_FUTURES_TICKER, params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_24h_ticker {symbol}: {e}")
        return None

# ============================================================
# INDICATORS
# ============================================================

def ema_series(values: List[float], period: int) -> List[float]:
    if not values or len(values) < period:
        return [values[0]] * len(values) if values else []
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

def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
    if len(highs) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, period + 1):
        h      = highs[-i]
        l      = lows[-i]
        c_prev = closes[-(i+1)]
        tr     = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    return sum(tr_list) / len(tr_list)

def atr_series(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
    """Wilder RMA ATR 시리즈. 첫 유효 인덱스 = period-1."""
    n = len(closes)
    if n == 0:
        return []

    tr_list = [0.0] * n
    tr_list[0] = highs[0] - lows[0]
    for i in range(1, n):
        h  = highs[i]
        l  = lows[i]
        cp = closes[i - 1]
        tr_list[i] = max(h - l, abs(h - cp), abs(l - cp))

    out = [0.0] * n

    if n < period:
        avg = sum(tr_list) / n
        return [avg] * n

    first_atr       = sum(tr_list[1:period + 1]) / period
    out[period - 1] = first_atr

    prev = first_atr
    for i in range(period, n):
        prev   = (prev * (period - 1) + tr_list[i]) / period
        out[i] = prev

    for i in range(period - 1):
        out[i] = out[period - 1]

    return out

# ============================================================
# STATE
# ============================================================

@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float
    pnl_pct: float

@dataclass
class BacktestResult:
    symbol: str
    fast: int
    mid: int
    arena: int
    exit_ema: int
    tolerance: float
    slope_threshold: float
    swing_lookback: int
    net_return: float
    mdd: float
    total_trades: int
    trades_per_day: float
    win_rate: float
    recent_2d_price_change: float
    score: float
    trades: List[Trade]

# ============================================================
# BACKTEST ENGINE (SHORT)
# ============================================================

def backtest_short(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    ema_fast_s: List[float],
    ema_mid_s: List[float],
    ema_arena_s: List[float],
    ema_exit_s: List[float],
    tolerance: float,
    slope_threshold: float,
    swing_lookback: int,
) -> Tuple[List[Trade], float, float]:

    trades   = []
    position = None

    equity = 1.0
    peak   = 1.0
    max_dd = 0.0

    fee_per_trade = CFG["110_FEE_RATE"] + CFG["111_SLIPPAGE_RATE"]

    for bar in range(60, len(closes) - 1):
        close_now = closes[bar]

        if position is not None:
            if bar == position['entry_bar']:
                continue

            if close_now > ema_exit_s[bar]:
                exit_price  = close_now
                pnl_pct     = (position['entry_price'] - exit_price) / position['entry_price']
                pnl_pct_net = pnl_pct - (fee_per_trade * 2)

                trades.append(Trade(
                    entry_bar=position['entry_bar'],
                    entry_price=position['entry_price'],
                    exit_bar=bar,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct_net
                ))

                equity *= (1 + pnl_pct_net)
                peak    = max(peak, equity)
                dd      = (peak - equity) / peak
                max_dd  = max(max_dd, dd)

                position = None
                continue  # 청산 bar 즉시 재진입 금지

        if position is None:
            short_arena = ema_fast_s[bar] < ema_arena_s[bar]

            slope_ok = False
            if bar >= swing_lookback:
                ref = ema_fast_s[bar - swing_lookback]
                if ref > 0:
                    slope_val = (ema_fast_s[bar] - ref) / ref
                    slope_ok  = slope_val <= -slope_threshold

            e1_signal = False
            if bar >= 1:
                e1_signal = (
                    ema_fast_s[bar-1] >= ema_mid_s[bar-1] and
                    ema_fast_s[bar]   <  ema_mid_s[bar]
                )

            e2_signal = False
            if bar >= 1:
                pullback  = highs[bar-1] >= ema_fast_s[bar-1] * (1.0 - tolerance)
                reentry   = closes[bar]  <  ema_fast_s[bar]
                e2_signal = pullback and reentry

            if short_arena and slope_ok and (e1_signal or e2_signal):
                position = {'entry_bar': bar, 'entry_price': close_now}

    if position is not None:
        exit_price  = closes[-2]
        pnl_pct     = (position['entry_price'] - exit_price) / position['entry_price']
        pnl_pct_net = pnl_pct - (fee_per_trade * 2)

        trades.append(Trade(
            entry_bar=position['entry_bar'],
            entry_price=position['entry_price'],
            exit_bar=len(closes) - 2,
            exit_price=exit_price,
            pnl_pct=pnl_pct_net
        ))

        equity *= (1 + pnl_pct_net)
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak
        max_dd  = max(max_dd, dd)

    return trades, max_dd, equity - 1.0

# ============================================================
# BACKTEST ENGINE (LONG)
# ============================================================

def backtest_long(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    ema_fast_s: List[float],
    ema_mid_s: List[float],
    ema_arena_s: List[float],
    ema_exit_s: List[float],
    tolerance: float,
    slope_threshold: float,
    swing_lookback: int,
) -> Tuple[List[Trade], float, float]:

    trades   = []
    position = None

    equity = 1.0
    peak   = 1.0
    max_dd = 0.0

    fee_per_trade = CFG["110_FEE_RATE"] + CFG["111_SLIPPAGE_RATE"]

    for bar in range(60, len(closes) - 1):
        close_now = closes[bar]

        if position is not None:
            if bar == position['entry_bar']:
                continue

            if close_now < ema_exit_s[bar]:
                exit_price  = close_now
                pnl_pct     = (exit_price - position['entry_price']) / position['entry_price']
                pnl_pct_net = pnl_pct - (fee_per_trade * 2)

                trades.append(Trade(
                    entry_bar=position['entry_bar'],
                    entry_price=position['entry_price'],
                    exit_bar=bar,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct_net
                ))

                equity *= (1 + pnl_pct_net)
                peak    = max(peak, equity)
                dd      = (peak - equity) / peak
                max_dd  = max(max_dd, dd)

                position = None
                continue  # 청산 bar 즉시 재진입 금지

        if position is None:
            long_arena = ema_fast_s[bar] > ema_arena_s[bar]

            slope_ok = False
            if bar >= swing_lookback:
                ref = ema_fast_s[bar - swing_lookback]
                if ref > 0:
                    slope_val = (ema_fast_s[bar] - ref) / ref
                    slope_ok  = slope_val >= slope_threshold

            e1_signal = False
            if bar >= 1:
                e1_signal = (
                    ema_fast_s[bar-1] <= ema_mid_s[bar-1] and
                    ema_fast_s[bar]   >  ema_mid_s[bar]
                )

            e2_signal = False
            if bar >= 1:
                pullback  = lows[bar-1] <= ema_fast_s[bar-1] * (1.0 + tolerance)
                reentry   = closes[bar]  >  ema_fast_s[bar]
                e2_signal = pullback and reentry

            if long_arena and slope_ok and (e1_signal or e2_signal):
                position = {'entry_bar': bar, 'entry_price': close_now}

    if position is not None:
        exit_price  = closes[-2]
        pnl_pct     = (exit_price - position['entry_price']) / position['entry_price']
        pnl_pct_net = pnl_pct - (fee_per_trade * 2)

        trades.append(Trade(
            entry_bar=position['entry_bar'],
            entry_price=position['entry_price'],
            exit_bar=len(closes) - 2,
            exit_price=exit_price,
            pnl_pct=pnl_pct_net
        ))

        equity *= (1 + pnl_pct_net)
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak
        max_dd  = max(max_dd, dd)

    return trades, max_dd, equity - 1.0

# ============================================================
# SCORE CALCULATION
# ============================================================

def calculate_recent_2d_price_change(closes: List[float], interval: str) -> float:
    bars_per_day = get_bars_per_day(interval)
    bars_2d = bars_per_day * 2
    if len(closes) < bars_2d:
        return 0.0
    return (closes[-1] - closes[-bars_2d]) / closes[-bars_2d] * 100

def calculate_score(net_return: float, mdd: float, trades_per_day: float) -> float:
    overtrade_penalty = max(0, trades_per_day - CFG["114_OVERTRADE_BASELINE_PER_DAY"]) * CFG["113_OVERTRADE_PENALTY_PER_TRADE"]
    return net_return - (abs(mdd) * CFG["115_MDD_WEIGHT"]) - overtrade_penalty

def optimize_symbol(symbol: str, direction: str, klines: List) -> Optional[BacktestResult]:
    closes = [float(k[4]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]

    interval      = CFG["107_INTERVAL"]
    backtest_days = CFG["105_BACKTEST_DAYS"]
    bars_per_day  = get_bars_per_day(interval)
    backtest_bars = backtest_days * bars_per_day

    if len(closes) < backtest_bars + 100:
        return None

    closes_trimmed = closes[-backtest_bars:]
    highs_trimmed  = highs[-backtest_bars:]
    lows_trimmed   = lows[-backtest_bars:]

    fast_range  = range(CFG["102_FAST_RANGE"][0],  CFG["102_FAST_RANGE"][1]  + 1)
    mid_range   = range(CFG["103_MID_RANGE"][0],   CFG["103_MID_RANGE"][1]   + 1)
    exit_range  = range(CFG["104_EXIT_RANGE"][0],  CFG["104_EXIT_RANGE"][1]  + 1)
    arena_range = range(CFG["150_ARENA_RANGE"][0], CFG["150_ARENA_RANGE"][1] + 1)

    all_periods = set(fast_range) | set(mid_range) | set(exit_range) | set(arena_range)
    ema_cache_full: Dict[int, List[float]] = {
        p: ema_series(closes, p) for p in all_periods
    }
    ema_cache: Dict[int, List[float]] = {
        p: ema_cache_full[p][-backtest_bars:] for p in all_periods
    }

    tolerance_range = CFG["151_TOUCH_TOLERANCE_RANGE"]
    slope_range     = CFG["152_SLOPE_THRESHOLD_RANGE"]
    swing_range     = range(CFG["153_SWING_LOOKBACK_RANGE"][0], CFG["153_SWING_LOOKBACK_RANGE"][1] + 1)

    best_result = None
    best_score  = -999999

    for fast in fast_range:
        ema_fast_s = ema_cache[fast]
        for mid in mid_range:
            if mid <= fast or mid - fast < 3:
                continue
            ema_mid_s = ema_cache[mid]

            for arena in arena_range:
                if arena <= mid:
                    continue
                ema_arena_s = ema_cache[arena]

                for tolerance in tolerance_range:
                    for slope_th in slope_range:
                        for swing_lb in swing_range:
                            for exit_ema in exit_range:
                                ema_exit_s = ema_cache[exit_ema]

                                if direction == "SHORT":
                                    trades, mdd, net_return_ratio = backtest_short(
                                        closes_trimmed, highs_trimmed, lows_trimmed,
                                        ema_fast_s, ema_mid_s, ema_arena_s, ema_exit_s,
                                        tolerance, slope_th, swing_lb,
                                    )
                                else:
                                    trades, mdd, net_return_ratio = backtest_long(
                                        closes_trimmed, highs_trimmed, lows_trimmed,
                                        ema_fast_s, ema_mid_s, ema_arena_s, ema_exit_s,
                                        tolerance, slope_th, swing_lb,
                                    )

                                if not trades:
                                    continue

                                net_return_pct = net_return_ratio * 100
                                trades_per_day = len(trades) / backtest_days
                                wins           = sum(1 for t in trades if t.pnl_pct > 0)
                                win_rate       = (wins / len(trades)) * 100 if trades else 0
                                recent_2d      = calculate_recent_2d_price_change(closes, interval)
                                score          = calculate_score(net_return_pct, mdd * 100, trades_per_day)

                                if score > best_score:
                                    best_score  = score
                                    best_result = BacktestResult(
                                        symbol=symbol,
                                        fast=fast, mid=mid, arena=arena, exit_ema=exit_ema,
                                        tolerance=tolerance, slope_threshold=slope_th, swing_lookback=swing_lb,
                                        net_return=net_return_pct, mdd=mdd * 100,
                                        total_trades=len(trades), trades_per_day=trades_per_day,
                                        win_rate=win_rate, recent_2d_price_change=recent_2d,
                                        score=score, trades=trades
                                    )

    return best_result

# ============================================================
# SHOCK REGIME
# ============================================================

def check_shock() -> bool:
    try:
        # 4h range 조건 — 미완성 캔들 회피
        klines_4h = fetch_klines("BTCUSDT", "4h", 1)
        if klines_4h and len(klines_4h) >= 1:
            k         = klines_4h[-2] if len(klines_4h) >= 2 else klines_4h[-1]
            range_pct = (float(k[2]) - float(k[3])) / float(k[3]) * 100
            if range_pct >= CFG["122_SHOCK_4H_RANGE_PCT"]:
                return True

        # 5m ATR 조건
        target_bars = 288 * 7 + 400
        klines_5m   = fetch_klines_paged("BTCUSDT", "5m", target_bars)

        if klines_5m and len(klines_5m) >= 288 + 15:
            closes_5m = [float(k[4]) for k in klines_5m]
            highs_5m  = [float(k[2]) for k in klines_5m]
            lows_5m   = [float(k[3]) for k in klines_5m]

            atr5 = atr_series(highs_5m, lows_5m, closes_5m, 14)

            if len(atr5) < 2:
                return False
            atr_now = atr5[-2]

            window    = 288 * 7
            atr_slice = atr5[-(window + 2):-2] if len(atr5) >= window + 2 else atr5[:-2]
            valid     = [v for v in atr_slice if v > 0]
            if not valid:
                return False
            atr_7d_avg = sum(valid) / len(valid)

            if atr_7d_avg > 0 and atr_now >= atr_7d_avg * CFG["120_SHOCK_ATR_MULTIPLIER"]:
                # 1h volume 조건 — 미완성 캔들 회피
                target_1h_bars = 24 * 7 + 10
                klines_1h = fetch_klines_paged("BTCUSDT", "1h", target_1h_bars)
                if klines_1h and len(klines_1h) >= 3:
                    volumes_1h = [float(k[5]) for k in klines_1h]
                    vol_now    = volumes_1h[-2]
                    use_bars   = min(len(volumes_1h) - 1, 24 * 7)
                    hist       = volumes_1h[-(use_bars + 2):-2]
                    if hist:
                        vol_7d_avg = sum(hist) / len(hist)
                        if vol_now >= vol_7d_avg * CFG["121_SHOCK_VOLUME_MULTIPLIER"]:
                            return True

        return False
    except Exception as e:
        log.error(f"check_shock: {e}")
        return False

# ============================================================
# EMAIL
# ============================================================

def format_table_row(rank: int, result: BacktestResult) -> str:
    wr_flag = "⚠" if result.win_rate < 35 else ("★" if result.win_rate > 55 else "")
    return (
        f"{rank} | {result.symbol} | "
        f"{result.fast} | {result.mid} | {result.arena} | {result.exit_ema} | "
        f"tol={result.tolerance:.4f} | slp={result.slope_threshold:.4f} | sw={result.swing_lookback} | "
        f"{result.net_return:.2f}% | {result.mdd:.2f}% | {result.recent_2d_price_change:.2f}% | "
        f"{result.win_rate:.1f}%{wr_flag} | {result.total_trades} | {result.score:.2f}"
    )

def generate_email_body(long_results: List[BacktestResult], short_results: List[BacktestResult], shock: bool) -> str:
    body = ["⚠ 본 리포트는 일봉 마감(09:00 KST) 전 데이터 기준입니다.\n"]

    if shock:
        body.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "⚠ 현재 시장은 SHOCK 상태입니다.",
            "변동성 급증 구간으로 포지션 강도 축소 권고.",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        ])

    header = "순위 | 종목 | FAST | MID | ARENA | EXIT | tolerance | slope | swing | 4일Net | 4일MDD | 2일가격변동 | 승률 | 트레이드 | 점수"
    sep    = "─────────────────────────────────────────────────────────────────"

    body.extend([
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "[LONG TOP 10]",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        header, sep
    ])

    for i, result in enumerate(long_results[:10], 1):
        body.append(format_table_row(i, result))

    body.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "[SHORT TOP 10]",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        header, sep
    ])

    for i, result in enumerate(short_results[:10], 1):
        body.append(format_table_row(i, result))

    body.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "VELLA_v10_OPTIMIZER v2.8.4",
        f"Generated: {datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ])

    return "\n".join(body)

def send_email(subject: str, body: str):
    try:
        smtp_user = os.getenv("AWS_SES_SMTP_USER")
        smtp_pass = os.getenv("AWS_SES_SMTP_PASS")

        if not smtp_user or not smtp_pass:
            log.error(f"AWS SES credentials missing - USER: {bool(smtp_user)}, PASS: {bool(smtp_pass)}")
            return

        log.info(f"Sending email to {CFG['130_EMAIL_TO']}")

        msg = MIMEMultipart()
        msg['From']    = CFG["131_EMAIL_FROM"]
        msg['To']      = CFG["130_EMAIL_TO"]
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        server = smtplib.SMTP(CFG["132_SMTP_HOST"], CFG["133_SMTP_PORT"])
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()

        log.info("Email sent successfully")
    except Exception as e:
        log.error(f"send_email: {e}")

# ============================================================
# SCHEDULE GATE
# ============================================================

LAST_SENT_FILE = "/home/ubuntu/vella_v10/last_sent.txt"

def get_last_sent_date() -> Optional[str]:
    try:
        if os.path.exists(LAST_SENT_FILE):
            with open(LAST_SENT_FILE, 'r') as f:
                return f.read().strip()
    except Exception as e:
        log.error(f"get_last_sent_date: {e}")
    return None

def update_last_sent_date(date_str: str):
    try:
        os.makedirs(os.path.dirname(LAST_SENT_FILE), exist_ok=True)
        with open(LAST_SENT_FILE, 'w') as f:
            f.write(date_str)
        log.info(f"Updated last_sent: {date_str}")
    except Exception as e:
        log.error(f"update_last_sent_date: {e}")

def run_optimizer():
    log.info("VELLA_v10_OPTIMIZER START")

    total_days = CFG["105_BACKTEST_DAYS"] + CFG["106_FETCH_EXTRA_DAYS"]
    interval   = CFG["107_INTERVAL"]

    long_results  = []
    short_results = []

    for symbol in CFG["100_LONG_SYMBOLS"]:
        log.info(f"Optimizing LONG: {symbol}")
        klines = fetch_klines(symbol, interval, total_days)
        if klines:
            result = optimize_symbol(symbol, "LONG", klines)
            if result:
                long_results.append(result)

    for symbol in CFG["101_SHORT_SYMBOLS"]:
        log.info(f"Optimizing SHORT: {symbol}")
        klines = fetch_klines(symbol, interval, total_days)
        if klines:
            result = optimize_symbol(symbol, "SHORT", klines)
            if result:
                short_results.append(result)

    long_results.sort(key=lambda x: x.score, reverse=True)
    short_results.sort(key=lambda x: x.score, reverse=True)

    shock = check_shock()

    kst     = timezone(timedelta(hours=9))
    subject = f"VELLA OPTIMIZER Report - {datetime.now(kst).strftime('%Y-%m-%d')}"
    body    = generate_email_body(long_results, short_results, shock)

    send_email(subject, body)

    log.info("VELLA_v10_OPTIMIZER DONE")

# ============================================================
# MAIN (DAEMON MODE)
# ============================================================

if __name__ == "__main__":
    log.info("VELLA_v10_OPTIMIZER DAEMON START")

    while True:
        try:
            utc_now = datetime.now(timezone.utc)
            kst     = timezone(timedelta(hours=9))
            kst_now = utc_now.astimezone(kst)

            if kst_now.hour == 8 and 0 <= kst_now.minute < 10:
                today_str = kst_now.strftime("%Y-%m-%d")
                last_sent = get_last_sent_date()

                if last_sent != today_str:
                    log.info(f"Schedule triggered: {today_str} {kst_now.strftime('%H:%M:%S')} KST")
                    run_optimizer()
                    update_last_sent_date(today_str)
                else:
                    log.info(f"Already sent today: {last_sent}")

            time.sleep(30)

        except Exception as e:
            log.error(f"daemon loop error: {e}")
            time.sleep(30)