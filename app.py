# ============================================================
# VELLA_v9_OPTIMIZER — EMA Combination Finder (v2.5 FINAL)
# - PURPOSE: Find optimal EMA combinations for LONG/SHORT strategies
# - TARGET: 10 LONG symbols / 10 SHORT symbols
# - OUTPUT: Daily email report with TOP 10 rankings
# - SCHEDULE: 07:50 backtest → 08:00 email send
# - REVISION: 벨라 최종 지적 반영 - 완전 정합 확정
# ============================================================

import os
import sys
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ============================================================
# CFG (ALL CONTROL HERE)
# ============================================================

CFG = {
    # -------------------------
    # SYMBOLS
    # -------------------------
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
    
    # -------------------------
    # EMA SEARCH RANGES
    # -------------------------
    "102_FAST_RANGE": [6, 14],
    "103_MID_RANGE": [10, 24],
    "104_EXIT_RANGE": [3, 6],
    
    # -------------------------
    # BACKTEST PERIOD
    # -------------------------
    "105_BACKTEST_DAYS": 4,
    "106_FETCH_EXTRA_DAYS": 1,  # EMA 안정용 여유
    "107_INTERVAL": "5m",
    
    # -------------------------
    # FEES & SLIPPAGE
    # -------------------------
    "110_FEE_RATE": 0.0004,        # 0.04%
    "111_SLIPPAGE_RATE": 0.00015,  # 0.015%
    
    # -------------------------
    # SCORE FORMULA
    # -------------------------
    "113_OVERTRADE_PENALTY_PER_TRADE": 0.05,
    "114_OVERTRADE_BASELINE_PER_DAY": 5,
    "115_MDD_WEIGHT": 1.5,
    
    # -------------------------
    # SHOCK REGIME (5m 고정)
    # -------------------------
    "120_SHOCK_ATR_MULTIPLIER": 2.2,
    "121_SHOCK_VOLUME_MULTIPLIER": 2.2,
    "122_SHOCK_4H_RANGE_PCT": 4.5,
    
    # -------------------------
    # EMAIL
    # -------------------------
    "130_EMAIL_TO": "jktiger486@gmail.com",
    "131_EMAIL_FROM": "vella-optimizer@noreply.com",
    "132_SMTP_HOST": "email-smtp.ap-northeast-2.amazonaws.com",  # AWS SES
    "133_SMTP_PORT": 587,
    
    # -------------------------
    # ENGINE
    # -------------------------
    "140_MAX_ENTRY_PER_TREND": 999,
    "141_LOG_LEVEL": "ERROR",
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CFG["141_LOG_LEVEL"], logging.ERROR),
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
    """interval 기준 하루 봉 수 계산"""
    mapping = {
        "1m": 1440,
        "3m": 480,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "2h": 12,
        "4h": 6,
        "6h": 4,
        "12h": 2,
        "1d": 1
    }
    return mapping.get(interval, 288)

def fetch_klines(symbol: str, interval: str, days: int) -> Optional[List]:
    """Binance Futures klines 조회"""
    try:
        bars_per_day = get_bars_per_day(interval)
        limit = days * bars_per_day + 100  # 여유분
        limit = min(limit, 1500)  # Binance Futures 최대 1500
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

def fetch_24h_ticker(symbol: str) -> Optional[Dict]:
    """24h ticker 조회"""
    try:
        r = requests.get(
            BINANCE_FUTURES_TICKER,
            params={"symbol": symbol},
            timeout=5
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_24h_ticker {symbol}: {e}")
        return None

# ============================================================
# INDICATORS
# ============================================================

def ema_series(values: List[float], period: int) -> List[float]:
    """EMA 계산"""
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
    """ATR 계산"""
    if len(highs) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, period + 1):
        h = highs[-i]
        l = lows[-i]
        c_prev = closes[-(i+1)]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    return sum(tr_list) / len(tr_list)

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
    exit_ema: int
    net_return: float
    mdd: float
    total_trades: int
    trades_per_day: float
    win_rate: float
    recent_2d_price_change: float  # 종목 가격 변동률 (전략 수익 아님)
    score: float
    trades: List[Trade]

# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest_short(
    closes: List[float],
    fast: int,
    mid: int,
    exit_ema: int,
    days: int
) -> Tuple[List[Trade], float, float]:
    """
    SHORT 백테스트 (v9 완료봉 기준 완전 정합)
    ENTRY 1: ema_fast ↓ ema_mid
    ENTRY 2: ema_fast < ema_mid AND close[-2] > ema_fast[-2] AND close[-1] < ema_fast[-1]
    EXIT: close > ema_exit
    """
    ema_fast_s = ema_series(closes, fast)
    ema_mid_s = ema_series(closes, mid)
    ema_exit_s = ema_series(closes, exit_ema)
    
    trades = []
    position = None
    entry_count = 0
    max_entry = CFG["140_MAX_ENTRY_PER_TREND"]
    
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    
    fee_per_trade = CFG["110_FEE_RATE"] + CFG["111_SLIPPAGE_RATE"]
    
    # 완료봉 기준 루프 - 마지막 봉 제외 (진행 중 봉)
    for bar in range(60, len(closes) - 1):
        close_now = closes[bar]
        
        # EXIT 우선
        if position is not None:
            if bar == position['entry_bar']:
                continue
            
            if close_now > ema_exit_s[bar]:
                # EXIT
                exit_price = close_now
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
                
                # 수수료 복리 반영
                pnl_pct_net = pnl_pct - (fee_per_trade * 2)  # 왕복
                
                trades.append(Trade(
                    entry_bar=position['entry_bar'],
                    entry_price=position['entry_price'],
                    exit_bar=bar,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct_net
                ))
                
                equity *= (1 + pnl_pct_net)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
                
                position = None
                entry_count = 0
        
        # ENTRY
        if position is None:
            if entry_count >= max_entry:
                continue
            
            # E1: Dead Cross (완료봉 기준)
            e1_signal = False
            if bar >= 1:
                e1_signal = (
                    ema_fast_s[bar-1] >= ema_mid_s[bar-1] and
                    ema_fast_s[bar] < ema_mid_s[bar]
                )
            
            # E2: Re-Acceleration
            e2_signal = False
            if not e1_signal:
                if ema_fast_s[bar] < ema_mid_s[bar]:
                    if bar >= 1:
                        pullback = closes[bar-1] > ema_fast_s[bar-1]
                        reentry = closes[bar] < ema_fast_s[bar]
                        e2_signal = pullback and reentry
            
            if e1_signal or e2_signal:
                position = {
                    'entry_bar': bar,
                    'entry_price': close_now
                }
                entry_count += 1
    
    # 루프 종료 시 미청산 포지션 강제 청산 (마지막 완료봉 기준)
    if position is not None:
        exit_price = closes[-2]
        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        pnl_pct_net = pnl_pct - (fee_per_trade * 2)
        
        trades.append(Trade(
            entry_bar=position['entry_bar'],
            entry_price=position['entry_price'],
            exit_bar=len(closes) - 2,
            exit_price=exit_price,
            pnl_pct=pnl_pct_net
        ))
        
        equity *= (1 + pnl_pct_net)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    
    return trades, max_dd, equity - 1.0

def backtest_long(
    closes: List[float],
    fast: int,
    mid: int,
    exit_ema: int,
    days: int
) -> Tuple[List[Trade], float, float]:
    """
    LONG 백테스트 (v10 완료봉 기준 완전 정합)
    ENTRY 1: ema_fast ↑ ema_mid
    ENTRY 2: ema_fast > ema_mid AND close[-2] < ema_fast[-2] AND close[-1] > ema_fast[-1]
    EXIT: close < ema_exit
    """
    ema_fast_s = ema_series(closes, fast)
    ema_mid_s = ema_series(closes, mid)
    ema_exit_s = ema_series(closes, exit_ema)
    
    trades = []
    position = None
    entry_count = 0
    max_entry = CFG["140_MAX_ENTRY_PER_TREND"]
    
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    
    fee_per_trade = CFG["110_FEE_RATE"] + CFG["111_SLIPPAGE_RATE"]
    
    # 완료봉 기준 루프 - 마지막 봉 제외 (진행 중 봉)
    for bar in range(60, len(closes) - 1):
        close_now = closes[bar]
        
        # EXIT 우선
        if position is not None:
            if bar == position['entry_bar']:
                continue
            
            if close_now < ema_exit_s[bar]:
                # EXIT
                exit_price = close_now
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                
                # 수수료 복리 반영
                pnl_pct_net = pnl_pct - (fee_per_trade * 2)  # 왕복
                
                trades.append(Trade(
                    entry_bar=position['entry_bar'],
                    entry_price=position['entry_price'],
                    exit_bar=bar,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct_net
                ))
                
                equity *= (1 + pnl_pct_net)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
                
                position = None
                entry_count = 0
        
        # ENTRY
        if position is None:
            if entry_count >= max_entry:
                continue
            
            # E1: Golden Cross (완료봉 기준)
            e1_signal = False
            if bar >= 1:
                e1_signal = (
                    ema_fast_s[bar-1] <= ema_mid_s[bar-1] and
                    ema_fast_s[bar] > ema_mid_s[bar]
                )
            
            # E2: Re-Acceleration
            e2_signal = False
            if not e1_signal:
                if ema_fast_s[bar] > ema_mid_s[bar]:
                    if bar >= 1:
                        pullback = closes[bar-1] < ema_fast_s[bar-1]
                        reentry = closes[bar] > ema_fast_s[bar]
                        e2_signal = pullback and reentry
            
            if e1_signal or e2_signal:
                position = {
                    'entry_bar': bar,
                    'entry_price': close_now
                }
                entry_count += 1
    
    # 루프 종료 시 미청산 포지션 강제 청산 (마지막 완료봉 기준)
    if position is not None:
        exit_price = closes[-2]
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        pnl_pct_net = pnl_pct - (fee_per_trade * 2)
        
        trades.append(Trade(
            entry_bar=position['entry_bar'],
            entry_price=position['entry_price'],
            exit_bar=len(closes) - 2,
            exit_price=exit_price,
            pnl_pct=pnl_pct_net
        ))
        
        equity *= (1 + pnl_pct_net)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    
    return trades, max_dd, equity - 1.0

def calculate_recent_2d_price_change(closes: List[float], interval: str) -> float:
    """최근 2일 종목 가격 변동률 (전략 수익 아님)"""
    bars_per_day = get_bars_per_day(interval)
    bars_2d = bars_per_day * 2
    
    if len(closes) < bars_2d:
        return 0.0
    start = closes[-bars_2d]
    end = closes[-1]
    return (end - start) / start * 100

def calculate_score(
    net_return: float,
    mdd: float,
    trades_per_day: float
) -> float:
    """점수 계산"""
    overtrade_penalty = max(0, trades_per_day - CFG["114_OVERTRADE_BASELINE_PER_DAY"]) * CFG["113_OVERTRADE_PENALTY_PER_TRADE"]
    score = net_return - (abs(mdd) * CFG["115_MDD_WEIGHT"]) - overtrade_penalty
    return score

def optimize_symbol(symbol: str, direction: str, klines: List) -> Optional[BacktestResult]:
    """단일 종목 최적 EMA 조합 탐색"""
    closes = [float(k[4]) for k in klines]
    
    interval = CFG["107_INTERVAL"]
    backtest_days = CFG["105_BACKTEST_DAYS"]
    bars_per_day = get_bars_per_day(interval)
    backtest_bars = backtest_days * bars_per_day
    
    if len(closes) < backtest_bars + 100:
        log.error(f"{symbol}: insufficient data")
        return None
    
    # 최근 N일만 사용
    closes_trimmed = closes[-backtest_bars:]
    
    best_result = None
    best_score = -999999
    
    fast_range = range(CFG["102_FAST_RANGE"][0], CFG["102_FAST_RANGE"][1] + 1)
    mid_range = range(CFG["103_MID_RANGE"][0], CFG["103_MID_RANGE"][1] + 1)
    exit_range = range(CFG["104_EXIT_RANGE"][0], CFG["104_EXIT_RANGE"][1] + 1)
    
    for fast in fast_range:
        for mid in mid_range:
            if mid <= fast:
                continue
            if mid - fast < 3:
                continue
            
            for exit_ema in exit_range:
                if direction == "SHORT":
                    trades, mdd, net_return_ratio = backtest_short(closes_trimmed, fast, mid, exit_ema, backtest_days)
                else:
                    trades, mdd, net_return_ratio = backtest_long(closes_trimmed, fast, mid, exit_ema, backtest_days)
                
                if not trades:
                    continue
                
                # 백테스트에서 이미 수수료 복리 반영됨
                net_return_pct = net_return_ratio * 100
                
                trades_per_day = len(trades) / backtest_days
                
                wins = sum(1 for t in trades if t.pnl_pct > 0)
                win_rate = (wins / len(trades)) * 100 if trades else 0
                
                recent_2d = calculate_recent_2d_price_change(closes, interval)
                
                score = calculate_score(net_return_pct, mdd * 100, trades_per_day)
                
                if score > best_score:
                    best_score = score
                    best_result = BacktestResult(
                        symbol=symbol,
                        fast=fast,
                        mid=mid,
                        exit_ema=exit_ema,
                        net_return=net_return_pct,
                        mdd=mdd * 100,
                        total_trades=len(trades),
                        trades_per_day=trades_per_day,
                        win_rate=win_rate,
                        recent_2d_price_change=recent_2d,
                        score=score,
                        trades=trades
                    )
    
    return best_result

# ============================================================
# SHOCK REGIME (5m 고정)
# ============================================================

def check_shock() -> bool:
    """SHOCK 판정 (5m 고정, BTC 4H 변동폭 OR ATR+Volume)"""
    try:
        # 조건 A: BTC 4H 변동폭
        klines_4h = fetch_klines("BTCUSDT", "4h", 1)
        if klines_4h and len(klines_4h) >= 1:
            k = klines_4h[-1]
            high = float(k[2])
            low = float(k[3])
            range_pct = (high - low) / low * 100
            if range_pct >= CFG["122_SHOCK_4H_RANGE_PCT"]:
                return True
        
        # 조건 B+C: 5m ATR 폭증 AND 거래대금 급증 (5m 고정)
        klines_5m = fetch_klines("BTCUSDT", "5m", 8)
        if klines_5m and len(klines_5m) > 288 * 7:
            closes = [float(k[4]) for k in klines_5m]
            highs = [float(k[2]) for k in klines_5m]
            lows = [float(k[3]) for k in klines_5m]
            
            # ATR 현재 vs 7일 평균 (5m 고정)
            atr_now = atr(highs, lows, closes, 14)
            atr_7d_list = []
            for i in range(0, 288*7, 288):  # 5m = 288 per day
                if i + 288 <= len(highs):
                    atr_7d_list.append(atr(highs[i:i+288], lows[i:i+288], closes[i:i+288], 14))
            
            if len(atr_7d_list) > 0:
                atr_7d_avg = sum(atr_7d_list) / len(atr_7d_list)
                
                if atr_7d_avg > 0 and atr_now >= atr_7d_avg * CFG["120_SHOCK_ATR_MULTIPLIER"]:
                    # 거래대금 체크
                    klines_1h = fetch_klines("BTCUSDT", "1h", 8)
                    if klines_1h and len(klines_1h) > 24 * 7:
                        volumes_1h = [float(k[5]) for k in klines_1h]
                        vol_now = volumes_1h[-1]
                        vol_7d_avg = sum(volumes_1h[-24*7:]) / (24 * 7)
                        
                        if vol_now >= vol_7d_avg * CFG["121_SHOCK_VOLUME_MULTIPLIER"]:
                            return True
        
        return False
    except Exception as e:
        log.error(f"check_shock: {e}")
        return False

# ============================================================
# EMAIL GENERATION
# ============================================================

def format_table_row(rank: int, result: BacktestResult) -> str:
    """테이블 행 생성"""
    wr_flag = ""
    if result.win_rate < 35:
        wr_flag = "⚠"
    elif result.win_rate > 55:
        wr_flag = "★"
    
    return (
        f"{rank} | {result.symbol} | {result.fast} | {result.mid} | {result.exit_ema} | "
        f"{result.net_return:.2f}% | {result.mdd:.2f}% | {result.recent_2d_price_change:.2f}% | "
        f"{result.win_rate:.1f}%{wr_flag} | {result.total_trades} | {result.score:.2f}"
    )

def generate_email_body(long_results: List[BacktestResult], short_results: List[BacktestResult], shock: bool) -> str:
    """이메일 본문 생성"""
    body = []
    
    # 상단 경고
    body.append("⚠ 본 리포트는 일봉 마감(09:00 KST) 전 데이터 기준입니다.\n")
    
    if shock:
        body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        body.append("⚠ 현재 시장은 SHOCK 상태입니다.")
        body.append("변동성 급증 구간으로 포지션 강도 축소 권고.")
        body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # LONG TOP 10
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("[LONG TOP 10]")
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("순위 | 종목 | FAST | MID | EXIT | 7일Net | 7일MDD | 2일가격변동 | 승률 | 트레이드 | 점수")
    body.append("─────────────────────────────────────")
    
    for i, result in enumerate(long_results[:10], 1):
        body.append(format_table_row(i, result))
    
    body.append("")
    
    # SHORT TOP 10
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("[SHORT TOP 10]")
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("순위 | 종목 | FAST | MID | EXIT | 7일Net | 7일MDD | 2일가격변동 | 승률 | 트레이드 | 점수")
    body.append("─────────────────────────────────────")
    
    for i, result in enumerate(short_results[:10], 1):
        body.append(format_table_row(i, result))
    
    body.append("")
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("VELLA_v9_OPTIMIZER v2.5")
    body.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    return "\n".join(body)

def send_email(subject: str, body: str):
    """AWS SES 이메일 발송"""
    try:
        smtp_user = os.getenv("AWS_SES_SMTP_USER")
        smtp_pass = os.getenv("AWS_SES_SMTP_PASS")
        
        if not smtp_user or not smtp_pass:
            log.error("AWS SES credentials not found in environment")
            return
        
        msg = MIMEMultipart()
        msg['From'] = CFG["131_EMAIL_FROM"]
        msg['To'] = CFG["130_EMAIL_TO"]
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
# MAIN
# ============================================================

def main():
    log.info("VELLA_v9_OPTIMIZER START")
    
    # 데이터 조회
    total_days = CFG["105_BACKTEST_DAYS"] + CFG["106_FETCH_EXTRA_DAYS"]
    interval = CFG["107_INTERVAL"]
    
    long_results = []
    short_results = []
    
    # LONG 종목 최적화
    for symbol in CFG["100_LONG_SYMBOLS"]:
        log.info(f"Optimizing LONG: {symbol}")
        klines = fetch_klines(symbol, interval, total_days)
        if klines:
            result = optimize_symbol(symbol, "LONG", klines)
            if result:
                long_results.append(result)
    
    # SHORT 종목 최적화
    for symbol in CFG["101_SHORT_SYMBOLS"]:
        log.info(f"Optimizing SHORT: {symbol}")
        klines = fetch_klines(symbol, interval, total_days)
        if klines:
            result = optimize_symbol(symbol, "SHORT", klines)
            if result:
                short_results.append(result)
    
    # 정렬
    long_results.sort(key=lambda x: x.score, reverse=True)
    short_results.sort(key=lambda x: x.score, reverse=True)
    
    # SHOCK 판정
    shock = check_shock()
    
    # 이메일 생성
    subject = f"VELLA OPTIMIZER Report - {datetime.now().strftime('%Y-%m-%d')}"
    body = generate_email_body(long_results, short_results, shock)
    
    # 이메일 발송
    send_email(subject, body)
    
    log.info("VELLA_v9_OPTIMIZER DONE")

if __name__ == "__main__":
    main()