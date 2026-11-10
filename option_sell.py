"""변동성 매도 + 과열 필터 백테스트 (BTC, ETH 하루짜리 ATM 스트래들 매도)
- 매일 08:00 UTC 에 하루짜리 스트래들(같은 행사가 콜+풋)을 팔고 다음 날 08:00 만기까지 보유
- 손익 (기초자산 가격 대비 %) = 옵션값 - |24시간 수익률| - 비용
    옵션값 ≈ 0.798 × IV × √(1/365),  IV = 진입 시점 DVOL (30일 내재변동성 지수)
- 비교 (미리 정함):
    A. 매일 매도
    B. 자기 코인 필터 : 직전 24시간 안에 그 코인에 과열 신호(펀딩비>=0.02% & RSI>=70 & OI 1h 증가)가 있었으면 쉼
    C. 시장 전체 필터: 직전 24시간 안에 상위 15개 코인 중 하나라도 과열 신호가 있었으면 쉼
한계: DVOL은 30일 옵션 기준 (하루짜리 실제 IV와 다름), 증거금·청산 위험·호가 깊이 미반영
사용법: python option_sell.py   (먼저 options_check.py 로 dvol 파일이 받아져 있어야 함)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path("data")
COST = 0.001                       # 하루 거래 비용 (기초자산 대비 0.1%) — 바꿔서 테스트 가능
COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
LOOKBACK = pd.Timedelta("24h")

def overheat_series(sym_dir):
    bt.D = sym_dir
    k = bt.load()
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    return pd.Series(sig.values, index=k["t"]), k

def recent_signal(sig, times):
    """각 진입 시각 직전 24시간 [t-24h, t) 안에 신호가 있었는가"""
    cs = sig.astype(int).cumsum()
    a = cs.reindex(times - LOOKBACK, method="ffill").fillna(0).values
    b = cs.reindex(times - pd.Timedelta("15min"), method="ffill").fillna(0).values
    return pd.Series(b - a > 0, index=times)

# 시장 전체 과열 (상위 15개 코인)
syms = (ROOT / "symbols.txt").read_text().split()
market = None
for s in syms:
    if not (ROOT / s / "klines_15m.csv").exists():
        continue
    sig, _ = overheat_series(ROOT / s)
    market = sig if market is None else market.reindex(market.index.union(sig.index), fill_value=False) | \
             sig.reindex(market.index.union(sig.index), fill_value=False)

def stats(p):
    if len(p) == 0:
        return {}
    eq = p.cumsum()
    worst5 = np.sort(p.values)[:max(1, len(p) // 100)].mean()
    return dict(일수=len(p), 누적=f"{p.sum():+.1%}", 하루평균=f"{p.mean():+.3%}", 승률=f"{(p > 0).mean():.0%}",
                최악의날=f"{p.min():+.2%}", 최악1퍼센트평균=f"{worst5:+.2%}",
                최대낙폭=f"{(eq - eq.cummax()).min():+.1%}",
                샤프=round(p.mean() / p.std() * np.sqrt(365), 2) if p.std() > 0 else np.nan)

pd.set_option("display.width", 220)
for cur, sym in COINS.items():
    dv = pd.read_csv(ROOT / f"dvol_{cur}.csv", parse_dates=["t"]).astype({"t": "datetime64[ns]"}).set_index("t")["dvol"]
    own, k = overheat_series(ROOT / sym)
    px = pd.Series(k["close"].values, index=k["t"])
    entries = px.index[(px.index.hour == 8) & (px.index.minute == 0)]
    entries = entries[entries + pd.Timedelta("24h") <= px.index[-1]]
    d = pd.DataFrame(index=entries)
    d["iv"] = dv.reindex(entries, method="ffill", tolerance=pd.Timedelta("2h")).values
    d["ret"] = px.reindex(entries + pd.Timedelta("24h")).values / px.reindex(entries).values - 1
    d = d.dropna()
    d["premium"] = 0.798 * d["iv"] / 100 * np.sqrt(1 / 365)
    d["pnl"] = d["premium"] - d["ret"].abs() - COST
    d["자기과열"] = recent_signal(own, d.index).values
    d["시장과열"] = recent_signal(market, d.index).values

    print(f"\n==================== {cur} 하루짜리 스트래들 매도 (비용 {COST:.1%}/일) ====================")
    rows = [dict(전략="A. 매일 매도", **stats(d.pnl)),
            dict(전략="B. 자기 코인 필터", **stats(d.pnl[~d.자기과열])),
            dict(전략="C. 시장 전체 필터", **stats(d.pnl[~d.시장과열]))]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n[쉰 날 vs 매도한 날] — 쉰 날이 실제로 손실이 큰 날이었나")
    rows = []
    for name, col in [("B 자기 코인", "자기과열"), ("C 시장 전체", "시장과열")]:
        skip, keep = d[d[col]], d[~d[col]]
        rows.append(dict(필터=name, 쉰날=len(skip), 쉰날_평균손익=f"{skip.pnl.mean():+.3%}",
                         매도한날_평균손익=f"{keep.pnl.mean():+.3%}",
                         쉰날_변동폭=f"{skip.ret.abs().mean():.2%}", 매도한날_변동폭=f"{keep.ret.abs().mean():.2%}",
                         쉰날_IV=f"{skip.iv.mean():.0f}%", 매도한날_IV=f"{keep.iv.mean():.0f}%",
                         최악10일중_쉰날=int(d.nsmallest(10, 'pnl')[col].sum())))
    print(pd.DataFrame(rows).to_string(index=False))
    d.to_csv(ROOT / f"option_sell_{cur}.csv")