"""알트코인 추세 추종 (채널 돌파, 터틀 방식) + 과열 필터 — 규칙 미리 확정
진입: 종가 > 직전 24시간(96봉) 최고가 → 롱 / 종가 < 직전 24시간 최저가 → 숏 (그 봉 종가에 진입)
청산: 롱은 직전 12시간(48봉) 최저가 이탈, 숏은 직전 12시간 최고가 돌파 시 (그 가격 또는 더 불리한 시가)
포지션: 종목당 한 번에 하나
비용: 편도 0.07% (시장가 0.05% + 슬리피지 0.02%)
비교: A 기본 / B 과열 필터 (직전 24시간 안에 그 코인에 과열 신호가 있었으면 신규 진입 안 함)
대상: data(기존 15종목), data_holdout(검증 15종목)
판정: 거래당 평균이 플러스 + 월 단위 군집 t >= 2 를 두 데이터 모두에서 → 기초 전략으로 인정
사용법: python trend.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
N_IN, N_OUT, DAY = 96, 48, 96
COST = 2 * 0.0007

def simulate(o, h, l, c, t, block):
    hi_in = pd.Series(h).rolling(N_IN).max().shift(1).values
    lo_in = pd.Series(l).rolling(N_IN).min().shift(1).values
    hi_out = pd.Series(h).rolling(N_OUT).max().shift(1).values
    lo_out = pd.Series(l).rolling(N_OUT).min().shift(1).values
    trades, side, entry, ei = [], 0, 0.0, 0
    for i in range(N_IN + 1, len(c)):
        if side == 1 and l[i] <= lo_out[i]:
            px = min(lo_out[i], o[i]); trades.append((t[ei], t[i], 1, (px - entry) / entry - COST, i - ei)); side = 0
        elif side == -1 and h[i] >= hi_out[i]:
            px = max(hi_out[i], o[i]); trades.append((t[ei], t[i], -1, (entry - px) / entry - COST, i - ei)); side = 0
        if side == 0 and not block[i]:
            if c[i] > hi_in[i]:
                side, entry, ei = 1, c[i], i
            elif c[i] < lo_in[i]:
                side, entry, ei = -1, c[i], i
    return trades

def cluster_t(x, times):
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.Series(times).dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x))

def trade_stats(d):
    if len(d) < 10:
        return dict(거래=len(d))
    r = d.ret
    m, t = cluster_t(r.values, d.entry)
    pf = r[r > 0].sum() / -r[r <= 0].sum()
    return dict(거래=len(d), 승률=f"{(r > 0).mean():.0%}", 거래당=f"{m:+.3%}", 평균이익=f"{r[r > 0].mean():+.2%}",
                평균손실=f"{r[r <= 0].mean():+.2%}", PF=round(pf, 2), 보유중앙=f"{d.bars.median() / 4:.1f}h", 군집t=round(t, 2))

pd.set_option("display.width", 220)
verdict = {}
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split() if (root / s / "klines_15m.csv").exists()]
    allt = {"A": [], "B": []}
    n_days = None
    for s in syms:
        bt.D = root / s
        k = bt.load()
        rsi = bt.rsi(k["close"], bt.P["rsi_len"])
        sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
               & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
        recent = sig.astype(int).rolling(DAY, min_periods=1).sum().shift(1).fillna(0).values > 0
        arr = [k[x].values for x in ["open", "high", "low", "close"]]
        for name, block in [("A", np.zeros(len(k), bool)), ("B", recent)]:
            for e, x, sd, r, b in simulate(*arr, k["t"].values, block):
                allt[name].append(dict(symbol=s, entry=pd.Timestamp(e), exit=pd.Timestamp(x), side=sd, ret=r, bars=b))
    print(f"\n==================== {label} (종목 {len(syms)}개) ====================")
    rows = []
    for name, title in [("A", "A. 기본"), ("B", "B. 과열 필터")]:
        d = pd.DataFrame(allt[name])
        for sub, sname in [(d, "전체"), (d[d.side == 1], "롱"), (d[d.side == -1], "숏")]:
            rows.append(dict(전략=title, 구분=sname, **trade_stats(sub)))
        # 종목 균등 배분 포트폴리오 (청산일 기준 일별 합산 / 종목 수)
        daily = d.groupby(d.exit.dt.normalize())["ret"].sum() / len(syms)
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
        eq = daily.cumsum()
        rows.append(dict(전략=title, 구분="포트폴리오", 거래=f"연 {daily.mean() * 365:+.1%}",
                         승률="", 거래당="", 평균이익=f"샤프 {daily.mean() / daily.std() * np.sqrt(365):.2f}",
                         평균손실=f"최대낙폭 {(eq - eq.cummax()).min():+.1%}"))
        if name == "A":
            m, t = cluster_t(d.ret.values, d.entry)
            verdict[label] = (m, t)
        if name == "B":
            a = pd.DataFrame(allt["A"])
            skipped = a.merge(d[["symbol", "entry"]], on=["symbol", "entry"], how="left", indicator=True)
            skipped = skipped[skipped._merge == "left_only"]
            rows.append(dict(전략="(B가 건너뛴 A의 거래)", 구분="", **trade_stats(skipped)))
    print(pd.DataFrame(rows).fillna("").to_string(index=False))

ok = len(verdict) == 2 and all(m > 0 and t >= 2 for m, t in verdict.values())
print("\n기초 전략 판정 (A 전체):", "통과" if ok else
      "방향만 맞음 (약함)" if len(verdict) == 2 and all(m > 0 for m, t in verdict.values()) else "실패")
