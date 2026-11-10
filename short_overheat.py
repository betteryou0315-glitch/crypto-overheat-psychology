"""숏 과열 탐색 — 롱 과열과 나란히 비교 (탐색용, data_new 는 절대 사용 안 함)
롱 과열 : 펀딩비 >= +0.02%, RSI >= 70, OI 1시간 증가   (지금까지 연구한 것)
숏 과열 : 펀딩비 <= -0.02%, RSI <= 30, OI 1시간 증가   (숏이 몰리고, 가격이 많이 빠졌는데, 새 포지션이 계속 들어옴)
연속 신호는 첫 봉만 (직전 4시간에 같은 신호 없을 때)
보는 것:
  1) 사건 수
  2) 폭발: 24시간 안에 ±5% 어느 쪽이든 닿은 비율 (기준 = 신호 없는 모든 봉)
  3) 방향: 24시간 안에 ±5% 중 먼저 닿은 쪽 (위 / 아래 / 안 터짐)
  4) 24시간 뒤 수익률: 평균, 중앙값, 상위 10% 제외 평균 (꼬리 착시 확인)
대상: data, data_holdout, data_final (이미 써온 45종목)
사용법: python short_overheat.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목")]
H, MOVE, COOLDOWN = 96, 0.05, 16

def first_of_episode(s):
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return s & ~prev

def first_touch(h, l, c, s):
    up, dn = c[s] * (1 + MOVE), c[s] * (1 - MOVE)
    for u in range(s + 1, min(s + 1 + H, len(c))):
        a, b = h[u] >= up, l[u] <= dn
        if a and b: return "불명"
        if a: return "위"
        if b: return "아래"
    return "안 터짐"

def cluster_t(x, times):
    if len(x) < 3:
        return np.nan
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.Series(times).dt.to_period("M").values).sum().values
    return m / (np.sqrt((g**2).sum()) / len(x))

pd.set_option("display.width", 230)
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        continue
    rows = {"롱 과열": [], "숏 과열": []}
    base_big, base_n = 0, 0
    for s in (root / "symbols.txt").read_text().split():
        if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"]):
            continue
        bt.D = root / s
        k = bt.load()
        rsi = bt.rsi(k["close"], bt.P["rsi_len"])
        oi_up = k["oi"] > k["oi"].shift(bt.P["oi_lag"])
        sig = {"롱 과열": ((k["funding"] >= bt.P["funding_min"]) & (rsi >= 70) & oi_up).fillna(False),
               "숏 과열": ((k["funding"] <= -bt.P["funding_min"]) & (rsi <= 30) & oi_up).fillna(False)}
        h, l, c = k["high"].values, k["low"].values, k["close"].values
        hmax = k["high"].rolling(H).max().shift(-H).values
        lmin = k["low"].rolling(H).min().shift(-H).values
        big = (hmax / c - 1 >= MOVE) | (lmin / c - 1 <= -MOVE)
        valid = ~np.isnan(hmax)
        anysig = (sig["롱 과열"] | sig["숏 과열"]).values
        base_big += big[valid & ~anysig].sum(); base_n += (valid & ~anysig).sum()
        for name, sg in sig.items():
            for e in np.flatnonzero(first_of_episode(sg).values):
                if e + H >= len(c):
                    continue
                rows[name].append(dict(time=k["t"].iloc[e], big=big[e], dir=first_touch(h, l, c, e),
                                       ret24=c[e + H] / c[e] - 1))
    base = base_big / base_n
    out = []
    for name, rs in rows.items():
        d = pd.DataFrame(rs)
        if d.empty:
            out.append(dict(종류=name, 사건=0)); continue
        q90 = d.ret24.quantile(0.9)
        up, dn = (d.dir == "위").mean(), (d.dir == "아래").mean()
        out.append(dict(종류=name, 사건=len(d), 폭발률=f"{d.big.mean():.0%}", 기준대비=f"{d.big.mean() / base:.2f}배",
                        위로=f"{up:.0%}", 아래로=f"{dn:.0%}", 위_아래=f"{up - dn:+.0%}",
                        안터짐=f"{(d.dir == '안 터짐').mean():.0%}",
                        평균24h=f"{d.ret24.mean():+.2%}", 중앙24h=f"{d.ret24.median():+.2%}",
                        상위10퍼제외=f"{d.ret24[d.ret24 < q90].mean():+.2%}",
                        평균t=round(cluster_t(d.ret24.values, d.time), 2)))
    print(f"\n==================== {label} (신호 없는 봉의 24시간 ±5% 폭발률: {base:.0%}) ====================")
    print(pd.DataFrame(out).fillna("").to_string(index=False))
print("\n※ 숏 과열은 드물어서 사건 수를 꼭 같이 볼 것. 위_아래 가 크게 + 이고 중앙값도 + 면 '숏 과열 → 위로' 후보")
