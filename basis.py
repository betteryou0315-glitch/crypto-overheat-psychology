"""베이시스(프리미엄)로 과열 이후 폭발 방향을 가를 수 있는가
가설: 같은 과열이라도
  - 선물이 현물보다 많이 비쌈 (프리미엄 높음) → 레버리지가 끌어올린 상승 → 아래로 터지기 쉬움
  - 선물 프리미엄이 낮음 (현물이 따라오거나 주도) → 실수요 상승 → 위로 터지기 쉬움
신호: 펀딩비>=0.02%, RSI14>=70, OI 1시간 증가 (연속 신호는 첫 봉만)
결과: 신호 후 24시간 안에 +5% / -5% 중 먼저 닿은 쪽
조건 (신호 시점):
  - 프리미엄 수준       : 신호 봉의 프리미엄 (원래 값, %)
  - 프리미엄 z점수      : 그 종목 최근 7일 대비 얼마나 높은가
  - 프리미엄 1h/4h 변화 : 최근 상승 동안 선물이 현물보다 더 빨리 올랐나
검증: 전체 3분위 표 + 기간 앞/뒤 절반 순위 상관 (경계는 앞쪽에서만 결정)
사용법: python basis.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
MOVE, WAIT, Z_WIN, COOLDOWN = 0.05, 96, 672, 16
FEATS = ["프리미엄 수준(%)", "프리미엄 z점수", "프리미엄 1h 변화", "프리미엄 4h 변화"]

def symbol_dirs():
    dirs = sorted(p for p in ROOT.iterdir() if (p / "klines_15m.csv").exists())
    if (ROOT / "symbols.txt").exists():
        keep = set((ROOT / "symbols.txt").read_text().split())
        dirs = [p for p in dirs if p.name in keep]
    return dirs or [ROOT]

def first_touch(h, l, c, s):
    up, dn = c[s] * (1 + MOVE), c[s] * (1 - MOVE)
    for u in range(s + 1, min(s + 1 + WAIT, len(c))):
        a, b = h[u] >= up, l[u] <= dn
        if a and b: return "불명"
        if a: return "위"
        if b: return "아래"
    return "안 터짐"

rows = []
for sd in symbol_dirs():
    if not (sd / "premium_15m.csv").exists():
        print(f"[{sd.name}] premium_15m.csv 없음, 건너뜀"); continue
    bt.D = sd
    k = bt.load()
    p = pd.read_csv(sd / "premium_15m.csv")
    p["t"] = pd.to_datetime(p["close_time"] + 1, unit="ms").astype("datetime64[ns]")
    p = p[["t", "close"]].rename(columns={"close": "prem"})
    k = k.merge(p, on="t", how="left")
    k["prem"] = k["prem"].ffill(limit=4)

    k["프리미엄 수준(%)"] = k["prem"] * 100
    m = k["prem"].rolling(Z_WIN, min_periods=Z_WIN // 2)
    k["프리미엄 z점수"] = (k["prem"] - m.mean()) / m.std()
    k["프리미엄 1h 변화"] = (k["prem"] - k["prem"].shift(4)) * 100
    k["프리미엄 4h 변화"] = (k["prem"] - k["prem"].shift(16)) * 100

    k["rsi"] = bt.rsi(k["close"], bt.P["rsi_len"])
    sig = ((k["funding"] >= bt.P["funding_min"]) & (k["rsi"] >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = sig.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    h, l, c = k.high.values, k.low.values, k.close.values
    for s in np.flatnonzero((sig & ~prev).values):
        r = dict(symbol=sd.name, time=k.t.iloc[s], 결과=first_touch(h, l, c, s),
                 ret24=c[s + WAIT] / c[s] - 1 if s + WAIT < len(c) else np.nan)
        r.update({f: k[f].iloc[s] for f in FEATS})
        rows.append(r)

ev = pd.DataFrame(rows).sort_values("time")
ev["점수"] = ev["결과"].map({"위": 1, "아래": -1}).fillna(0)
print(f"신호 사건 {len(ev)}건 | 위 {(ev.결과=='위').mean():.0%} / 아래 {(ev.결과=='아래').mean():.0%}"
      f" | 프리미엄 값 있는 사건 {ev['프리미엄 수준(%)'].notna().sum()}건\n")

pd.set_option("display.width", 200)
print("=== [1] 전체 기간 3분위 — 위-아래가 하위→상위로 꾸준히 변하는가 ===")
for f in FEATS:
    d = ev.dropna(subset=[f])
    if len(d) < 30:
        print(f"[{f}] 데이터 부족\n"); continue
    d = d.assign(구간=pd.qcut(d[f], 3, labels=["하위", "중간", "상위"]))
    t = d.groupby("구간", observed=True).agg(
        건수=("결과", "size"), 값평균=(f, "mean"),
        위로=("결과", lambda x: (x == "위").mean()),
        아래로=("결과", lambda x: (x == "아래").mean()),
        중앙24h=("ret24", "median"))
    t["위-아래"] = (t["위로"] - t["아래로"]).map(lambda v: f"{v:+.0%}")
    t["위로"] = t["위로"].map(lambda v: f"{v:.0%}")
    t["아래로"] = t["아래로"].map(lambda v: f"{v:.0%}")
    t["중앙24h"] = t["중앙24h"].map(lambda v: f"{v:+.2%}")
    t["값평균"] = t["값평균"].map(lambda v: f"{v:+.3f}")
    print(f"[{f}]"); print(t.to_string()); print()

print("=== [2] 기간 분할 검증 — 순위 상관 (두 기간 같은 부호 + |t| 2 근처 이상이면 유력) ===")
mid = ev["time"].iloc[len(ev) // 2]
out = []
for f in FEATS:
    for name, d in [("앞쪽", ev[ev.time < mid]), ("뒤쪽", ev[ev.time >= mid])]:
        d = d.dropna(subset=[f])
        if len(d) < 30:
            continue
        r = d[f].rank().corr(d["점수"].rank())
        out.append(dict(조건=f, 기간=name, 건수=len(d), 순위상관=round(r, 3),
                        t값=round(r * np.sqrt((len(d) - 2) / (1 - r**2)), 2)))
print(pd.DataFrame(out).to_string(index=False))
ev.to_csv(ROOT / "basis_events.csv", index=False)