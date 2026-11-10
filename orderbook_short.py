"""오더북 재테스트 — 짧은 시간 범위 (미리 확정한 1회 테스트, 실패 시 오더북은 종료)
가설 그대로: 매수 쪽이 두꺼울수록(불균형 +) 위로 → 순위 상관 '양수' 기대
결과 기준만 변경: 신호 후 1시간(15분봉 4개) 안에 ±1.5% 중 먼저 닿은 쪽
- orderbook.py 가 저장한 각 폴더의 orderbook_events.csv (불균형 값) 재사용
- 15분봉 하나에서 위아래 둘 다 닿으면 '불명' (점수 0)
사용법: python orderbook_short.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

MOVE, WAIT = 0.015, 4
FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
FEATS = ["불균형 ±1%", "불균형 ±2%", "불균형 ±5%", "불균형 변화 ±2% (1h)"]

def first_touch(h, l, c, s):
    up, dn = c[s] * (1 + MOVE), c[s] * (1 - MOVE)
    for u in range(s + 1, min(s + 1 + WAIT, len(c))):
        a, b = h[u] >= up, l[u] <= dn
        if a and b: return "불명"
        if a: return "위"
        if b: return "아래"
    return "안 터짐"

def rank_t(d, f):
    d = d.dropna(subset=[f])
    x = d[f].rank().values; y = d["점수"].rank().values
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = float(np.mean(x * y)); e = y - r * x
    g = pd.Series(x * e).groupby(d["time"].dt.to_period("M").values).sum().values
    return len(d), len(g), r, r / (np.sqrt((g**2).sum()) / (x**2).sum())

results = {}
for folder, label in FOLDERS:
    ev = pd.read_csv(Path(folder) / "orderbook_events.csv", parse_dates=["time"])
    out = []
    for sym, g in ev.groupby("symbol"):
        k = pd.read_csv(Path(folder) / sym / "klines_15m.csv")
        t = pd.to_datetime(k["close_time"] + 1, unit="ms").astype("datetime64[ns]")
        pos = pd.Series(np.arange(len(k)), index=t)
        h, l, c = k.high.values, k.low.values, k.close.values
        for _, e in g.iterrows():
            s = pos.get(e.time)
            out.append(first_touch(h, l, c, int(s)) if s is not None else np.nan)
    ev = ev.sort_values(["symbol", "time"]).reset_index(drop=True)
    ev["결과1h"] = out
    ev["점수"] = ev["결과1h"].map({"위": 1, "아래": -1}).fillna(0)
    results[label] = ev
    vc = ev["결과1h"].value_counts(normalize=True)
    print(f"[{label}] {len(ev)}건 | " + ", ".join(f"{i} {v:.0%}" for i, v in vc.items()))

pd.set_option("display.width", 200)
print("\n=== 순위 상관 (기대: 양수) — 두 데이터 모두 군집t >= 2 면 통과 ===")
tab = []
for f in FEATS:
    ts = []
    for label, d in results.items():
        n, nm, r, tc = rank_t(d, f)
        ts.append(tc)
        tab.append(dict(조건=f, 데이터=label, 건수=n, 월수=nm, 순위상관=round(r, 3), 군집t=round(tc, 2), 판정=""))
    tab[-1]["판정"] = "통과" if all(t >= 2 for t in ts) else ("방향만 맞음" if all(t > 0 for t in ts) else "실패")
print(pd.DataFrame(tab).to_string(index=False))

print("\n=== 3분위 (불균형 ±2%) ===")
for label, d in results.items():
    x = d.dropna(subset=["불균형 ±2%"])
    x = x.assign(구간=pd.qcut(x["불균형 ±2%"], 3, labels=["매도 두꺼움", "중간", "매수 두꺼움"]))
    t = x.groupby("구간", observed=True).agg(
        건수=("결과1h", "size"),
        위로=("결과1h", lambda s: (s == "위").mean()), 아래로=("결과1h", lambda s: (s == "아래").mean()),
        안터짐=("결과1h", lambda s: (s == "안 터짐").mean()))
    t["위-아래"] = (t.위로 - t.아래로).map(lambda v: f"{v:+.0%}")
    for col in ["위로", "아래로", "안터짐"]:
        t[col] = t[col].map(lambda v: f"{v:.0%}")
    print(f"[{label}]"); print(t.to_string()); print()