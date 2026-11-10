"""약한 단서 합치기: '레버리지 쏠림 점수'
- direction.py → data/direction_events.csv, basis.py → data/basis_events.csv 를 먼저 실행해둔 상태
- 가설: 쏠림 점수가 높을수록 (개미 롱 쏠림 + 선물 프리미엄 높음) 아래로 터진다 → 순위 상관 '음수' 기대
- 합치는 방식: 가중치를 맞추지 않고 단순 평균 (결과를 보고 가중치를 고르면 과최적화)
  A안 = (개미 롱숏 z + 프리미엄 z) / 2
  B안 = (개미 롱숏 z + 프리미엄 z + 프리미엄 수준 z) / 3
    · 프리미엄 수준은 종목 간 크기가 달라서 '앞쪽 절반'의 평균/표준편차로만 표준화
사용법: python combine.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
d1 = pd.read_csv(ROOT / "direction_events.csv", parse_dates=["time"])
d2 = pd.read_csv(ROOT / "basis_events.csv", parse_dates=["time"])
ev = d1.merge(d2[["symbol", "time", "프리미엄 z점수", "프리미엄 수준(%)"]], on=["symbol", "time"])
ev = ev.sort_values("time").reset_index(drop=True)
ev["점수"] = ev["결과"].map({"위": 1, "아래": -1}).fillna(0)
mid = ev["time"].iloc[len(ev) // 2]
front = ev.time < mid

lvl = ev["프리미엄 수준(%)"]
ev["프리미엄 수준 z"] = (lvl - lvl[front].mean()) / lvl[front].std()
ev["A안: 개미+프리미엄z"] = ev[["개미 롱숏비율", "프리미엄 z점수"]].mean(axis=1, skipna=False)
ev["B안: +프리미엄 수준"] = ev[["개미 롱숏비율", "프리미엄 z점수", "프리미엄 수준 z"]].mean(axis=1, skipna=False)

parts = ["개미 롱숏비율", "프리미엄 z점수", "프리미엄 수준 z"]
combos = ["A안: 개미+프리미엄z", "B안: +프리미엄 수준"]
print(f"사건 {len(ev)}건 (두 파일 매칭)\n")

print("=== [0] 단서끼리 상관 — 낮을수록 서로 다른 정보 (합치는 효과 큼) ===")
print(ev[parts].corr(method="spearman").round(2).to_string(), "\n")

def rank_t(d, f):
    d = d.dropna(subset=[f])
    r = d[f].rank().corr(d["점수"].rank())
    return len(d), r, r * np.sqrt((len(d) - 2) / (1 - r**2))

print("=== [1] 기간 분할 순위 상관 — 음수이고 |t| 클수록 가설 지지 ===")
rows = []
for f in parts + combos:
    for name, d in [("전체", ev), ("앞쪽", ev[front]), ("뒤쪽", ev[~front])]:
        n, r, t = rank_t(d, f)
        rows.append(dict(조건=f, 기간=name, 건수=n, 순위상관=round(r, 3), t값=round(t, 2)))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False), "\n")

print("=== [2] 합친 점수 3분위 (경계는 앞쪽에서 결정, 뒤쪽에 그대로 적용) ===")
for f in combos:
    cuts = [-np.inf, ev.loc[front, f].quantile(1/3), ev.loc[front, f].quantile(2/3), np.inf]
    tbl = []
    for name, d in [("앞쪽", ev[front]), ("뒤쪽", ev[~front])]:
        d = d.dropna(subset=[f]).assign(구간=lambda x: pd.cut(x[f], cuts, labels=["하위", "중간", "상위"]))
        for g, gd in d.groupby("구간", observed=True):
            up, dn = (gd.결과 == "위").mean(), (gd.결과 == "아래").mean()
            tbl.append(dict(기간=name, 구간=g, 건수=len(gd), 위로=f"{up:.0%}", 아래로=f"{dn:.0%}",
                            위_아래=f"{up-dn:+.0%}", 중앙24h=f"{gd.ret24.median():+.2%}"))
    print(f"[{f}]")
    print(pd.DataFrame(tbl).to_string(index=False), "\n")

ev.to_csv(ROOT / "combined_events.csv", index=False)