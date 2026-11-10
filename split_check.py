"""'큰손 vs 개미' 패턴이 기간을 나눠도 유지되는지 검증
- direction.py 가 저장한 data/direction_events.csv 사용 (먼저 direction.py 실행 필요)
- 사건을 시간순으로 앞쪽 절반 / 뒤쪽 절반으로 나눔
- 1/3 구간 경계는 '앞쪽 절반'에서만 정하고 뒤쪽에 그대로 적용 (뒤쪽 결과를 보고 경계를 고르지 않도록)
- 추가로 순위 상관(스피어만): 조건 값이 클수록 위로 터지는 경향이 있으면 +
  결과 점수: 위=+1, 아래=-1, 안 터짐/불명=0
사용법: python split_check.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
FEATS = ["큰손-개미 차이", "개미 롱숏비율", "큰손 롱숏비율"]

ev = pd.read_csv(ROOT / "direction_events.csv", parse_dates=["time"]).sort_values("time")
ev["점수"] = ev["결과"].map({"위": 1, "아래": -1}).fillna(0)
mid = ev["time"].iloc[len(ev) // 2]
halves = {"앞쪽": ev[ev.time < mid], "뒤쪽": ev[ev.time >= mid]}

for name, d in halves.items():
    print(f"{name}: {d.time.min():%Y-%m-%d} ~ {d.time.max():%Y-%m-%d}, {len(d)}건, "
          f"종목 {d.symbol.nunique()}개, 위 {(d.결과=='위').mean():.0%} / 아래 {(d.결과=='아래').mean():.0%}")
print()

def corr_t(x, y):
    m = x.notna() & y.notna()
    r = x[m].rank().corr(y[m].rank())
    n = m.sum()
    return r, r * np.sqrt((n - 2) / (1 - r**2)), n

pd.set_option("display.width", 200)
summary = []
for f in FEATS:
    first = halves["앞쪽"][f].dropna()
    cuts = [-np.inf, first.quantile(1/3), first.quantile(2/3), np.inf]
    print(f"[{f}]  (구간 경계는 앞쪽에서 결정: {cuts[1]:+.2f}, {cuts[2]:+.2f})")
    tbl = []
    for name, d in halves.items():
        d = d.dropna(subset=[f]).assign(구간=lambda x: pd.cut(x[f], cuts, labels=["하위", "중간", "상위"]))
        for g, gd in d.groupby("구간", observed=True):
            up, dn = (gd.결과 == "위").mean(), (gd.결과 == "아래").mean()
            tbl.append(dict(기간=name, 구간=g, 건수=len(gd), 위로=f"{up:.0%}", 아래로=f"{dn:.0%}",
                            위_아래=f"{up-dn:+.0%}", 중앙24h=f"{gd.ret24.median():+.2%}"))
        r, t, n = corr_t(d[f], d["점수"])
        summary.append(dict(조건=f, 기간=name, 건수=n, 순위상관=round(r, 3), t값=round(t, 2)))
    print(pd.DataFrame(tbl).to_string(index=False))
    print()

print("=== 요약: 순위 상관 (두 기간 모두 같은 부호 + |t| 2 근처 이상이면 유력) ===")
print(pd.DataFrame(summary).to_string(index=False))