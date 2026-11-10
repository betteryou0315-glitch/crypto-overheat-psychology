"""홀드아웃 최종 판정 — 미리 정한 두 가설을 규칙 변경 없이 테스트
  가설 1: A안 쏠림 점수(개미 롱숏 z + 프리미엄 z)가 높을수록 아래로 터진다 → 음수 기대
  가설 2: BTC 30일 수익률이 높을수록 (시장 전체 과열) 아래로 터진다     → 음수 기대
  (참고) BTC 7일 수익률도 같은 방향 기대
판정: 기대한 부호 + 월 단위 군집 t값 -2 이하 = 통과
사용법: python holdout_test.py  (data_holdout/regime_events.csv 필요)
"""
from pathlib import Path
import numpy as np
import pandas as pd

TESTS = [("A안: 개미+프리미엄z", "가설 1: 쏠림 높을수록 아래"),
         ("BTC 30일 수익률", "가설 2: 시장 과열일수록 아래"),
         ("BTC 7일 수익률", "(참고) BTC 7일")]

def rank_t(d, f):
    d = d.dropna(subset=[f])
    x = d[f].rank().values; y = d["점수"].rank().values
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = float(np.mean(x * y)); e = y - r * x
    g = pd.Series(x * e).groupby(d["time"].dt.to_period("M").values).sum().values
    return len(d), len(g), r, r / (np.sqrt((g**2).sum()) / (x**2).sum())

rows = []
for folder, label in [("data", "기존 15종목 (단서를 찾은 곳)"), ("data_holdout", "검증 15종목 (처음 보는 곳)")]:
    p = Path(folder) / "regime_events.csv"
    if not p.exists():
        print(f"{p} 없음"); continue
    ev = pd.read_csv(p, parse_dates=["time"])
    if label.startswith("검증"):
        print(f"검증 종목 사건 {len(ev)}건, 종목 {ev.symbol.nunique()}개 | "
              f"위 {(ev.결과=='위').mean():.0%} / 아래 {(ev.결과=='아래').mean():.0%} / "
              f"안 터짐 {(ev.결과=='안 터짐').mean():.0%}\n")
    for f, name in TESTS:
        n, nm, r, tc = rank_t(ev, f)
        verdict = ("통과" if tc <= -2 else "방향만 맞음" if r < 0 else "실패") if folder == "data_holdout" else ""
        rows.append(dict(데이터=label, 가설=name, 건수=n, 월수=nm, 순위상관=round(r, 3), 군집t=round(tc, 2), 판정=verdict))

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))

hp = Path("data_holdout") / "regime_events.csv"
if hp.exists():
    ev = pd.read_csv(hp, parse_dates=["time"])
    print("\n=== 검증 종목 3분위 (전체 기간) ===")
    for f, name in TESTS[:2]:
        d = ev.dropna(subset=[f]).assign(구간=lambda x: pd.qcut(x[f], 3, labels=["하위", "중간", "상위"]))
        t = d.groupby("구간", observed=True).agg(
            건수=("결과", "size"),
            위로=("결과", lambda x: (x == "위").mean()),
            아래로=("결과", lambda x: (x == "아래").mean()),
            중앙24h=("ret24", "median"))
        t["위-아래"] = (t.위로 - t.아래로).map(lambda v: f"{v:+.0%}")
        t["위로"] = t.위로.map(lambda v: f"{v:.0%}"); t["아래로"] = t.아래로.map(lambda v: f"{v:.0%}")
        t["중앙24h"] = t.중앙24h.map(lambda v: f"{v:+.2%}")
        print(f"[{name}]"); print(t.to_string()); print()