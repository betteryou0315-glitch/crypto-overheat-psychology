"""시장 레짐(큰 흐름)으로 과열 이후 폭발 방향을 가를 수 있는가
- combine.py 가 저장한 data/combined_events.csv 사용 (사건 + A안 쏠림 점수)
- 가설: 큰 흐름이 위일수록 같은 과열도 위로 터진다 → 순위 상관 '양수' 기대
레짐 조건 (신호 시점까지의 과거 데이터만 사용):
  - 코인 50일선 괴리 : 종가가 50일 이동평균보다 몇 % 위인가 (코인 자체의 큰 추세)
  - 코인 7일 수익률
  - BTC 30일 수익률   : 시장 전체 분위기
  - BTC 7일 수익률
  - 코인-BTC 7일 차이 : 시장보다 강한 코인인가 (상대강도)
C안(미리 정한 합산) = 평균( -A안 쏠림점수, 코인 50일선 괴리, BTC 30일 수익률 )
  · 각각 앞쪽 절반의 평균/표준편차로 표준화 / 가중치 없음
  · 쏠림은 '높을수록 아래'라 부호를 뒤집어서 셋 다 '높을수록 위' 방향으로 맞춤
사용법: python regime.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
DAY = 96
ev = pd.read_csv(ROOT / "combined_events.csv", parse_dates=["time"])
A = "A안: 개미+프리미엄z"

def feats(sym_dir):
    bt.D = sym_dir
    k = bt.load()[["t", "close"]].set_index("t")["close"]
    return pd.DataFrame({
        "ma50": k / k.rolling(50 * DAY, min_periods=20 * DAY).mean() - 1,
        "r7": k / k.shift(7 * DAY) - 1,
        "r30": k / k.shift(30 * DAY) - 1,
    })

btc_dir = ROOT / "BTCUSDT"
if not (btc_dir / "klines_15m.csv").exists():
    sys.exit("data/BTCUSDT 가 필요해요")
btc = feats(btc_dir)

parts = []
for sym, g in ev.groupby("symbol"):
    f = feats(ROOT / sym)
    g = g.set_index("time")
    g["코인 50일선 괴리"] = f["ma50"].reindex(g.index).values
    g["코인 7일 수익률"] = f["r7"].reindex(g.index).values
    g["BTC 30일 수익률"] = btc["r30"].reindex(g.index).values
    g["BTC 7일 수익률"] = btc["r7"].reindex(g.index).values
    g["코인-BTC 7일 차이"] = g["코인 7일 수익률"] - g["BTC 7일 수익률"]
    parts.append(g.reset_index())
ev = pd.concat(parts).sort_values("time").reset_index(drop=True)

REG = ["코인 50일선 괴리", "코인 7일 수익률", "BTC 30일 수익률", "BTC 7일 수익률", "코인-BTC 7일 차이"]
mid = ev["time"].iloc[len(ev) // 2]
front = ev.time < mid

def zf(s):
    return (s - s[front].mean()) / s[front].std()

ev["C안: 쏠림반대+추세+BTC"] = pd.concat(
    [zf(-ev[A]), zf(ev["코인 50일선 괴리"]), zf(ev["BTC 30일 수익률"])], axis=1).mean(axis=1, skipna=False)

def rank_t(d, f):
    """순위 상관 + '월 단위 군집' 보정 t값
    천천히 변하는 레짐 값은 같은 주의 사건들이 거의 같은 값을 가져서 일반 t값이 부풀려짐
    → 같은 달(month)의 사건을 한 덩어리로 보고 표준오차를 계산 (cluster-robust)"""
    d = d.dropna(subset=[f])
    if len(d) < 10 or d[f].std() == 0:
        return len(d), np.nan, np.nan, np.nan, 0
    x = d[f].rank().values; y = d["점수"].rank().values
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = float(np.mean(x * y))
    t_naive = r * np.sqrt((len(d) - 2) / (1 - r**2))
    e = y - r * x
    wk = d["time"].dt.to_period("M").values
    g = pd.Series(x * e).groupby(wk).sum().values
    se = np.sqrt((g**2).sum()) / (x**2).sum()
    return len(d), r, t_naive, r / se, len(g)

pd.set_option("display.width", 200)
print(f"사건 {len(ev)}건\n")
print("=== [0] 쏠림 점수(A안)와의 상관 — 0에 가까울수록 새로운 정보 ===")
print(ev[[A] + REG].corr(method="spearman")[A].drop(A).round(2).to_string(), "\n")

print("=== [1] 기간 분할 순위 상관 — 레짐은 양수 기대 / A안은 음수 기대 / C안은 양수 기대 ===")
print("※ 판단은 '군집t' 기준 (같은 달 사건을 묶어 보정한 값). 일반t는 참고용\n")
rows = []
for f in REG + [A, "C안: 쏠림반대+추세+BTC"]:
    for name, d in [("전체", ev), ("앞쪽", ev[front]), ("뒤쪽", ev[~front])]:
        n, r, t, tc, nw = rank_t(d, f)
        rows.append(dict(조건=f, 기간=name, 건수=n, 월수=nw, 순위상관=round(r, 3),
                         일반t=round(t, 2), 군집t=round(tc, 2)))
print(pd.DataFrame(rows).to_string(index=False), "\n")

print("=== [2] 3분위 (경계는 앞쪽에서 결정) ===")
for f in ["코인 50일선 괴리", "BTC 30일 수익률", "C안: 쏠림반대+추세+BTC"]:
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

ev.to_csv(ROOT / "regime_events.csv", index=False)