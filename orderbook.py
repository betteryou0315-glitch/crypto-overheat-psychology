"""오더북 호가 불균형으로 과열 이후 폭발 방향을 가를 수 있는가
가설 (미리 정함): 매수 쪽이 두꺼울수록(불균형 +) 위로 터진다 → 순위 상관 '양수' 기대
- 대상: data(기존 15종목) + data_holdout(검증 15종목) 두 곳 동시에 테스트
- 사건: 각 폴더의 direction_events.csv (direction.py 결과)
- 데이터: bookDepth (약 30초 간격, 현재가 ±1~5% 누적 주문 금액) — 사건이 있는 날만 다운로드
- 조건 (신호 시점 직전 15분 평균):
    불균형 ±k% = (매수 금액 - 매도 금액) / (매수 + 매도),  k = 1, 2, 5
    불균형 변화 ±2% = 지금 - 1시간 전 (매도 벽이 빠지는 중인가)
- 판정: 기대 부호(+) & 월 단위 군집 t >= 2 를 두 폴더 모두에서 만족하면 통과
사용법: python orderbook.py
"""
import io, sys, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
FEATS = ["불균형 ±1%", "불균형 ±2%", "불균형 ±5%", "불균형 변화 ±2% (1h)"]

def cache_path(folder, sym, day):
    return Path(folder) / sym / "bookdepth" / f"{day}.csv.gz"

def fetch(args):
    folder, sym, day = args
    p = cache_path(folder, sym, day)
    if p.exists():
        return "cached"
    try:
        r = requests.get(f"{BASE}/{sym}/{sym}-bookDepth-{day}.zip", timeout=60)
    except requests.RequestException:
        return "error"
    if r.status_code != 200:
        return "missing"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["timestamp", "percentage", "notional"])
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, compression="gzip")
    return "ok"

def imbalance(snap, k):
    b, a = snap.get(-k, np.nan), snap.get(k, np.nan)
    return (b - a) / (b + a)

def window_mean(book, t0, t1):
    w = book[(book.timestamp > t0) & (book.timestamp <= t1)]
    return w.groupby("percentage")["notional"].mean() if len(w) else pd.Series(dtype=float)

def rank_t(d, f):
    d = d.dropna(subset=[f])
    if len(d) < 20:
        return len(d), 0, np.nan, np.nan
    x = d[f].rank().values; y = d["점수"].rank().values
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = float(np.mean(x * y)); e = y - r * x
    g = pd.Series(x * e).groupby(d["time"].dt.to_period("M").values).sum().values
    return len(d), len(g), r, r / (np.sqrt((g**2).sum()) / (x**2).sum())

all_rows, results = [], {}
for folder, label in FOLDERS:
    p = Path(folder) / "direction_events.csv"
    if not p.exists():
        print(f"{p} 없음 — 건너뜀"); continue
    ev = pd.read_csv(p, parse_dates=["time"])
    ev["점수"] = ev["결과"].map({"위": 1, "아래": -1}).fillna(0)

    need = set()
    for s, t in zip(ev.symbol, ev.time):
        for tt in (t, t - pd.Timedelta("75min")):
            need.add((folder, s, tt.strftime("%Y-%m-%d")))
    print(f"[{label}] 사건 {len(ev)}건 → 필요한 종목-날짜 {len(need)}개 다운로드 중...", flush=True)
    with ThreadPoolExecutor(16) as ex:
        st = pd.Series(list(ex.map(fetch, sorted(need)))).value_counts()
    print("   ", dict(st))

    rows = []
    for sym, g in ev.groupby("symbol"):
        files = sorted((Path(folder) / sym / "bookdepth").glob("*.csv.gz"))
        if not files:
            continue
        book = pd.concat([pd.read_csv(f) for f in files])
        book["timestamp"] = pd.to_datetime(book["timestamp"])
        book = book.sort_values("timestamp")
        for _, e in g.iterrows():
            t = e.time
            now = window_mean(book, t - pd.Timedelta("15min"), t)
            ago = window_mean(book, t - pd.Timedelta("75min"), t - pd.Timedelta("60min"))
            r = dict(symbol=sym, time=t, 결과=e.결과, 점수=e.점수, ret24=e.ret24)
            for k in (1, 2, 5):
                r[f"불균형 ±{k}%"] = imbalance(now, k) if len(now) else np.nan
            r["불균형 변화 ±2% (1h)"] = (imbalance(now, 2) - imbalance(ago, 2)) if len(now) and len(ago) else np.nan
            rows.append(r)
    d = pd.DataFrame(rows).sort_values("time")
    d.to_csv(Path(folder) / "orderbook_events.csv", index=False)
    results[label] = d
    print(f"    오더북 매칭된 사건: {d['불균형 ±2%'].notna().sum()} / {len(d)}\n")

pd.set_option("display.width", 200)
print("=== 순위 상관 (기대: 양수) — 두 데이터 모두 군집t >= 2 면 통과 ===")
tab = []
for f in FEATS:
    ts = []
    for label, d in results.items():
        n, nm, r, tc = rank_t(d, f)
        ts.append(tc)
        tab.append(dict(조건=f, 데이터=label, 건수=n, 월수=nm, 순위상관=round(r, 3), 군집t=round(tc, 2), 판정=""))
    ok = len(ts) == 2 and all(t >= 2 for t in ts)
    same = len(ts) == 2 and all(t > 0 for t in ts)
    tab[-1]["판정"] = "통과" if ok else ("방향만 맞음" if same else "실패")
print(pd.DataFrame(tab).to_string(index=False))

print("\n=== 3분위 (불균형 ±2%, 각 데이터 안에서 나눔) ===")
for label, d in results.items():
    x = d.dropna(subset=["불균형 ±2%"])
    if len(x) < 30:
        continue
    x = x.assign(구간=pd.qcut(x["불균형 ±2%"], 3, labels=["매도 두꺼움", "중간", "매수 두꺼움"]))
    t = x.groupby("구간", observed=True).agg(
        건수=("결과", "size"), 불균형평균=("불균형 ±2%", "mean"),
        위로=("결과", lambda s: (s == "위").mean()), 아래로=("결과", lambda s: (s == "아래").mean()),
        중앙24h=("ret24", "median"))
    t["위-아래"] = (t.위로 - t.아래로).map(lambda v: f"{v:+.0%}")
    t["위로"] = t.위로.map(lambda v: f"{v:.0%}"); t["아래로"] = t.아래로.map(lambda v: f"{v:.0%}")
    t["중앙24h"] = t.중앙24h.map(lambda v: f"{v:+.2%}"); t["불균형평균"] = t.불균형평균.map(lambda v: f"{v:+.3f}")
    print(f"[{label}]"); print(t.to_string()); print()