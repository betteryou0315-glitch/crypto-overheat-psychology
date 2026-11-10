"""과열 신호의 '폭발 예측력'은 심리 덕분인가, 그냥 최근에 많이 움직여서인가?
방법: 같은 변동성끼리 비교 (매칭)
  1) 모든 15분봉의 '최근 변동성'을 계산
       - 4시간 변동성 : 최근 16봉 15분 수익률의 표준편차
       - 24시간 변동성: 최근 96봉
  2) 종목마다 변동성을 10분위로 나눔 (1 = 가장 잠잠, 10 = 가장 요동)
  3) 같은 분위 안에서 '과열 신호가 없는 봉'의 폭발 확률 = 기대치
  4) 과열 사건의 실제 폭발 여부 - 기대치 = 초과 폭발 확률
  폭발 = 다음 24시간 안에 ±5% 어느 쪽이든 닿음
판정: 초과 폭발 확률이 양수 + 월 단위 군집 t >= 2 → 심리 지표가 변동성 이상의 정보를 줌
데이터: data(기존 15종목), data_holdout(검증 15종목) 둘 다
사용법: python vol_filter.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
MOVE, H, COOLDOWN = 0.05, 96, 16
VOLS = {"4시간 변동성": 16, "24시간 변동성": 96}

def symbol_dirs(root):
    keep = set((root / "symbols.txt").read_text().split())
    return [root / s for s in sorted(keep) if (root / s / "klines_15m.csv").exists()]

def cluster_t(x, times):
    m = x.mean()
    g = pd.Series(x - m).groupby(times.dt.to_period("M").values).sum().values
    se = np.sqrt((g**2).sum()) / len(x)
    return m, m / se

pd.set_option("display.width", 200)
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        print(f"{folder} 없음"); continue
    bars, eps = [], []
    for sd in symbol_dirs(root):
        bt.D = sd
        k = bt.load()
        k["rsi"] = bt.rsi(k["close"], bt.P["rsi_len"])
        sig = ((k["funding"] >= bt.P["funding_min"]) & (k["rsi"] >= bt.P["rsi_min"])
               & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
        prev = sig.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
        ep = sig & ~prev

        r = np.log(k["close"]).diff()
        hmax = k["high"].rolling(H).max().shift(-H)
        lmin = k["low"].rolling(H).min().shift(-H)
        big = ((hmax / k["close"] - 1 >= MOVE) | (lmin / k["close"] - 1 <= -MOVE)).astype(float)
        big[hmax.isna()] = np.nan

        df = pd.DataFrame({"symbol": sd.name, "t": k["t"], "big": big, "sig": sig, "ep": ep})
        for name, n in VOLS.items():
            v = r.rolling(n).std()
            df[name] = pd.qcut(v.rank(method="first"), 10, labels=range(1, 11)).astype(float)
        df = df.dropna(subset=["big"] + list(VOLS))
        bars.append(df)

    allb = pd.concat(bars, ignore_index=True)
    base_all = allb.loc[~allb.sig, "big"].mean()
    ev = allb[allb.ep].copy()
    print(f"\n==================== {label} ====================")
    print(f"전체 봉 {len(allb):,}개 | 과열 사건 {len(ev)}건")
    print(f"폭발 확률 — 과열 신호 없는 봉 전체: {base_all:.1%} | 과열 사건: {ev.big.mean():.1%}  "
          f"(단순 비교 {ev.big.mean() / base_all:.1f}배)\n")

    summ = []
    for name in VOLS:
        exp = allb[~allb.sig].groupby(["symbol", name])["big"].mean().rename("기대")
        e = ev.join(exp, on=["symbol", name])
        e["초과"] = e["big"] - e["기대"]
        m, t = cluster_t(e["초과"].values, e["t"])
        summ.append(dict(매칭기준=name, 과열사건_폭발=f"{e.big.mean():.1%}",
                         같은변동성_기대=f"{e.기대.mean():.1%}", 초과=f"{m:+.1%}p",
                         배수=f"{e.big.mean() / e.기대.mean():.2f}배", 군집t=round(t, 2)))
    print("[요약] 같은 변동성의 평범한 봉과 비교했을 때 과열 사건이 더 자주 폭발하는가")
    print(pd.DataFrame(summ).to_string(index=False))

    name = "4시간 변동성"
    tbl = allb.groupby(name).apply(lambda g: pd.Series({
        "평범한봉_폭발": g.loc[~g.sig, "big"].mean(),
        "과열사건_건수": int(g.ep.sum()),
        "과열사건_폭발": g.loc[g.ep, "big"].mean() if g.ep.sum() else np.nan,
    }), include_groups=False)
    tbl["차이"] = tbl["과열사건_폭발"] - tbl["평범한봉_폭발"]
    for c in ["평범한봉_폭발", "과열사건_폭발", "차이"]:
        tbl[c] = tbl[c].map(lambda v: "-" if pd.isna(v) else (f"{v:+.0%}" if c == "차이" else f"{v:.0%}"))
    tbl["과열사건_건수"] = tbl["과열사건_건수"].astype(int)
    print(f"\n[{name} 10분위별] (1 = 가장 잠잠, 10 = 가장 요동)")
    print(tbl.to_string())
