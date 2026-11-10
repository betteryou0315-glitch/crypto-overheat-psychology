"""폭발 예측력 분리: RSI(가격) 몫 vs 펀딩비+OI(레버리지 심리) 몫
그룹 (각각 연속 신호는 첫 봉만 = 독립 사건)
  - RSI만   : RSI>=70, 그러나 (펀딩비>=0.02% & OI 1시간 증가)는 아님
  - 심리만  : 펀딩비>=0.02% & OI 1시간 증가, 그러나 RSI<70
  - 셋 다   : 지금까지 쓴 과열 신호
폭발 = 다음 24시간 안에 ±5% 어느 쪽이든 닿음
[1] 각 그룹 vs 같은 변동성(4시간·24시간, 종목별 10분위)의 평범한 봉
[2] 핵심: '셋 다' vs 같은 변동성 분위의 'RSI만' 사건 → 남는 차이 = 펀딩비+OI 몫
판정: [2]의 초과가 양수 + 월 단위 군집 t >= 2 를 두 데이터 모두에서 → 레버리지 심리가 추가 정보를 줌
사용법: python vol_filter2.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
MOVE, H, COOLDOWN = 0.05, 96, 16
VOLS = {"4시간": 16, "24시간": 96}
GROUPS = ["RSI만", "심리만", "셋 다"]

def first_of_episode(s):
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return s & ~prev

def cluster_t(x, times):
    m = x.mean()
    g = pd.Series(x - m).groupby(times.dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x))

pd.set_option("display.width", 200)
verdict = []
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split() if (root / s / "klines_15m.csv").exists()]
    bars = []
    for s in syms:
        bt.D = root / s
        k = bt.load()
        rsi_hot = bt.rsi(k["close"], bt.P["rsi_len"]) >= bt.P["rsi_min"]
        psy = ((k["funding"] >= bt.P["funding_min"]) & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
        raw = {"RSI만": rsi_hot & ~psy, "심리만": psy & ~rsi_hot, "셋 다": rsi_hot & psy}
        r = np.log(k["close"]).diff()
        hmax = k["high"].rolling(H).max().shift(-H)
        lmin = k["low"].rolling(H).min().shift(-H)
        big = ((hmax / k["close"] - 1 >= MOVE) | (lmin / k["close"] - 1 <= -MOVE)).astype(float)
        big[hmax.isna()] = np.nan
        df = pd.DataFrame({"symbol": s, "t": k["t"], "big": big,
                           "평범": ~(rsi_hot | psy)})
        for g in GROUPS:
            df[g] = first_of_episode(raw[g].fillna(False))
        for name, n in VOLS.items():
            df[name] = pd.qcut(r.rolling(n).std().rank(method="first"), 10, labels=range(1, 11)).astype(float)
        bars.append(df.dropna(subset=["big"] + list(VOLS)))
    allb = pd.concat(bars, ignore_index=True)

    print(f"\n==================== {label} ====================")
    rows = []
    for name in VOLS:
        exp = allb[allb.평범].groupby(["symbol", name])["big"].mean().rename("기대")
        for g in GROUPS:
            e = allb[allb[g]].join(exp, on=["symbol", name])
            m, t = cluster_t((e.big - e.기대).values, e.t)
            rows.append(dict(매칭=f"{name} 변동성", 그룹=g, 건수=len(e), 폭발=f"{e.big.mean():.1%}",
                             평범한봉_기대=f"{e.기대.mean():.1%}", 초과=f"{m:+.1%}p", 군집t=round(t, 2)))
    print("[1] 각 그룹 vs 같은 변동성의 평범한 봉 (펀딩비·OI·RSI 어느 것도 해당 안 되는 봉)")
    print(pd.DataFrame(rows).to_string(index=False))

    rows = []
    for name in VOLS:
        # RSI만 사건의 분위별 폭발 확률 (종목 섞어서 분위별로 — 칸마다 사건이 적어서)
        exp = allb[allb["RSI만"]].groupby(name)["big"].mean().rename("기대")
        e = allb[allb["셋 다"]].join(exp, on=name)
        m, t = cluster_t((e.big - e.기대).values, e.t)
        rows.append(dict(매칭=f"{name} 변동성", 셋다_폭발=f"{e.big.mean():.1%}",
                         같은변동성_RSI만=f"{e.기대.mean():.1%}", 펀딩비OI_몫=f"{m:+.1%}p", 군집t=round(t, 2)))
        verdict.append(t)
    print("\n[2] 핵심: '셋 다' vs 같은 변동성의 'RSI만' → 남는 차이가 펀딩비+OI(레버리지 심리)의 몫")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n최종 판정:", "통과 — 레버리지 심리가 RSI·변동성 이상의 정보를 줌" if len(verdict) == 4 and all(t >= 2 for t in verdict)
      else "방향만 일치 (약함)" if len(verdict) == 4 and all(t > 0 for t in verdict) else "실패 — 추가 정보 대부분이 RSI(가격) 몫")
