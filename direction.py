"""과열 이후 '어느 쪽으로 터지는가'를 가르는 조건 찾기
신호: 펀딩비>=0.02%, RSI14>=70, OI 1시간 증가 (연속 신호는 첫 봉만)
결과: 신호 후 24시간 안에 +5% / -5% 중 먼저 닿은 쪽 (= 폭발 방향)
후보 조건 (metrics 파일, 신호 시점 값):
  - 상위 트레이더 롱숏 비율 (포지션 기준)   : 큰손들이 롱에 얼마나 쏠렸나
  - 전체 계정 롱숏 비율                    : 개미들이 롱에 얼마나 쏠렸나
  - 큰손 - 개미 쏠림 차이                  : 개미만 롱에 몰렸나, 큰손도 같이 몰렸나
  - 테이커 매수/매도 비율 (최근 1시간 평균) : 시장가로 공격적으로 사는 힘
  - 펀딩비 크기
종목마다 수준이 달라서 각 조건은 '그 종목의 최근 7일 대비 z점수'로 바꿔서 비교
각 조건을 하위/중간/상위 1/3로 나눠서 위로 터진 비율 vs 아래로 터진 비율 비교
사용법: python direction.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
MOVE = 0.05       # ±5% 중 먼저 닿는 쪽
WAIT = 96         # 24시간
Z_WIN = 672       # 7일 (15분봉)
COOLDOWN = 16

FEATURES = {
    "큰손 롱숏비율": "sum_toptrader_long_short_ratio",
    "개미 롱숏비율": "count_long_short_ratio",
    "테이커 매수비율(1h)": "sum_taker_long_short_vol_ratio",
}

def symbol_dirs():
    dirs = sorted(p for p in ROOT.iterdir() if (p / "klines_15m.csv").exists())
    if (ROOT / "symbols.txt").exists():
        keep = set((ROOT / "symbols.txt").read_text().split())
        dirs = [p for p in dirs if p.name in keep]
    return dirs or [ROOT]

def zscore(s):
    m = s.rolling(Z_WIN, min_periods=Z_WIN // 2).mean()
    sd = s.rolling(Z_WIN, min_periods=Z_WIN // 2).std()
    return (s - m) / sd

def first_touch(h, l, c, s):
    ref = c[s]; up, dn = ref * (1 + MOVE), ref * (1 - MOVE)
    for u in range(s + 1, min(s + 1 + WAIT, len(c))):
        a, b = h[u] >= up, l[u] <= dn
        if a and b:
            return "불명"
        if a:
            return "위"
        if b:
            return "아래"
    return "안 터짐"

rows = []
for sd in symbol_dirs():
    bt.D = sd
    k = bt.load()
    m = pd.read_csv(sd / "metrics.csv")
    missing = [c for c in FEATURES.values() if c not in m.columns]
    if missing:
        print(f"[{sd.name}] metrics에 {missing} 없음, 건너뜀"); continue
    m["mt"] = pd.to_datetime(m["create_time"]).astype("datetime64[ns]")
    m = m[["mt"] + list(FEATURES.values())].sort_values("mt")
    k = pd.merge_asof(k, m, left_on="t", right_on="mt", direction="backward",
                      tolerance=pd.Timedelta("15min")).drop(columns="mt")

    tk = "sum_taker_long_short_vol_ratio"
    k[tk] = k[tk].rolling(4, min_periods=1).mean()            # 최근 1시간 평균
    k["rsi"] = bt.rsi(k["close"], bt.P["rsi_len"])
    for name, col in FEATURES.items():
        k[name] = zscore(k[col])
    k["큰손-개미 차이"] = k["큰손 롱숏비율"] - k["개미 롱숏비율"]
    k["펀딩비 크기"] = zscore(k["funding"])

    sig = ((k["funding"] >= bt.P["funding_min"]) & (k["rsi"] >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = sig.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    idx = np.flatnonzero((sig & ~prev).values)

    h, l, c = k.high.values, k.low.values, k.close.values
    for s in idx:
        r = dict(symbol=sd.name, time=k.t.iloc[s], 결과=first_touch(h, l, c, s),
                 ret24=c[s + WAIT] / c[s] - 1 if s + WAIT < len(c) else np.nan)
        for f in list(FEATURES) + ["큰손-개미 차이", "펀딩비 크기"]:
            r[f] = k[f].iloc[s]
        rows.append(r)

ev = pd.DataFrame(rows)
print(f"신호 사건 {len(ev)}건 | 결과 분포: " +
      ", ".join(f"{k} {v}" for k, v in ev["결과"].value_counts().items()))
base_up = (ev["결과"] == "위").mean(); base_dn = (ev["결과"] == "아래").mean()
print(f"전체 기준: 위로 터짐 {base_up:.0%}, 아래로 터짐 {base_dn:.0%}\n")
print("※ 각 조건을 1/3씩 나눔. '위-아래' 차이가 하위→상위로 갈수록 한 방향으로 꾸준히 변하면 의미 있는 신호.")
print("※ 1/3 그룹 하나에 약 {}건 → 비율 오차 대략 ±{:.0%}\n".format(len(ev) // 3, 1 / np.sqrt(max(len(ev) // 3, 1))))

pd.set_option("display.width", 200)
for f in list(FEATURES) + ["큰손-개미 차이", "펀딩비 크기"]:
    d = ev.dropna(subset=[f])
    if len(d) < 30:
        print(f"[{f}] 데이터 부족\n"); continue
    d = d.assign(구간=pd.qcut(d[f], 3, labels=["하위", "중간", "상위"]))
    t = d.groupby("구간", observed=True).agg(
        건수=("결과", "size"),
        z평균=(f, "mean"),
        위로=("결과", lambda x: (x == "위").mean()),
        아래로=("결과", lambda x: (x == "아래").mean()),
        안터짐=("결과", lambda x: (x == "안 터짐").mean()),
        평균24h=("ret24", "mean"),
        중앙24h=("ret24", "median"),
    )
    t["위-아래"] = t["위로"] - t["아래로"]
    for col in ["위로", "아래로", "안터짐", "위-아래"]:
        t[col] = t[col].map(lambda v: f"{v:+.0%}" if col == "위-아래" else f"{v:.0%}")
    for col in ["평균24h", "중앙24h"]:
        t[col] = t[col].map(lambda v: f"{v:+.2%}")
    t["z평균"] = t["z평균"].map(lambda v: f"{v:+.2f}")
    print(f"[{f}]")
    print(t.to_string())
    print()

ev.to_csv(ROOT / "direction_events.csv", index=False)