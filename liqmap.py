"""추정 청산 맵으로 과열 후 폭발 방향 예측 (2026-11-08 22:2x 등록, data_final 판정 전 확정)
가설: 과열 후 폭발은 청산 물량이 더 두껍게 쌓인 쪽으로 터진다 (가격이 청산 물량을 사냥)
청산 맵 추정 (신호 전 30일):
  - OI가 늘어난 15분봉마다 그 봉 종가에 새 포지션 진입으로 가정 (명목 = OI 증가량 × 가격)
  - 롱/숏 나누기: 전체 계정 롱숏비율 r → 롱 비중 r/(1+r)
  - 레버리지 분포: 10배 30% / 25배 40% / 50배 30%
  - 청산가: 롱 = 진입가 × (1 - 1/L + 0.5%), 숏 = 진입가 × (1 + 1/L - 0.5%)
  - 이후 가격이 청산가를 지나갔으면 이미 청산 → 제거
  - OI가 줄어든 봉에서는 기존 포지션 전체를 같은 비율로 축소
특징값 (신호 시점, 현재가 P):
  불균형 = (위 5% 안 숏 청산량 - 아래 5% 안 롱 청산량) / 합계   (+ 면 위쪽에 청산이 더 많음)
결과: 신호 후 24시간 안에 ±5% 중 먼저 닿은 쪽 (위 +1 / 아래 -1 / 안 터짐·불명 0)
판정: data_final 에서 순위 상관 > 0 + 월 단위 군집 t >= 2.3 (최종 데이터에 가설 2개 → 기준 상향)
      기존 30종목은 참고용
사용법: python liqmap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목 (판정)")]
WIN, H, MOVE, BAND, MMR, COOLDOWN = 30 * 96, 96, 0.05, 0.05, 0.005, 16
LEVS = {10: 0.3, 25: 0.4, 50: 0.3}

def load(root, s):
    bt.D = root / s
    k = bt.load()
    m = pd.read_csv(root / s / "metrics.csv", usecols=["create_time", "count_long_short_ratio"])
    m["mt"] = pd.to_datetime(m["create_time"]).astype("datetime64[ns]")
    k = pd.merge_asof(k, m[["mt", "count_long_short_ratio"]].sort_values("mt"), left_on="t", right_on="mt",
                      direction="backward", tolerance=pd.Timedelta("15min"))
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = sig.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return k, np.flatnonzero((sig & ~prev).values)

def first_touch(h, l, c, s):
    up, dn = c[s] * (1 + MOVE), c[s] * (1 - MOVE)
    for u in range(s + 1, min(s + 1 + H, len(c))):
        a, b = h[u] >= up, l[u] <= dn
        if a and b: return 0
        if a: return 1
        if b: return -1
    return 0

def imbalance(k, s):
    a = max(1, s - WIN)
    c, h, l = k["close"].values[a:s + 1], k["high"].values[a:s + 1], k["low"].values[a:s + 1]
    raw_oi = k["oi"].iloc[a:s + 1]
    if raw_oi.isna().mean() > 0.2 or (raw_oi <= 0).mean() > 0.2:
        return np.nan
    oi = raw_oi.where(raw_oi > 0).ffill().bfill().values          # 0·빈 값은 직전 값으로
    r = k["count_long_short_ratio"].ffill().values[a:s + 1]
    if np.isnan(r).all():
        return np.nan
    d = np.diff(oi, prepend=oi[0])
    ratio = np.where(d < 0, oi / (oi - d), 1.0)                     # OI 감소 봉의 축소 비율 (0~1)
    logcum = np.cumsum(np.log(np.clip(ratio, 1e-9, 1.0)))
    keep = np.exp(logcum[-1] - logcum)                              # 진입 이후 누적 축소 (0으로 나누기 없음)
    # 이후 구간 최저/최고 (진입 봉 다음부터 신호 봉까지)
    sufmin = np.minimum.accumulate(l[::-1])[::-1]
    sufmax = np.maximum.accumulate(h[::-1])[::-1]
    after_min = np.append(sufmin[1:], np.inf)
    after_max = np.append(sufmax[1:], -np.inf)
    P = c[-1]
    longshare = np.nan_to_num(r / (1 + r), nan=0.5)
    new = np.where(d > 0, d * c, 0.0) * keep
    L_dn = S_up = 0.0
    for L, w in LEVS.items():
        lp = c * (1 - 1 / L + MMR)
        sp = c * (1 + 1 / L - MMR)
        alive_l = after_min > lp
        alive_s = after_max < sp
        L_dn += (new * longshare * w)[alive_l & (lp >= P * (1 - BAND)) & (lp <= P)].sum()
        S_up += (new * (1 - longshare) * w)[alive_s & (sp <= P * (1 + BAND)) & (sp >= P)].sum()
    tot = L_dn + S_up
    return (S_up - L_dn) / tot if tot > 0 else np.nan

def rank_t(d):
    d = d.dropna(subset=["imb"])
    x = d.imb.rank().values; y = d.score.rank().values
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = float(np.mean(x * y)); e = y - r * x
    g = pd.Series(x * e).groupby(d.time.dt.to_period("M").values).sum().values
    return len(d), r, r / (np.sqrt((g**2).sum()) / (x**2).sum())

pd.set_option("display.width", 220)
rows, final = [], None
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        print(f"[{label}] 없음"); continue
    syms = (root / "symbols.txt").read_text().split()
    skipped = [s for s in syms if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"])]
    if folder == "data_final" and skipped:
        print(f"[{label}] 다운로드가 아직 안 끝났어요 ({len(syms) - len(skipped)}/{len(syms)}개 완료) — 최종 판정은 다 받은 뒤에"); continue
    ev = []
    for s in syms:
        if s in skipped:
            continue
        k, eps = load(root, s)
        h, l, c = k["high"].values, k["low"].values, k["close"].values
        for e in eps:
            if e + H >= len(c) or e < 96:
                continue
            ev.append(dict(symbol=s, time=k["t"].iloc[e], imb=imbalance(k, e), score=first_touch(h, l, c, e)))
    d = pd.DataFrame(ev)
    n, r, t = rank_t(d)
    q = d.dropna(subset=["imb"])
    q = q.assign(구간=pd.qcut(q.imb.rank(method="first"), 3, labels=["아래 청산 많음", "중간", "위 청산 많음"]))
    g = q.groupby("구간", observed=True).score.agg(
        건수="size", 위로=lambda x: (x == 1).mean(), 아래로=lambda x: (x == -1).mean())
    rows.append(dict(데이터=label, 사건=n, 순위상관=round(r, 3), 군집t=round(t, 2),
                     **{f"{i} 위-아래": f"{g.loc[i, '위로'] - g.loc[i, '아래로']:+.0%}" for i in g.index}))
    if folder == "data_final":
        final = (r, t)
    d.to_csv(root / "liqmap_events.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("※ 가설대로면 순위상관 +, '위 청산 많음' 구간의 위-아래가 가장 높아야 함")
if final is None:
    print("\n최종 판정: 최종 15종목 데이터가 아직 준비 안 됨")
else:
    r, t = final
    print("\n최종 판정 (최종 15종목, 기준 t >= 2.3):", "통과" if r > 0 and t >= 2.3 else ("방향만 맞음 (약함)" if r > 0 else "실패"))
