"""과열 후 지켜보고 따라가기 — 규칙 미리 확정
가설: 과열 직후의 혼란(휩쏘)이 지나간 뒤에는, 그동안 이긴 방향으로 계속 간다
규칙:
  - 과열 신호 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만)
  - 신호 후 W 동안 관찰. 그 사이 수익률 r1 이 ±2% 이상이면 그 방향으로 진입 (W 시점 종가)
  - 신호 후 24시간 시점 종가에 청산 (손절·익절 없음), 비용 왕복 0.14%
  - W = 4시간 (판정용) / 1시간, 8시간 (참고용 — 판정에 쓰지 않음)
판정: W=4시간, 기존 15종목·검증 15종목 모두 거래당 평균 > 0 + 30종목 합산 월 단위 군집 t >= 2 → 통과
참고: '관찰 구간 방향'과 '그 이후 방향'의 일치율 (진입 조건 없이 모든 사건)
사용법: python wait_follow.py   (data_final 이 있으면 같이 출력)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목")]
WAITS = {"1시간": 4, "4시간 (판정)": 16, "8시간": 32}
H, MOVE, COST, COOLDOWN = 96, 0.02, 0.0014, 16

def episodes(k):
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    s = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
         & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return np.flatnonzero((s & ~prev).values)

def cluster_t(x, times):
    if len(x) < 3:
        return np.nan, np.nan
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.Series(times).dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x))

pd.set_option("display.width", 220)
alltr = {w: [] for w in WAITS}
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        continue
    syms = [s for s in (root / "symbols.txt").read_text().split() if (root / s / "klines_15m.csv").exists()]
    ev = []
    for s in syms:
        bt.D = root / s
        k = bt.load()
        c = k["close"].values
        for e in episodes(k):
            if e + H >= len(c):
                continue
            row = dict(symbol=s, time=k["t"].iloc[e], data=label)
            for wn, W in WAITS.items():
                row[f"r1_{wn}"] = c[e + W] / c[e] - 1
                row[f"r2_{wn}"] = c[e + H] / c[e + W] - 1
            ev.append(row)
    ev = pd.DataFrame(ev)
    print(f"\n==================== {label} — 과열 사건 {len(ev)}건 ====================")
    rows = []
    for wn in WAITS:
        r1, r2 = ev[f"r1_{wn}"], ev[f"r2_{wn}"]
        agree = (np.sign(r1) == np.sign(r2)).mean()
        sel = ev[r1.abs() >= MOVE].copy()
        sel["side"] = np.sign(sel[f"r1_{wn}"])
        sel["ret"] = sel.side * sel[f"r2_{wn}"] - COST
        alltr[wn].append(sel.assign(data=label))
        m, t = cluster_t(sel.ret.values, sel.time)
        lg, sh = sel[sel.side > 0], sel[sel.side < 0]
        rows.append(dict(관찰시간=wn, 방향일치율_전체사건=f"{agree:.0%}", 진입=len(sel),
                         승률=f"{(sel.ret > 0).mean():.0%}" if len(sel) else "", 거래당=f"{m:+.3%}", 군집t=round(t, 2),
                         롱=f"{len(lg)}건 {lg.ret.mean():+.2%}" if len(lg) else "-",
                         숏=f"{len(sh)}건 {sh.ret.mean():+.2%}" if len(sh) else "-"))
    print(pd.DataFrame(rows).to_string(index=False))

print("\n==================== 판정 (4시간, 기존 + 검증 30종목) ====================")
d = pd.concat([x for x in alltr["4시간 (판정)"] if x.data.iloc[0] in ("기존 15종목", "검증 15종목")] if alltr["4시간 (판정)"] else [], ignore_index=True)
if len(d):
    means = d.groupby("data").ret.mean()
    m, t = cluster_t(d.ret.values, d.time)
    print(f"기존 {means.get('기존 15종목', np.nan):+.3%} | 검증 {means.get('검증 15종목', np.nan):+.3%} | 합산 {m:+.3%} (군집t {t:.2f})")
    ok = all(means.get(x, -1) > 0 for x in ("기존 15종목", "검증 15종목")) and t >= 2
    print("판정:", "통과" if ok else ("방향만 맞음 (약함)" if all(means.get(x, -1) > 0 for x in ("기존 15종목", "검증 15종목")) else "실패"))
