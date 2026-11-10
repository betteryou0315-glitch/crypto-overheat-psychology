"""과거 기간(2022-01 ~ 2023-10, 28종목) 판정 — 2026-11-09 18:5x 등록, data_past 를 보기 전에 확정
이 기간은 지금까지 한 번도 안 본 데이터. 2022년 LUNA·FTX 대하락장 포함 → 지금까지(2023-11 이후, 대체로 상승장)와 다른 시장
가설 1 (주): CVD — 첫 ±2% 폭발 직전 15분 시장가 우위가 움직인 방향과 같을수록 지속 (cvd.py 규칙 그대로)
             판정: 순위상관 > 0 + 월 단위 군집 t >= 2.3
가설 2     : 청산 흡수 — 흡수 후 4시간 순수익(왕복 0.14% 차감) > 0 (absorption.py 규칙 그대로)
             판정: 평균 > 0 + 월 단위 군집 t >= 2.3
             (가설 2개 → 기준 2.3)
참고 (판정 없음): 롱 과열 → 폭발이 하락장에서도 성립하나 (24시간 ±5% 폭발률, 신호 없는 봉 대비 배수, 방향)
사용법: python past_verdict.py   (python download_past.py → python download_agg1m.py data_past 후)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt
import cvd
import absorption as ab

FOLDER, COST, H, MOVE = "data_past", 0.0014, 96, 0.05
root = Path(FOLDER)
pd.set_option("display.width", 230)

def ready():
    """종목 수가 아니라 '받아야 할 종목·날짜'를 실제로 다 받았는지로 확인
    (과열 사건이 0건인 코인은 받을 체결 데이터가 없어서 agg1m 폴더가 없는 게 정상)"""
    if not (root / "symbols.txt").exists():
        return False, "data_past 없음 — python download_past.py 먼저"
    syms = (root / "symbols.txt").read_text().split()
    base = [s for s in syms if all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"])]
    if len(base) < len(syms):
        return False, f"15분봉·펀딩비·OI {len(base)}/{len(syms)}개"
    import download_agg1m as dam
    dam.FOLDERS = [FOLDER]
    jobs = dam.needed_days()
    have = [j for j in jobs if (j[0] / j[1] / "agg1m" / f"{j[2]}.csv.gz").exists()]
    with_events = len({j[1] for j in jobs})
    msg = (f"과열 사건 있는 종목 {with_events}/{len(syms)}개 | 필요한 종목·날짜 {len(jobs)}개 중 {len(have)}개 받음")
    # 바이낸스에 파일이 없는 날이 소수 있을 수 있어 95% 이상이면 진행
    if len(jobs) and len(have) < len(jobs) * 0.95:
        return False, msg + " — python download_agg1m.py data_past 를 다시 실행해서 마저 받기"
    return True, msg

def mcluster_t(x, times):
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.to_datetime(pd.Series(times)).dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x))

ok, msg = ready()
print(f"[과거 기간 28종목] {msg}")
if not ok:
    raise SystemExit("판정 보류")

# ── 참고: 롱 과열 → 폭발 (하락장에서도?) ─────────────────────────────
ev, base_big, base_n = [], 0, 0
for s in (root / "symbols.txt").read_text().split():
    bt.D = root / s
    k = bt.load()
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    h, l, c = k["high"].values, k["low"].values, k["close"].values
    hmax = k["high"].rolling(H).max().shift(-H).values
    lmin = k["low"].rolling(H).min().shift(-H).values
    big = (hmax / c - 1 >= MOVE) | (lmin / c - 1 <= -MOVE)
    valid = ~np.isnan(hmax)
    base_big += big[valid & ~sig.values].sum(); base_n += (valid & ~sig.values).sum()
    for e in cvd.episodes(k):
        if e + H >= len(c):
            continue
        up, dn, first = c[e] * (1 + MOVE), c[e] * (1 - MOVE), "안 터짐"
        for u in range(e + 1, e + 1 + H):
            a, b = h[u] >= up, l[u] <= dn
            if a and b: first = "불명"; break
            if a: first = "위"; break
            if b: first = "아래"; break
        ev.append(dict(big=big[e], first=first))
ev = pd.DataFrame(ev)
base = base_big / base_n
print(f"\n=== 참고: 롱 과열 → 폭발 (하락장 포함 기간) ===")
print(f"과열 사건 {len(ev)}건 | 24시간 ±5% 폭발률 {ev.big.mean():.0%} vs 신호 없는 봉 {base:.0%} → {ev.big.mean() / base:.2f}배 | "
      f"위 {(ev['first'] == '위').mean():.0%} / 아래 {(ev['first'] == '아래').mean():.0%} / 안 터짐 {(ev['first'] == '안 터짐').mean():.0%}")
print("(지금까지 2023-11 이후: 폭발률 77~81%, 기준 대비 약 1.5배, 위/아래 거의 반반)")

# ── 가설 1: CVD ─────────────────────────────────────────────────────
df, miss = cvd.analyze(FOLDER)
row, r1, t1 = cvd.summarize(df, "과거 28종목", miss)
print("\n=== 가설 1: CVD ===")
print(pd.DataFrame([row]).to_string(index=False))
v1 = "통과" if r1 > 0 and t1 >= 2.3 else ("방향만 맞음 (약함)" if r1 > 0 else "실패")
print("판정 (기준 t >= 2.3):", v1)

# ── 가설 2: 청산 흡수 ───────────────────────────────────────────────
d, (nev, nflush) = ab.analyze(FOLDER)
print(f"\n=== 가설 2: 청산 흡수 === (과열 사건 {nev} 중 청산 투매 {nflush})")
if len(d):
    print(ab.table(d).to_string(index=False))
    g = d[d.kind == "흡수"].dropna(subset=["r4"]).copy()
    g["net4"] = g.r4 - COST
    if len(g) >= 10:
        m2, t2 = mcluster_t(g.net4.values, g.time)
        print(f"흡수 {len(g)}건 | 4시간 순수익 평균 {m2:+.2%} | 중앙값 {g.net4.median():+.2%} | 승률 {(g.net4 > 0).mean():.0%} | 군집t {t2:.2f}")
        print("판정 (기준 t >= 2.3):", "통과" if m2 > 0 and t2 >= 2.3 else ("방향만 맞음 (약함)" if m2 > 0 else "실패"))
    else:
        print(f"흡수 {len(g)}건 — 너무 적어 판정 불가")
else:
    print("청산 투매 사건 없음")
