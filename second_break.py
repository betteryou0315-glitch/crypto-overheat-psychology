"""두 번째 돌파 = 진짜 폭발 (2026-11-09 등록, data_new 를 보기 전에 확정)
가설: 과열 후 첫 돌파는 대부분 가짜다. 첫 가짜 돌파를 흘려보낸 뒤 나오는 두 번째 돌파가 진짜 폭발의 시작이다
규칙:
  1) 과열 신호 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만), 기준선 H/L = 신호 봉까지 24시간 최고/최저가
  2) 첫 번째 가짜 돌파: 신호 후 처음으로 고가>H 또는 저가<L, 그 봉부터 1시간(4봉) 안에 종가가 [L, H] 안으로 복귀
     → 1시간 안에 복귀 안 하면 (첫 돌파가 진짜) 이 패턴 아님, 제외
     → 한 봉에서 H, L 둘 다 뚫으면 제외
  3) 두 번째 돌파: 가짜 확인 이후, 신호 후 24시간 안에 종가 > H 또는 종가 < L 로 마감하는 첫 봉 (방향은 첫 번째와 같아도 반대여도 됨)
  4) 그 봉 종가에 그 방향 진입, 24시간(96봉) 뒤 종가 청산, 손절·익절 없음, 비용 왕복 0.14%
판정: data_new (46~60위 새 15종목) 거래당 평균 > 0 + 월 단위 군집 t >= 2 → 통과
      data, data_holdout, data_final 은 참고용
사용법: python second_break.py   (판정은 python download_new.py 후)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목 (참고)"),
           ("data_new", "새 15종목 (판정)")]
LOOK, CONFIRM, HOLD, COST, COOLDOWN = 96, 4, 96, 0.0014, 16

def episodes(k):
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    s = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
         & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return np.flatnonzero((s & ~prev).values)

def scan(h, l, c, s):
    n = len(c)
    if s < LOOK:
        return dict(kind="판단 불가")
    H, L = h[s - LOOK + 1:s + 1].max(), l[s - LOOK + 1:s + 1].min()
    end = min(s + LOOK, n - 1)
    for u in range(s + 1, end + 1):
        up, dn = h[u] > H, l[u] < L
        if not (up or dn):
            continue
        if up and dn:
            return dict(kind="판단 불가")
        first = 1 if up else -1
        back = None
        for v in range(u, min(u + CONFIRM + 1, end + 1)):
            if L <= c[v] <= H:
                back = v; break
        if back is None:
            return dict(kind="첫 돌파가 진짜")
        for w in range(back + 1, end + 1):
            if c[w] > H or c[w] < L:
                d = 1 if c[w] > H else -1
                if w + HOLD >= n:
                    return dict(kind="판단 불가")
                return dict(kind="두 번째 돌파", entry_i=w, side=d, first=first,
                            ret=d * (c[w + HOLD] / c[w] - 1) - COST)
        return dict(kind="두 번째 돌파 없음")
    return dict(kind="돌파 없음")

def cluster_t(x, times):
    if len(x) < 3:
        return np.nan, np.nan
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.Series(times).dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x))

def stats(d):
    if len(d) < 5:
        return dict(거래=len(d))
    m, t = cluster_t(d.ret.values, d.time)
    q90 = d.ret.quantile(0.9)
    return dict(거래=len(d), 승률=f"{(d.ret > 0).mean():.0%}", 거래당=f"{m:+.2%}", 중앙값=f"{d.ret.median():+.2%}",
                상위10퍼제외=f"{d.ret[d.ret < q90].mean():+.2%}", 군집t=round(t, 2))

pd.set_option("display.width", 220)
final = None
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        print(f"\n[{label}] {folder} 없음"); continue
    syms = (root / "symbols.txt").read_text().split()
    skipped = [s for s in syms if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"])]
    if folder == "data_new" and skipped:
        print(f"\n[{label}] 다운로드가 아직 안 끝났어요 ({len(syms) - len(skipped)}/{len(syms)}개)"); continue
    rows, cnt = [], {}
    for s in syms:
        if s in skipped:
            continue
        bt.D = root / s
        k = bt.load()
        h, l, c = k["high"].values, k["low"].values, k["close"].values
        busy = -1
        for e in episodes(k):
            if e <= busy:
                continue
            r = scan(h, l, c, e)
            cnt[r["kind"]] = cnt.get(r["kind"], 0) + 1
            if r["kind"] == "두 번째 돌파":
                busy = r["entry_i"] + HOLD
                rows.append(dict(symbol=s, time=k["t"].iloc[r["entry_i"]], side=r["side"], first=r["first"], ret=r["ret"],
                                 wait_h=(r["entry_i"] - e) / 4))
    d = pd.DataFrame(rows)
    print(f"\n==================== {label} ====================")
    print("사건 분류:", ", ".join(f"{k} {v}" for k, v in cnt.items()))
    if d.empty:
        continue
    print(f"두 번째 돌파까지 걸린 시간 (중앙): 신호 후 {d.wait_h.median():.1f}시간")
    print(pd.DataFrame([dict(구분="전체", **stats(d)),
                        dict(구분="같은 방향 (첫 가짜와 같은 쪽으로 다시)", **stats(d[d.side == d.first])),
                        dict(구분="반대 방향 (가짜 반대쪽으로)", **stats(d[d.side != d.first])),
                        dict(구분="롱", **stats(d[d.side == 1])),
                        dict(구분="숏", **stats(d[d.side == -1]))]).fillna("").to_string(index=False))
    d.to_csv(root / "second_break_trades.csv", index=False)
    if folder == "data_new":
        final = cluster_t(d.ret.values, d.time)

if final is None:
    print("\n최종 판정: data_new 없음 — python download_new.py 먼저 실행")
else:
    m, t = final
    print("\n최종 판정 (새 15종목):", "통과" if m > 0 and t >= 2 else ("방향만 맞음 (약함)" if m > 0 else "실패"))
