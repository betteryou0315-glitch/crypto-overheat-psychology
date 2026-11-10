"""최종 가설 검증 — 과열 후 하락 반등 (2026-11-08 22:0x 등록, data_final 을 보기 전에 확정)
근거 (모두 기존 30종목, 결과를 보고 찾은 패턴이라 확인 필요):
  - wait_follow.py: 과열 후 하락을 숏으로 따라가면 두 데이터 × 세 관찰시간 모두 큰 손실 (−1.1 ~ −3.2%)
  - trend.py: 하락 돌파 숏이 두 데이터 모두 손실
  - fakeout.py: 아래 가짜 이탈 후 롱이 두 데이터 모두 플러스
규칙 (변경 없음):
  - 과열 신호 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만)
  - 신호 후 4시간(16봉) 수익률이 −2% 이하이면 그 시점 종가에 롱
  - 신호 후 24시간 시점 종가에 청산, 손절·익절 없음, 비용 왕복 0.14%
판정: data_final 거래당 평균 > 0 + 월 단위 군집 t >= 2 → 통과 (기준 완화 없음)
참고 출력: 기존 30종목 같은 규칙 결과, 중앙값, 승률, 상위 10% 제외 평균 (소수 대박 의존 여부)
사용법: python dip_rebound_final.py   (python download_final.py 를 먼저 실행)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목 (판정)")]
W, H, MOVE, COST, COOLDOWN = 16, 96, -0.02, 0.0014, 16

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
rows, final = [], None
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        print(f"[{label}] {folder} 없음"); continue
    tr, skipped = [], []
    syms = (root / "symbols.txt").read_text().split()
    for s in syms:
        if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"]):
            skipped.append(s); continue
        bt.D = root / s
        k = bt.load()
        c = k["close"].values
        for e in episodes(k):
            if e + H >= len(c):
                continue
            if c[e + W] / c[e] - 1 <= MOVE:
                tr.append(dict(symbol=s, time=k["t"].iloc[e + W], ret=c[e + H] / c[e + W] - 1 - COST))
    if skipped:
        print(f"[{label}] 파일이 덜 받아진 종목 건너뜀: {', '.join(skipped)}")
    if folder == "data_final" and skipped:
        print(f"[{label}] 다운로드가 아직 안 끝났어요 — 최종 판정은 다 받은 뒤에 ({len(syms) - len(skipped)}/{len(syms)}개 완료)")
        continue
    d = pd.DataFrame(tr)
    if d.empty:
        rows.append(dict(데이터=label, 거래=0)); continue
    m, t = cluster_t(d.ret.values, d.time)
    q90 = d.ret.quantile(0.9)
    rows.append(dict(데이터=label, 거래=len(d), 종목=d.symbol.nunique(), 승률=f"{(d.ret > 0).mean():.0%}",
                     거래당=f"{m:+.2%}", 중앙값=f"{d.ret.median():+.2%}",
                     상위10퍼제외=f"{d.ret[d.ret < q90].mean():+.2%}", 최악=f"{d.ret.min():+.1%}",
                     최고=f"{d.ret.max():+.1%}", 군집t=round(t, 2)))
    if folder == "data_final":
        final = (m, t)
print(pd.DataFrame(rows).fillna("").to_string(index=False))
if final is None:
    print("\n최종 판정: data_final 없음 — python download_final.py 먼저 실행")
else:
    m, t = final
    print("\n최종 판정:", "통과" if m > 0 and t >= 2 else ("방향만 맞음 (약함)" if m > 0 else "실패"))
