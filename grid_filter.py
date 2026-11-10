"""알트코인 중립 그리드 + 과열 필터 백테스트 (설정은 미리 확정)
그리드 (매일 00:00 UTC 리셋):
  - 기준가 = 그날 00:00 가격, 간격 0.5%, 범위 ±5% (위아래 10칸)
  - 기준가 아래 k칸이면 롱 k단위, 위 k칸이면 숏 k단위 (1단위 = 자본의 1/10 → 범위 끝에서 1배)
  - 칸을 지날 때마다 그 칸 가격에 지정가 체결 (수수료 0.02%)
  - 범위 이탈 → 전부 청산, 그날 종료 / 하루 끝 → 남은 포지션 청산 (시장가 0.05%)
  - 가격 경로는 15분봉 종가만 사용 (봉 안의 왕복은 놓침 → 체결 횟수 과소, 보수적)
비교:
  A. 매일 그리드
  B. 시작 전 필터 : 직전 24시간 안에 그 코인에 과열 신호가 있었으면 그날 쉼
  C. 실시간 정지  : 돌리는 중 과열 신호가 뜨면 즉시 청산하고 그날 종료
대상: data(기존 15종목), data_holdout(검증 15종목)
사용법: python grid_filter.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
G, K = 0.005, 10
UNIT = 1 / K
MAKER, TAKER = 0.0002, 0.0005
DAY = 96

def run_day(p, sig, stop_on_signal):
    """하루치 종가 경로 p (길이 97: 시작가 + 96봉), 손익(자본 대비) 반환"""
    c0 = p[0]
    levels = c0 * (1 + G * np.arange(-K, K + 1))
    pos, cash, idx = 0, 0.0, 0                      # pos: 단위 수 (+롱/-숏), cash: 실현 손익(가격 대비 비율 누적)
    for j in range(1, len(p)):
        r = p[j] / c0 - 1
        if abs(r) > G * K or (stop_on_signal and sig[j]):
            px = p[j]
            cash += pos * UNIT * (px / c0 - 1) - abs(pos) * UNIT * TAKER
            return cash, bool(stop_on_signal and sig[j])
        new = int(np.trunc(r / G))                    # 현재 칸 (0 = 기준가 근처)
        while idx != new:                             # 한 칸씩 체결
            step = 1 if new > idx else -1
            idx += step
            lv = levels[idx + K]
            trade = -step                             # 위로 한 칸 → 1단위 매도, 아래로 → 1단위 매수
            cash += -trade * UNIT * (lv / c0 - 1) - UNIT * MAKER
            pos += trade
    cash += pos * UNIT * (p[-1] / c0 - 1) - abs(pos) * UNIT * TAKER
    return cash, False

def portfolio_stats(daily):
    if daily.empty:
        return {}
    eq = daily.cumsum()
    w = np.sort(daily.values)[:max(1, len(daily) // 100)].mean()
    return dict(일수=len(daily), 하루평균=f"{daily.mean():+.3%}", 승률=f"{(daily > 0).mean():.0%}",
                최악의날=f"{daily.min():+.2%}", 최악1퍼센트=f"{w:+.2%}", 최대낙폭=f"{(eq - eq.cummax()).min():+.1%}",
                샤프=round(daily.mean() / daily.std() * np.sqrt(365), 2))

pd.set_option("display.width", 220)
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split() if (root / s / "klines_15m.csv").exists()]
    rows = []
    for s in syms:
        bt.D = root / s
        k = bt.load()
        rsi = bt.rsi(k["close"], bt.P["rsi_len"])
        sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
               & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False).values
        c = k["close"].values
        starts = np.flatnonzero((k["t"].dt.hour == 0) & (k["t"].dt.minute == 0))
        for st in starts:
            if st + DAY >= len(c) or st < DAY:
                continue
            p, sg = c[st:st + DAY + 1], sig[st:st + DAY + 1]
            pre = bool(sig[st - DAY:st].any())
            a, _ = run_day(p, sg, False)
            cc, stopped = run_day(p, sg, True)
            rows.append(dict(symbol=s, day=k["t"].iloc[st].normalize(), A=a, C=cc, 직전과열=pre, 중간정지=stopped))
    d = pd.DataFrame(rows)
    d["B"] = np.where(d["직전과열"], 0.0, d["A"])

    print(f"\n==================== {label} — 종목 {d.symbol.nunique()}개, 종목·일 {len(d):,}개 ====================")
    port = d.groupby("day")[["A", "B", "C"]].mean()            # 종목 균등 배분 포트폴리오의 하루 수익
    print("[포트폴리오 (모든 종목에 자본 균등 배분)]")
    print(pd.DataFrame([dict(전략="A. 매일 그리드", **portfolio_stats(port.A)),
                        dict(전략="B. 시작 전 필터", **portfolio_stats(port.B)),
                        dict(전략="C. 실시간 정지", **portfolio_stats(port.C))]).to_string(index=False))

    skip, keep = d[d.직전과열], d[~d.직전과열]
    print("\n[B: 쉰 날 vs 돌린 날 — 종목·일 단위, 그리드 A의 손익]")
    print(f"쉰 날 {len(skip):,}개 평균 {skip.A.mean():+.3%} (승률 {(skip.A > 0).mean():.0%}, 최악 {skip.A.min():+.2%})"
          f"  |  돌린 날 {len(keep):,}개 평균 {keep.A.mean():+.3%} (승률 {(keep.A > 0).mean():.0%})")
    worst = d.nsmallest(max(10, len(d) // 100), "A")
    print(f"최악 1% 종목·일 {len(worst)}개 중 B가 미리 피한 비율: {worst.직전과열.mean():.0%}  (전체 중 쉰 날 비율 {d.직전과열.mean():.0%})")

    st = d[d.중간정지]
    print(f"\n[C: 실시간 정지가 일어난 날 {len(st):,}개] 그대로 돌렸으면 {st.A.mean():+.3%} → 정지해서 {st.C.mean():+.3%}")
    d.to_csv(root / "grid_days.csv", index=False)
