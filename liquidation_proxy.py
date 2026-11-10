"""청산 대리 지표(OI 급감) — 청산 소진 후 반전 가설 (미리 확정한 마지막 방향 테스트)
가설: 과열 이후 한쪽이 강제청산으로 털리면 (OI 급감 + 가격 급변), 그다음엔 반대로 튕긴다
규칙 (미리 확정, 변경 없음):
  - 사건: 각 폴더 direction_events.csv 의 과열 신호
  - 청산 추정 봉: 신호 후 24시간 안에 처음 나온 15분봉 중
        OI 15분 변화 <= -2%  AND  |가격 변화(시가→종가)| >= 1.5%
  - 청산 방향: 그 봉에서 가격 하락 = 롱 청산 / 상승 = 숏 청산
  - 결과: 그 봉 종가 이후 24시간 안에 ±3% 중 먼저 닿는 쪽
        청산 방향의 반대 = 반전(+1), 같은 방향 = 지속(-1), 안 터짐/불명 = 0
  - 판정: 두 폴더 모두 반전 우세 + 월 단위 군집 t >= 2 → 통과
          두 폴더 모두 t <= -2 → '지속' 우세로 기록
사용법: python liquidation_proxy.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
OI_DROP, PX_MOVE, WAIT, MOVE = -0.02, 0.015, 96, 0.03

def first_touch(h, l, c, s, n):
    up, dn = c[s] * (1 + MOVE), c[s] * (1 - MOVE)
    for u in range(s + 1, min(s + 1 + WAIT, n)):
        a, b = h[u] >= up, l[u] <= dn
        if a and b: return 0
        if a: return 1
        if b: return -1
    return 0

def cluster_t(d):
    """반전 점수 평균이 0보다 큰가 — 월 단위 군집 보정 t"""
    y = d["점수"].values
    m = y.mean()
    g = pd.Series(y - m).groupby(d["time"].dt.to_period("M").values).sum().values
    se = np.sqrt((g**2).sum()) / len(y)
    return m, m / se if se > 0 else np.nan, len(g)

pd.set_option("display.width", 200)
summary = []
for folder, label in FOLDERS:
    ev = pd.read_csv(Path(folder) / "direction_events.csv", parse_dates=["time"])
    rows = []
    for sym, g in ev.groupby("symbol"):
        bt.D = Path(folder) / sym
        k = bt.load()
        n = len(k)
        o, h, l, c = k.open.values, k.high.values, k.low.values, k.close.values
        oi_chg = (k["oi"] / k["oi"].shift(1) - 1).values
        pos = pd.Series(np.arange(n), index=k["t"])
        for t in g["time"]:
            s = pos.get(t)
            if s is None:
                continue
            for u in range(int(s) + 1, min(int(s) + 1 + WAIT, n)):
                px = c[u] / o[u] - 1
                if oi_chg[u] <= OI_DROP and abs(px) >= PX_MOVE:
                    liq = -1 if px < 0 else 1            # -1 롱 청산(가격↓), +1 숏 청산(가격↑)
                    nxt = first_touch(h, l, c, u, n)     # +1 위 / -1 아래 / 0
                    score = 0 if nxt == 0 else (1 if nxt == -liq else -1)
                    rows.append(dict(symbol=sym, time=k["t"].iloc[u], 신호시각=t,
                                     청산="롱 청산" if liq == -1 else "숏 청산",
                                     OI변화=oi_chg[u], 가격변화=px, 점수=score))
                    break
    d = pd.DataFrame(rows, columns=["symbol", "time", "신호시각", "청산", "OI변화", "가격변화", "점수"])
    d.to_csv(Path(folder) / "liquidation_events.csv", index=False)
    print(f"[{label}] 과열 사건 {len(ev)}건 중 청산 추정 봉이 나온 사건: {len(d)}건 "
          f"(롱 청산 {sum(d.청산=='롱 청산')}, 숏 청산 {sum(d.청산=='숏 청산')})")
    if d.empty:
        continue
    for name, sub in [("전체", d), ("롱 청산 후", d[d.청산 == "롱 청산"]), ("숏 청산 후", d[d.청산 == "숏 청산"])]:
        if len(sub) < 10:
            summary.append(dict(데이터=label, 구분=name, 건수=len(sub))); continue
        m, t, nm = cluster_t(sub)
        summary.append(dict(데이터=label, 구분=name, 건수=len(sub), 월수=nm,
                            반전=f"{(sub.점수 == 1).mean():.0%}", 지속=f"{(sub.점수 == -1).mean():.0%}",
                            안터짐=f"{(sub.점수 == 0).mean():.0%}", 평균점수=round(m, 3), 군집t=round(t, 2)))

print("\n=== 청산 추정 이후: 반전 vs 지속 (군집t 양수 = 반전 우세) ===")
s = pd.DataFrame(summary)
print(s.to_string(index=False))

ts = s[s.구분 == "전체"]["군집t"].tolist() if "군집t" in s else []
if len(ts) == 2:
    verdict = ("통과 (반전 우세)" if all(t >= 2 for t in ts) else
               "지속 우세로 기록" if all(t <= -2 for t in ts) else
               "방향만 일치 (약함)" if (ts[0] > 0) == (ts[1] > 0) else "실패")
    print(f"\n최종 판정: {verdict}")
