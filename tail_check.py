"""과열 신호의 '평균 상승'이 소수 폭등 때문인지 확인
1) 꼬리 자르기: 신호/기준 모두 똑같이 상위 5%(또는 위아래 5%) 제거 후 평균 비교
2) 꼬리 빈도: 큰 움직임(+5%, +10%, -5%, -10%)이 평소보다 자주 오는가
3) 기여도: 전체 수익 합계 중 상위 5% 사건이 차지하는 비율
사용법: python tail_check.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
HORIZONS = {"4h": 16, "24h": 96}
COOLDOWN = 16
THRESH = [0.05, 0.10, -0.05, -0.10]

def symbol_dirs():
    dirs = sorted(p for p in ROOT.iterdir() if (p / "klines_15m.csv").exists())
    if (ROOT / "symbols.txt").exists():
        keep = set((ROOT / "symbols.txt").read_text().split())
        dirs = [p for p in dirs if p.name in keep]
    return dirs or [ROOT]

def first_of_episode(s):
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return s & ~prev

data = {(h, g): [] for h in HORIZONS for g in ["기준", "셋 다"]}
for sd in symbol_dirs():
    bt.D = sd
    k = bt.load()
    k["rsi"] = bt.rsi(k["close"], bt.P["rsi_len"])
    sig = ((k["funding"] >= bt.P["funding_min"]) & (k["rsi"] >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    sig = first_of_episode(sig)
    for h, n in HORIZONS.items():
        f = (k["close"].shift(-n) / k["close"] - 1)
        data[(h, "기준")].append(f.dropna().values)
        data[(h, "셋 다")].append(f[sig].dropna().values)

def trimmed(r, top=0.05, bottom=0.0):
    lo, hi = np.quantile(r, bottom), np.quantile(r, 1 - top)
    return r[(r >= lo) & (r <= hi)].mean()

for h in HORIZONS:
    base = np.concatenate(data[(h, "기준")])
    s = np.concatenate(data[(h, "셋 다")])
    print(f"=============== {h} 뒤 (신호 {len(s)}건) ===============")

    print("\n[1] 꼬리 자르기 — 둘 다 똑같이 잘라서 비교")
    rows = []
    for name, fn in [("원래 평균", lambda r: r.mean()),
                     ("상위 5% 제거", lambda r: trimmed(r, 0.05)),
                     ("위아래 5% 제거", lambda r: trimmed(r, 0.05, 0.05)),
                     ("중앙값", np.median)]:
        b, x = fn(base), fn(s)
        rows.append(dict(방법=name, 기준=f"{b:+.3%}", 신호=f"{x:+.3%}", 차이=f"{x-b:+.3%}"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n[2] 큰 움직임 빈도 — 신호 뒤에 평소보다 자주 오는가")
    rows = []
    for t in THRESH:
        pb = (base >= t).mean() if t > 0 else (base <= t).mean()
        ps = (s >= t).mean() if t > 0 else (s <= t).mean()
        rows.append(dict(움직임=f"{t:+.0%} {'이상' if t > 0 else '이하'}",
                         기준=f"{pb:.2%}", 신호=f"{ps:.2%}",
                         배수=f"{ps/pb:.2f}배" if pb > 0 else "-"))
    print(pd.DataFrame(rows).to_string(index=False))

    q = np.quantile(s, 0.95)
    if s.sum() > 0:
        share = s[s >= q].sum() / s.sum()
        print(f"\n[3] 신호 쪽 상위 5% 사건(+{q:.1%} 이상)이 수익 합계에서 차지하는 비율: {share:.0%}"
              "  (100% 넘으면 나머지 95%는 합쳐서 손실)\n")
    else:
        print("\n[3] 신호 쪽 수익 합계가 0 이하라 기여도 계산 생략\n")