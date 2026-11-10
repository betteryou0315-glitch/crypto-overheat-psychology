"""펀딩비 캐리 개선 — 보유 완충으로 교체 비용 줄이기 (개선 하나만 미리 정해서 테스트)
원래 규칙  : 매주 지난 7일 펀딩비 합계 상위 3개로 교체
완충 규칙  : 상위 3개에 들면 진입, 6위 밖으로 밀려나거나 합계가 0 이하가 될 때까지 보유 (항상 3개 유지)
비용 (편도, 코인 한 개 교체 = 현물+선물 한쪽씩):
  - 시장가 : 현물 0.1% + 선물 0.05% + 슬리피지 0.02% × 2 = 0.19%   ← 판정 기준
  - 지정가 : 현물 0.075%(BNB 할인) + 선물 0.02% = 0.095%            ← 참고 (체결 보장 없음, 낙관적)
나머지는 carry.py 와 같음 (현물 상장 코인만, 손익 = 펀딩비 - 베이시스 변화 - 비용)
판정: 완충 + 시장가 의 주간 순수익 > 0 + t >= 2 를 두 데이터 모두에서 → 통과
사용법: python carry2.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
TOP, EXIT_RANK = 3, 6
COSTS = {"시장가": 0.0019, "지정가": 0.00095}
WEEK = pd.Timedelta("7D")

def spot_list():
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=30).json()
        return {s["symbol"] for s in r["symbols"] if s.get("status") == "TRADING" and s["quoteAsset"] == "USDT"}
    except Exception as e:
        print("현물 목록을 못 받았어요 — 모든 코인 대상:", e); return None

SPOT = spot_list()

def load(root, s):
    k = pd.read_csv(root / s / "klines_15m.csv", usecols=["close_time", "close"])
    idx = pd.to_datetime(k["close_time"] + 1, unit="ms").astype("datetime64[ns]")
    p = pd.read_csv(root / s / "premium_15m.csv", usecols=["close_time", "close"])
    f = pd.read_csv(root / s / "funding.csv")
    return dict(close=pd.Series(k["close"].values, index=idx),
                prem=pd.Series(p["close"].values, index=pd.to_datetime(p["close_time"] + 1, unit="ms").astype("datetime64[ns]")),
                fr=pd.Series(f["last_funding_rate"].values,
                             index=pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[ns]")).sort_index())

def run(D, buffer):
    end = max(d["close"].index.max() for d in D.values())
    held, rows = [], []
    for t0 in pd.date_range("2023-11-13", end - WEEK, freq="W-MON"):
        t1 = t0 + WEEK
        cand = {}
        for s, d in D.items():
            if any(x not in d[c].index for x in (t0, t1) for c in ("close", "prem")):
                continue
            past = d["fr"][(d["fr"].index > t0 - WEEK) & (d["fr"].index <= t0)]
            if past.size >= 7:
                cand[s] = past.sum()
        ranked = [s for s, v in sorted(cand.items(), key=lambda x: -x[1]) if v > 0]
        if buffer:
            keep = [s for s in held if s in ranked[:EXIT_RANK]]
            new = keep + [s for s in ranked if s not in keep][:TOP - len(keep)]
        else:
            new = ranked[:TOP]
        changes = len(set(new) ^ set(held))
        held = new
        fund = sum(D[s]["fr"][(D[s]["fr"].index > t0) & (D[s]["fr"].index <= t1)].sum() for s in new) / TOP
        basis = sum(-(D[s]["prem"][t1] - D[s]["prem"][t0]) for s in new) / TOP
        rows.append(dict(week=t0, changes=changes, 펀딩비=fund, 베이시스=basis))
    return pd.DataFrame(rows).set_index("week")

pd.set_option("display.width", 220)
verdict = []
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split()
            if (root / s / "premium_15m.csv").exists()
            and (SPOT is None or re.sub(r"^1000+", "", s) in SPOT)]
    D = {s: load(root, s) for s in syms}
    print(f"\n==================== {label} — 대상 {len(syms)}개 ====================")
    rows = []
    for rule, buf in [("원래 (매주 상위 3)", False), ("완충 (6위 밖이면 교체)", True)]:
        w = run(D, buf)
        for cname, c in COSTS.items():
            net = w.펀딩비 + w.베이시스 - w.changes * c / TOP
            t = net.mean() / (net.std() / np.sqrt(len(net)))
            eq = net.cumsum()
            rows.append(dict(규칙=rule, 비용=cname, 주당교체=f"{w.changes.mean():.2f}개",
                             펀딩비_연=f"{w.펀딩비.mean() * 52:+.1%}", 비용_연=f"{-(w.changes * c / TOP).mean() * 52:+.1%}",
                             순수익_연=f"{net.mean() * 52:+.1%}", 주간승률=f"{(net > 0).mean():.0%}", t값=round(t, 2),
                             최대낙폭=f"{(eq - eq.cummax()).min():+.1%}",
                             자본대비_연_2배=f"{net.mean() * 52 / 1.5:+.1%}"))
            if buf and cname == "시장가":
                verdict.append((net.mean(), t))
    print(pd.DataFrame(rows).to_string(index=False))

ok = len(verdict) == 2 and all(m > 0 and t >= 2 for m, t in verdict)
print("\n최종 판정 (완충 + 시장가):", "통과" if ok else
      "방향만 맞음 (약함)" if len(verdict) == 2 and all(m > 0 for m, _ in verdict) else "실패")
print("※ 자본대비_연_2배 = 선물 레버리지 2배일 때 (현물 + 증거금 = 명목의 1.5배)")
