"""펀딩비 캐리 + 과열 적응형 레버리지 (carry2 의 완충 규칙 + 시장가 비용 그대로)
비교:
  - 전부 2배 / 전부 1배
  - 적응형: 매주 월요일 기준 직전 24시간 안에 그 코인에 과열 신호가 있었으면 1배, 아니면 2배
자본 = 각 코인 명목 × (1 + 1/L)  (현물 + 선물 증거금)
청산: 보유 1주 중 최고가 상승률 >= 1/L - 0.5% → 강제청산
      (현물 이익이 선물 손실을 메우므로 순손실 = 청산 수수료 1% + 재진입 비용 0.19%, 명목 대비)
판정 (미리 정함): 적응형 샤프 > 전부 2배 샤프 를 두 데이터 모두에서 → 통과
사용법: python carry3.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
TOP, EXIT_RANK, ONE_WAY = 3, 6, 0.0019
MMR, LIQ_FEE = 0.005, 0.01
WEEK, DAY = pd.Timedelta("7D"), 96
SCHEMES = {"전부 2배": lambda hot: 2, "전부 1배": lambda hot: 1, "적응형 (과열 1배 / 평소 2배)": lambda hot: 1 if hot else 2}

try:
    SPOT = {s["symbol"] for s in requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=30).json()["symbols"]
            if s.get("status") == "TRADING" and s["quoteAsset"] == "USDT"}
except Exception as e:
    print("현물 목록을 못 받았어요 — 모든 코인 대상:", e); SPOT = None

def load(root, s):
    bt.D = root / s
    k = bt.load()
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
           & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    k = k.set_index("t")
    p = pd.read_csv(root / s / "premium_15m.csv", usecols=["close_time", "close"])
    f = pd.read_csv(root / s / "funding.csv")
    return dict(close=k["close"], high=k["high"],
                hot=pd.Series(sig.values, index=k.index).astype(int).rolling(DAY, min_periods=1).sum() > 0,
                prem=pd.Series(p["close"].values, index=pd.to_datetime(p["close_time"] + 1, unit="ms").astype("datetime64[ns]")),
                fr=pd.Series(f["last_funding_rate"].values,
                             index=pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[ns]")).sort_index())

pd.set_option("display.width", 220)
sharpes = {}
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split()
            if (root / s / "premium_15m.csv").exists() and (SPOT is None or re.sub(r"^1000+", "", s) in SPOT)]
    D = {s: load(root, s) for s in syms}
    end = max(d["close"].index.max() for d in D.values())
    held, weeks = [], []
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
        keep = [s for s in held if s in ranked[:EXIT_RANK]]
        new = keep + [s for s in ranked if s not in keep][:TOP - len(keep)]
        changed = set(new) ^ set(held)
        held = new
        pos = []
        for s in new:
            d = D[s]
            pos.append(dict(symbol=s,
                            fund=d["fr"][(d["fr"].index > t0) & (d["fr"].index <= t1)].sum(),
                            basis=-(d["prem"][t1] - d["prem"][t0]),
                            maxup=d["high"][(d["high"].index > t0) & (d["high"].index <= t1)].max() / d["close"][t0] - 1,
                            hot=bool(d["hot"].get(t0, False))))
        weeks.append(dict(week=t0, pos=pos, changes=len(changed)))

    print(f"\n==================== {label} — 대상 {len(syms)}개, {len(weeks)}주 ====================")
    rows = []
    for name, lev in SCHEMES.items():
        rets, liqs, npos, hotpos = [], 0, 0, 0
        for w in weeks:
            if not w["pos"]:
                rets.append(0.0); continue
            pnl, cap = 0.0, 0.0
            for p in w["pos"]:
                L = lev(p["hot"])
                n = 1 / TOP                                    # 코인당 명목
                cap += n * (1 + 1 / L)
                r = p["fund"] + p["basis"]
                if p["maxup"] >= 1 / L - MMR:
                    r -= LIQ_FEE + ONE_WAY; liqs += 1
                pnl += n * r
                npos += 1; hotpos += p["hot"]
            pnl -= w["changes"] * ONE_WAY / TOP
            rets.append(pnl / cap)
        x = pd.Series(rets)
        eq = x.cumsum()
        sh = x.mean() / x.std() * np.sqrt(52)
        sharpes.setdefault(label, {})[name] = sh
        rows.append(dict(방식=name, 자본대비_연수익=f"{x.mean() * 52:+.1%}", 샤프=round(sh, 2),
                         최대낙폭=f"{(eq - eq.cummax()).min():+.1%}", 최악의주=f"{x.min():+.2%}",
                         청산=f"{liqs} / {npos}"))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"(보유 포지션 중 과열 진입: {hotpos} / {npos})")

ok = all(v["적응형 (과열 1배 / 평소 2배)"] > v["전부 2배"] for v in sharpes.values()) and len(sharpes) == 2
print("\n최종 판정 (적응형 샤프 > 전부 2배 샤프, 두 데이터 모두):", "통과" if ok else "실패")
