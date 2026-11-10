"""과열 신호 이후 가격이 실제로 내렸는가? (진입/청산 규칙 없이 신호 자체의 예측력만 확인)
- 신호: 펀딩비>=0.02%, RSI14>=70, OI 1시간 증가 (각각 단독 + 셋 다)
- 신호 시점 종가 대비 1시간 / 4시간 / 24시간 뒤 수익률
- 비교 기준: 아무 조건 없는 '전체 봉'의 평균 (시장 자체가 오른 효과를 빼고 봐야 하므로)
- 연속으로 켜진 신호는 첫 봉만 사용 (직전 4시간 안에 같은 신호가 없을 때만 = 독립 사건)
사용법: python forward_returns.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
HORIZONS = {"1h": 4, "4h": 16, "24h": 96}   # 15분봉 개수
COOLDOWN = 16                                # 4시간

def symbol_dirs():
    dirs = sorted(p for p in ROOT.iterdir() if (p / "klines_15m.csv").exists())
    if (ROOT / "symbols.txt").exists():
        keep = set((ROOT / "symbols.txt").read_text().split())
        dirs = [p for p in dirs if p.name in keep]
    return dirs or [ROOT]

def first_of_episode(s):
    """True 가 연속될 때 첫 봉만 남김 (직전 COOLDOWN 봉 안에 True 없을 때)"""
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return s & ~prev

rows = []
for sd in symbol_dirs():
    bt.D = sd
    k = bt.load()
    k["rsi"] = bt.rsi(k["close"], bt.P["rsi_len"])
    c = k["close"]
    fwd = {h: c.shift(-n) / c - 1 for h, n in HORIZONS.items()}

    sigs = {
        "전체 봉(기준)": pd.Series(True, index=k.index),
        "펀딩비>=0.02%": k["funding"] >= bt.P["funding_min"],
        "RSI>=70": k["rsi"] >= bt.P["rsi_min"],
        "OI 1시간 증가": k["oi"] > k["oi"].shift(bt.P["oi_lag"]),
    }
    sigs["셋 다"] = sigs["펀딩비>=0.02%"] & sigs["RSI>=70"] & sigs["OI 1시간 증가"]

    for name, s in sigs.items():
        s = s.fillna(False)
        if name != "전체 봉(기준)":
            s = first_of_episode(s)
        idx = np.flatnonzero(s.values)
        for h in HORIZONS:
            r = fwd[h].iloc[idx].dropna()
            rows.append(pd.DataFrame({"symbol": sd.name, "signal": name, "h": h, "ret": r.values}))

df = pd.concat(rows, ignore_index=True)

def summarize(g):
    r = g["ret"]
    se = r.std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
    return pd.Series({"건수": len(r), "평균": r.mean(), "중앙값": r.median(),
                      "하락비율": (r < 0).mean(), "t값": r.mean() / se if se else np.nan})

order = ["전체 봉(기준)", "펀딩비>=0.02%", "RSI>=70", "OI 1시간 증가", "셋 다"]
out = df.groupby(["h", "signal"]).apply(summarize, include_groups=False).reset_index()
out["signal"] = pd.Categorical(out["signal"], order, ordered=True)
out["h"] = pd.Categorical(out["h"], list(HORIZONS), ordered=True)
out = out.sort_values(["h", "signal"])

# 기준 대비 초과 수익 (시장 평균을 뺀 값)
base = out[out.signal == "전체 봉(기준)"].set_index("h")["평균"]
out["기준대비"] = out.apply(lambda x: x["평균"] - base[x["h"]], axis=1)

fmt = out.copy()
for col in ["평균", "중앙값", "기준대비"]:
    fmt[col] = fmt[col].map(lambda v: f"{v:+.3%}")
fmt["하락비율"] = fmt["하락비율"].map(lambda v: f"{v:.1%}")
fmt["t값"] = fmt["t값"].map(lambda v: f"{v:+.2f}")
fmt["건수"] = fmt["건수"].astype(int)
pd.set_option("display.width", 200)
print("=== 신호 이후 수익률 (전 종목 합산, 독립 사건 기준) ===")
print("※ 평균이 음수이고 기준보다 낮을수록 '숏에 유리'. |t값|이 2 이상이면 우연일 가능성이 낮음\n")
for h in HORIZONS:
    print(f"--- {h} 뒤 ---")
    print(fmt[fmt.h == h].drop(columns="h").to_string(index=False))
    print()

# 셋 다 신호의 종목별 24시간 결과
sub = df[(df.signal == "셋 다") & (df.h == "24h")]
if not sub.empty:
    per = sub.groupby("symbol")["ret"].agg(건수="size", 평균="mean", 하락비율=lambda r: (r < 0).mean())
    per["평균"] = per["평균"].map(lambda v: f"{v:+.2%}")
    per["하락비율"] = per["하락비율"].map(lambda v: f"{v:.0%}")
    print("--- '셋 다' 신호, 종목별 24시간 뒤 ---")
    print(per.to_string())