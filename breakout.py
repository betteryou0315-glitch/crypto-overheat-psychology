"""과열 신호 → 폭발 방향 따라가기 (돌파 추종)
1) 신호: 펀딩비>=0.02%, RSI14>=70, OI 1시간 증가 (연속 신호는 첫 봉만)
2) 대기: 신호 봉 종가(기준가)에서 WAIT 시간 안에
     기준가 +B% 돌파 → 롱 / 기준가 -B% 이탈 → 숏
     (같은 봉에서 둘 다 닿으면 방향 불명 → 건너뜀)
3) 청산: 추적 손절 — 진입 후 가장 유리했던 가격에서 T% 되돌리면 청산
         (처음엔 진입가에서 T% 손절), 최대 보유 MAX_HOLD 지나면 종가 청산
비용: 수수료 편도 0.04% + 슬리피지 편도 0.02%
사용법: python breakout.py [data폴더]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
B = 0.03          # 돌파 폭 3%
T = 0.03          # 추적 손절 3%
WAIT = 96         # 신호 후 대기 24시간 (15분봉 96개)
MAX_HOLD = 672    # 최대 보유 7일
COST = 2 * (0.0004 + 0.0002)
COOLDOWN = 16
GRID_B = [0.02, 0.03, 0.05]
GRID_T = [0.02, 0.03, 0.05]

def symbol_dirs():
    dirs = sorted(p for p in ROOT.iterdir() if (p / "klines_15m.csv").exists())
    if (ROOT / "symbols.txt").exists():
        keep = set((ROOT / "symbols.txt").read_text().split())
        dirs = [p for p in dirs if p.name in keep]
    return dirs or [ROOT]

def signals(k):
    k["rsi"] = bt.rsi(k["close"], bt.P["rsi_len"])
    s = ((k["funding"] >= bt.P["funding_min"]) & (k["rsi"] >= bt.P["rsi_min"])
         & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return np.flatnonzero((s & ~prev).values)

def simulate(o, h, l, c, times, sig_idx, b, t):
    n, trades, busy_until = len(c), [], -1
    for s in sig_idx:
        if s <= busy_until:
            continue
        ref = c[s]; up, dn = ref * (1 + b), ref * (1 - b)
        side = entry = ei = None
        for u in range(s + 1, min(s + 1 + WAIT, n)):
            hit_up, hit_dn = h[u] >= up, l[u] <= dn
            if hit_up and hit_dn:
                break                                   # 방향 불명
            if hit_up:
                side, entry, ei = 1, max(up, o[u]), u; break
            if hit_dn:
                side, entry, ei = -1, min(dn, o[u]), u; break
        if side is None:
            continue
        best = entry
        exit_px, xi, why = None, None, None
        # 진입 봉에서는 진입 이후 경로를 모르므로 그 봉 종가 기준으로만 갱신
        best = max(best, c[ei]) if side == 1 else min(best, c[ei])
        for u in range(ei + 1, min(ei + 1 + MAX_HOLD, n)):
            stop = best * (1 - t) if side == 1 else best * (1 + t)
            if side == 1 and l[u] <= stop:
                exit_px, xi, why = min(stop, o[u]), u, "추적손절"; break
            if side == -1 and h[u] >= stop:
                exit_px, xi, why = max(stop, o[u]), u, "추적손절"; break
            best = max(best, h[u]) if side == 1 else min(best, l[u])
        if exit_px is None:
            xi = min(ei + MAX_HOLD, n - 1); exit_px, why = c[xi], "시간청산"
        ret = side * (exit_px - entry) / entry - COST
        trades.append(dict(entry_time=times[ei], side="롱" if side == 1 else "숏",
                           entry=entry, exit=exit_px, ret=ret, why=why, hold_h=(xi - ei) / 4))
        busy_until = xi
    return trades

def stats(tr):
    if tr.empty:
        return dict(거래=0)
    r = tr.ret
    pf = r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else np.inf
    se = r.std() / np.sqrt(len(r)) if len(r) > 1 else np.nan
    return dict(거래=len(r), 승률=f"{(r > 0).mean():.0%}", 평균=f"{r.mean():+.2%}",
                중앙값=f"{r.median():+.2%}", 평균이익=f"{r[r>0].mean():+.2%}",
                평균손실=f"{r[r<=0].mean():+.2%}", PF=round(pf, 2), t값=round(r.mean() / se, 2),
                날짜수=tr.entry_time.dt.date.nunique())

# 데이터 한 번만 로드
loaded = []
for sd in symbol_dirs():
    bt.D = sd
    k = bt.load()
    loaded.append((sd.name, k, signals(k)))

def run_all(b, t):
    out = []
    for name, k, sig in loaded:
        tr = simulate(k.open.values, k.high.values, k.low.values, k.close.values,
                      k.t.values, sig, b, t)
        for x in tr:
            x["symbol"] = name
        out += tr
    df = pd.DataFrame(out)
    if not df.empty:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    return df

pd.set_option("display.width", 220)
tr = run_all(B, T)
print(f"=== 기본 설정: 돌파 {B:.0%}, 추적손절 {T:.0%}, 대기 {WAIT/4:.0f}시간 ===")
print(pd.DataFrame([dict(구분="전체", **stats(tr)),
                    dict(구분="롱", **stats(tr[tr.side == "롱"])) if not tr.empty else {},
                    dict(구분="숏", **stats(tr[tr.side == "숏"])) if not tr.empty else {}]).to_string(index=False))
if not tr.empty:
    print(f"\n누적(1배, 복리): {(1 + tr.sort_values('entry_time').ret).prod() - 1:+.1%}"
          f" | 평균 보유 {tr.hold_h.mean():.1f}시간 (중앙값 {tr.hold_h.median():.1f})"
          f" | 시간청산 {int((tr.why == '시간청산').sum())}건")
    print("\n--- 종목별 ---")
    print(pd.DataFrame([dict(종목=s, **stats(g)) for s, g in tr.groupby("symbol")]).to_string(index=False))
    tr.to_csv(ROOT / "breakout_trades.csv", index=False)

print("\n=== 파라미터 민감도 (전체 합산) — 최고값 고르기 X, 넓게 고르게 좋은지 보기 ===")
rows = []
for b in GRID_B:
    for t in GRID_T:
        g = run_all(b, t)
        s = stats(g)
        rows.append(dict(돌파=f"{b:.0%}", 추적손절=f"{t:.0%}", 거래=s.get("거래", 0),
                         승률=s.get("승률", "-"), 평균=s.get("평균", "-"), PF=s.get("PF", "-"),
                         t값=s.get("t값", "-")))
print(pd.DataFrame(rows).to_string(index=False))