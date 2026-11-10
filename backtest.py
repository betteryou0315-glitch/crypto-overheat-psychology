"""저항구간 숏 백테스트
진입: 펀딩비>=0.02%, RSI14>=70, OI(현재)>OI(1시간 전), 이중 저항구간 터치 후 종가 기준 숏
청산: 익절 진입가-5%, 손절 저항선+2%  (같은 봉에서 둘 다 닿으면 손절로 처리 = 보수적)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data")

P = dict(
    funding_min=0.0002,   # 0.02%
    rsi_len=14, rsi_min=70,
    oi_lag=4,             # 15분봉 4개 = 1시간
    k=4,                  # 고점 확인: 앞뒤 4봉(1시간)
    zone_tol=0.003,       # 고점끼리 ±0.3%
    min_gap=8,            # 두 고점 최소 간격 8봉(2시간)
    lookback=672,         # 7일
    tp=0.05, sl=0.02,
    fee=0.0004,           # 편도 0.04% (taker)
)

def load():
    k = pd.read_csv(D/"klines_15m.csv")
    k["t"] = pd.to_datetime(k["close_time"] + 1, unit="ms").astype("datetime64[ns]")   # 봉 마감 시각
    k = k[["t","open","high","low","close"]].reset_index(drop=True)

    f = pd.read_csv(D/"funding.csv")
    f["ft"] = pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[ns]")
    f = f[["ft","last_funding_rate"]].rename(columns={"last_funding_rate":"funding"})

    m = pd.read_csv(D/"metrics.csv")
    m["mt"] = pd.to_datetime(m["create_time"]).astype("datetime64[ns]")
    m = m[["mt","sum_open_interest"]].rename(columns={"sum_open_interest":"oi"})

    # 봉 마감 시점에 '이미 알 수 있는' 값만 붙임 (backward asof)
    k = pd.merge_asof(k.sort_values("t"), f.sort_values("ft"), left_on="t", right_on="ft", direction="backward")
    k = pd.merge_asof(k, m.sort_values("mt"), left_on="t", right_on="mt", direction="backward",
                      tolerance=pd.Timedelta("15min"))
    return k.drop(columns=["ft","mt"])

def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def pivot_highs(high, k):
    h = high.values
    n = len(h)
    piv = np.zeros(n, bool)
    for i in range(k, n - k):
        if h[i] > h[i-k:i].max() and h[i] > h[i+1:i+k+1].max():
            piv[i] = True
    return piv

def find_zone(t, piv_idx, h, close):
    """t 시점에 확정된 고점들 중 이중 저항구간 레벨 반환 (없으면 None)"""
    k, lb = P["k"], P["lookback"]
    cand = piv_idx[(piv_idx <= t - k) & (piv_idx >= t - lb)]   # 확정된 것만
    best = None
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            i, j = cand[a], cand[b]
            if j - i < P["min_gap"]:
                continue
            lvl = max(h[i], h[j])
            if abs(h[i] - h[j]) / lvl > P["zone_tol"]:
                continue
            # 두번째 고점 이후 ~ 직전 봉까지 종가로 돌파된 적 없어야 '살아있는' 저항
            if j + 1 < t and close[j+1:t].max() > lvl * (1 + P["zone_tol"]):
                continue
            # 지금 봉이 그 구간을 터치했고, 종가로 뚫지 못함
            if h[t] >= lvl * (1 - P["zone_tol"]) and close[t] < lvl * (1 + P["zone_tol"]):
                if best is None or lvl < best:
                    best = lvl
    return best

def run(k):
    k["rsi"] = rsi(k["close"], P["rsi_len"])
    k["oi_up"] = k["oi"] > k["oi"].shift(P["oi_lag"])
    sig = (k["funding"] >= P["funding_min"]) & (k["rsi"] >= P["rsi_min"]) & k["oi_up"]

    h, lo, c = k["high"].values, k["low"].values, k["close"].values
    piv_idx = np.flatnonzero(pivot_highs(k["high"], P["k"]))

    trades, t, n = [], 0, len(k)
    sig_idx = np.flatnonzero(sig.values)
    stats = dict(base_signals=len(sig_idx), zone_hits=0)
    pos_end = -1
    for t in sig_idx:
        if t <= pos_end:                 # 포지션 보유 중이면 스킵 (한 번에 1개)
            continue
        lvl = find_zone(t, piv_idx, h, c)
        if lvl is None:
            continue
        stats["zone_hits"] += 1
        entry = c[t]
        tp, sl = entry * (1 - P["tp"]), lvl * (1 + P["sl"])
        exit_px, exit_i, why = None, None, None
        for u in range(t + 1, n):
            if h[u] >= sl:
                exit_px, exit_i, why = sl, u, "SL"; break
            if lo[u] <= tp:
                exit_px, exit_i, why = tp, u, "TP"; break
        if exit_px is None:
            exit_px, exit_i, why = c[-1], n - 1, "OPEN"
        ret = (entry - exit_px) / entry - 2 * P["fee"]
        trades.append(dict(entry_time=k["t"][t], exit_time=k["t"][exit_i], entry=entry,
                           level=lvl, tp=tp, sl=sl, exit=exit_px, result=why, ret=ret,
                           hold_h=(exit_i - t) / 4, funding=k["funding"][t], rsi=k["rsi"][t]))
        pos_end = exit_i
    return pd.DataFrame(trades), stats

def report(tr, stats, k):
    print(f"기간: {k['t'].iloc[0]} ~ {k['t'].iloc[-1]}  ({len(k)}봉)")
    print(f"펀딩+RSI+OI 동시 충족 봉: {stats['base_signals']}  → 저항구간까지 충족: {stats['zone_hits']}")
    if tr.empty:
        print("거래 없음"); return
    closed = tr[tr.result != "OPEN"]
    eq = (1 + tr["ret"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    print(f"거래 수: {len(tr)} (TP {sum(tr.result=='TP')}, SL {sum(tr.result=='SL')}, 미청산 {sum(tr.result=='OPEN')})")
    print(f"승률: {(closed.ret>0).mean():.1%}")
    print(f"평균 수익: {tr.ret.mean():.2%}  | 평균 이익 {tr.ret[tr.ret>0].mean():.2%} / 평균 손실 {tr.ret[tr.ret<=0].mean():.2%}")
    print(f"누적 수익(1배, 복리): {eq.iloc[-1]-1:.2%}  | 최대 낙폭: {dd:.2%}")
    print(f"평균 보유: {tr.hold_h.mean():.1f}시간 (중앙값 {tr.hold_h.median():.1f})")
    print(f"평균 손절폭(진입가 대비): {((tr.sl/tr.entry)-1).mean():.2%}")
    pf = tr.ret[tr.ret>0].sum() / -tr.ret[tr.ret<=0].sum() if (tr.ret<=0).any() else np.inf
    print(f"Profit factor: {pf:.2f}")

def summary_row(sym, tr, stats):
    if tr.empty:
        return dict(종목=sym, 기본신호=stats["base_signals"], 거래=0)
    pf = tr.ret[tr.ret > 0].sum() / -tr.ret[tr.ret <= 0].sum() if (tr.ret <= 0).any() else np.inf
    return dict(종목=sym, 기본신호=stats["base_signals"], 거래=len(tr),
                TP=int((tr.result == "TP").sum()), SL=int((tr.result == "SL").sum()),
                승률=f"{(tr.ret > 0).mean():.0%}", 평균수익=f"{tr.ret.mean():.2%}",
                누적=f"{(1 + tr.ret).prod() - 1:.1%}", PF=round(pf, 2))

if __name__ == "__main__":
    root = D
    sym_dirs = sorted(p for p in root.iterdir() if (p / "klines_15m.csv").exists())
    if (root / "symbols.txt").exists():   # download.py 가 고른 종목만
        keep = set((root / "symbols.txt").read_text().split())
        sym_dirs = [p for p in sym_dirs if p.name in keep]
    if not sym_dirs:                      # 예전 단일 종목 구조 (data/klines_15m.csv)
        sym_dirs = [root]

    rows, all_tr = [], []
    for sd in sym_dirs:
        D = sd
        k = load()
        tr, stats = run(k)
        tr.insert(0, "symbol", sd.name)
        tr.to_csv(sd / "trades.csv", index=False)
        rows.append(summary_row(sd.name, tr, stats))
        all_tr.append(tr)

    pd.set_option("display.width", 200)
    print("=== 종목별 결과 ===")
    print(pd.DataFrame(rows).to_string(index=False))

    tr = pd.concat(all_tr, ignore_index=True).sort_values("entry_time")
    print("\n=== 전체 합산 ===")
    if tr.empty:
        print("거래 없음")
    else:
        n = len(tr)
        win = (tr.ret > 0).mean()
        pf = tr.ret[tr.ret > 0].sum() / -tr.ret[tr.ret <= 0].sum() if (tr.ret <= 0).any() else np.inf
        days = tr.entry_time.dt.date.nunique()
        print(f"거래 수: {n} (TP {sum(tr.result=='TP')}, SL {sum(tr.result=='SL')}, 미청산 {sum(tr.result=='OPEN')})")
        aw, al = tr.ret[tr.ret>0].mean(), -tr.ret[tr.ret<=0].mean()
        print(f"승률: {win:.1%}  (손익분기 약 {al/(al+aw):.0%})")
        print(f"평균 수익(거래당): {tr.ret.mean():.2%}  | Profit factor: {pf:.2f}")
        print(f"평균 보유: {tr.hold_h.mean():.1f}시간 (중앙값 {tr.hold_h.median():.1f})")
        print(f"거래가 발생한 날짜 수: {days}  (같은 날 여러 종목 동시 신호 → 사실상 같은 사건)")
        tr.to_csv(root / "trades_all.csv", index=False)