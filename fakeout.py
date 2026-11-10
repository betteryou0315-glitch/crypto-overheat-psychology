"""과열 후 가짜 돌파 역매매 — 규칙 미리 확정
1) 과열 신호 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만)
2) 기준선 = 신호 봉까지 24시간 최고가 H / 최저가 L
3) 신호 후 24시간 안에 처음으로 H 위 또는 L 아래로 나간 봉 = 돌파 봉
4) 돌파 봉부터 1시간(4봉) 안에 종가가 기준선 안으로 돌아오면 '가짜 돌파' → 반대로 진입 (그 봉 종가)
     위로 돌파 후 복귀 → 숏 / 아래로 이탈 후 복귀 → 롱
   1시간 안에 안 돌아오면 진짜 돌파로 보고 거래 안 함
5) 손절 = 돌파 이후 가장 멀리 간 가격 (위: 최고가, 아래: 최저가), 익절 = 손절폭 × 2
6) 둘 다 안 닿으면 24시간 뒤 종가 청산, 같은 봉에서 둘 다 닿으면 손절, 비용 편도 0.07%
판정: 기존 30종목(data + data_holdout) 거래당 평균 > 0
      AND 최종 15종목(data_final) 거래당 평균 > 0 + 월 단위 군집 t >= 2  → 통과
사용법: python fakeout.py   (data_final 이 없으면 기존 30종목만 출력)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목 (31~45위)")]
LOOK, CONFIRM, HOLD, COOLDOWN = 96, 4, 96, 16
COST = 2 * 0.0007

def episodes(k):
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    s = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
         & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return np.flatnonzero((s & ~prev).values)

def trade(o, h, l, c, s):
    n = len(c)
    if s < LOOK:
        return None
    H, L = h[s - LOOK + 1:s + 1].max(), l[s - LOOK + 1:s + 1].min()
    for u in range(s + 1, min(s + 1 + LOOK, n)):
        up, dn = h[u] > H, l[u] < L
        if not (up or dn):
            continue
        if up and dn:
            return None                                   # 한 봉에 양쪽 돌파 — 판단 불가
        side = -1 if up else 1                            # 가짜 돌파면 반대로
        ext = h[u] if up else l[u]
        for v in range(u, min(u + CONFIRM + 1, n)):
            ext = max(ext, h[v]) if up else min(ext, l[v])
            back = c[v] < H if up else c[v] > L
            if back:
                entry = c[v]
                stop = ext
                risk = abs(stop - entry)
                if risk <= 0:
                    return None
                tp = entry + side * 2 * risk
                for w in range(v + 1, min(v + 1 + HOLD, n)):
                    hit_sl = h[w] >= stop if side == -1 else l[w] <= stop
                    hit_tp = l[w] <= tp if side == -1 else h[w] >= tp
                    if hit_sl:
                        px = max(stop, o[w]) if side == -1 else min(stop, o[w]); why = "손절"; break
                    if hit_tp:
                        px = tp; why = "익절"; break
                else:
                    w = min(v + HOLD, n - 1); px = c[w]; why = "시간"
                ret = side * (px - entry) / entry - COST
                return dict(entry_i=v, exit_i=w, side=side, ret=ret, risk=risk / entry, why=why,
                            R=side * (px - entry) / risk)
        return dict(none="진짜 돌파")                       # 1시간 안에 안 돌아옴
    return dict(none="돌파 없음")

def cluster_t(x, times):
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.Series(times).dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x)) if len(x) > 2 else np.nan

def summary(d, name):
    if len(d) < 5:
        return dict(구분=name, 거래=len(d))
    r = d.ret
    m, t = cluster_t(r.values, d.time)
    return dict(구분=name, 거래=len(d), 승률=f"{(d.why == '익절').mean():.0%}", 손절=f"{(d.why == '손절').mean():.0%}",
                시간청산=f"{(d.why == '시간').mean():.0%}", 평균R=f"{d.R.mean():+.2f}", 손절폭중앙=f"{d.risk.median():.2%}",
                거래당=f"{m:+.3%}", PF=round(r[r > 0].sum() / -r[r <= 0].sum(), 2) if (r <= 0).any() else np.inf,
                군집t=round(t, 2))

pd.set_option("display.width", 220)
res = {}
for folder, label in FOLDERS:
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        print(f"\n[{label}] {folder} 없음 — 건너뜀"); continue
    syms = [s for s in (root / "symbols.txt").read_text().split() if (root / s / "klines_15m.csv").exists()]
    rows, cnt = [], {"진짜 돌파": 0, "돌파 없음": 0, "판단 불가": 0}
    for s in syms:
        bt.D = root / s
        k = bt.load()
        o, h, l, c = (k[x].values for x in ["open", "high", "low", "close"])
        busy = -1
        for e in episodes(k):
            if e <= busy:
                continue
            r = trade(o, h, l, c, e)
            if r is None:
                cnt["판단 불가"] += 1; continue
            if "none" in r:
                cnt[r["none"]] += 1; continue
            busy = r["exit_i"]
            rows.append(dict(symbol=s, time=k["t"].iloc[r["entry_i"]], **{x: r[x] for x in ["side", "ret", "risk", "why", "R"]}))
    d = pd.DataFrame(rows)
    res[label] = d
    print(f"\n==================== {label} ({len(syms)}개) ====================")
    print(f"가짜 돌파 거래 {len(d)} | 진짜 돌파(안 돌아옴) {cnt['진짜 돌파']} | 24시간 내 돌파 없음 {cnt['돌파 없음']} | 판단 불가 {cnt['판단 불가']}")
    if len(d):
        print(pd.DataFrame([summary(d, "전체"), summary(d[d.side == -1], "숏 (위 가짜 돌파)"),
                            summary(d[d.side == 1], "롱 (아래 가짜 이탈)")]).fillna("").to_string(index=False))
        d.to_csv(root / "fakeout_trades.csv", index=False)

disc = pd.concat([res[l] for l in ["기존 15종목", "검증 15종목"] if l in res and len(res[l])], ignore_index=True)
if len(disc):
    m, t = cluster_t(disc.ret.values, disc.time)
    print(f"\n[기존 30종목 합산] 거래 {len(disc)} | 거래당 {m:+.3%} | 군집t {t:.2f}")
fin = res.get("최종 15종목 (31~45위)")
if fin is not None and len(fin):
    mf, tf = cluster_t(fin.ret.values, fin.time)
    ok = len(disc) and disc.ret.mean() > 0 and mf > 0 and tf >= 2
    print("최종 판정:", "통과" if ok else ("방향만 맞음 (약함)" if len(disc) and disc.ret.mean() > 0 and mf > 0 else "실패"))
else:
    print("최종 판정: data_final 이 아직 없음 — python download_final.py 후 다시 실행")
