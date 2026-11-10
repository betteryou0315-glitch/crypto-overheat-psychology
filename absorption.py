"""청산 흡수 탐색 (롱 과열 → 롱 청산 투매 → 받아냈나?) — 탐색용, data_new 는 사용 안 함
1) 롱 과열 신호 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만) 후 24시간
2) 청산 투매 = 첫 번째로 아래 조건을 모두 만족한 5분 (1분 체결 요약 사용)
   - 5분 동안 가격 -1.5% 이상 하락 (5분 전 종가 대비 이번 종가)
   - 5분 시장가 매도 금액 >= 직전 4시간 '5분 시장가 매도' 중앙값 × 3  (직전 2시간 이상 데이터 필요)
   - 그 시점이 속한 15분봉 OI 가 직전 15분봉 대비 -1% 이상 감소
3) 투매 직후 15분으로 분류
   - 흡수 : 투매 저점 아래로 0.3% 넘게 안 빠짐 + 그 15분 시장가 매도 >= 시장가 매수
   - 반등 : 저점 유지 + 시장가 매수 > 매도
   - 붕괴 : 저점 아래로 0.3% 넘게 더 빠짐
4) 결과: 투매 후 15분 시점 가격 기준 4시간 / 24시간 수익률, 24시간 안에 +3% / -3% 중 먼저 닿은 쪽 (1분봉 → 없으면 15분봉)
대상: data, data_holdout, data_final (agg1m 필요 — download_agg1m.py)
사용법: python absorption.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목")]
DROP, MULT, OI_DROP, HOLD_TOL, AFTER, COOLDOWN = -0.015, 3.0, -0.01, 0.003, 15, 16
DAY = pd.Timedelta("24h")

def episodes(k):
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    s = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
         & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return np.flatnonzero((s & ~prev).values)

def load_minutes(dirp, days):
    parts = []
    for d in days:
        f = dirp / f"{d:%Y-%m-%d}.csv.gz"
        if f.exists():
            parts.append(pd.read_csv(f))
    if not parts:
        return None
    m = pd.concat(parts, ignore_index=True).drop_duplicates("minute").sort_values("minute")
    m["t"] = pd.to_datetime(m["minute"], unit="ms").astype("datetime64[ns]")
    return m.set_index("t")

def analyze(folder):
    """폴더 하나의 청산 투매 사건 표 (DataFrame) 와 (과열 사건 수, 투매 수) 반환"""
    root = Path(folder)
    rows, nflush, nev = [], 0, 0
    if not (root / "symbols.txt").exists():
        return pd.DataFrame(), (0, 0)
    for s in (root / "symbols.txt").read_text().split():
        if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"]):
            continue
        bt.D = root / s
        k = bt.load()
        oi15 = pd.Series(k["oi"].values, index=k["t"])            # 봉 마감 시각 기준
        oi_chg = oi15 / oi15.shift(1) - 1
        kc = pd.Series(k["close"].values, index=k["t"])
        kh, kl = pd.Series(k["high"].values, index=k["t"]), pd.Series(k["low"].values, index=k["t"])
        for e in episodes(k):
            ts = k["t"].iloc[e]
            m = load_minutes(root / s / "agg1m", [(ts - DAY).normalize(), ts.normalize(), (ts + DAY).normalize()])
            if m is None:
                continue
            nev += 1
            m = m.asfreq("1min")
            m["close"] = m["close"].ffill()
            m["low"] = m["low"].fillna(m["close"]); m["high"] = m["high"].fillna(m["close"])
            m[["buy", "sell"]] = m[["buy", "sell"]].fillna(0)
            r5 = m["close"] / m["close"].shift(5) - 1
            sell5 = m["sell"].rolling(5).sum()
            base = sell5.rolling(240, min_periods=120).median().shift(5)
            win = m.loc[(m.index > ts) & (m.index <= ts + DAY)]
            cand = win.index[(r5.reindex(win.index) <= DROP) & (sell5.reindex(win.index) >= MULT * base.reindex(win.index))]
            hit = None
            for t in cand:
                bar_end = t.ceil("15min") if t != t.floor("15min") else t
                oc = oi_chg.get(bar_end, np.nan)
                if oc <= OI_DROP:
                    hit = t; break
            if hit is None:
                continue
            nflush += 1
            lo = m.loc[hit - pd.Timedelta(minutes=4):hit, "low"].min()
            after = m.loc[hit + pd.Timedelta(minutes=1):hit + pd.Timedelta(minutes=AFTER)]
            if len(after) < AFTER:
                continue
            broke = after["low"].min() < lo * (1 - HOLD_TOL)
            sell_dom = after["sell"].sum() >= after["buy"].sum()
            kind = "붕괴" if broke else ("흡수" if sell_dom else "반등")
            t0 = hit + pd.Timedelta(minutes=AFTER)
            p0 = m.loc[t0, "close"]
            fut = m.loc[(m.index > t0) & (m.index <= t0 + DAY)]
            # 1분 데이터가 24시간을 다 덮지 못하면 15분봉으로 이어서
            r4 = r24 = np.nan
            p4 = m["close"].get(t0 + pd.Timedelta("4h"), np.nan)
            if np.isnan(p4):
                p4 = kc.asof(t0 + pd.Timedelta("4h"))
            p24 = kc.asof(t0 + DAY)
            r4, r24 = p4 / p0 - 1, p24 / p0 - 1
            hi_seq = pd.concat([fut["high"], kh[(kh.index > (fut.index.max() if len(fut) else t0)) & (kh.index <= t0 + DAY)]])
            lo_seq = pd.concat([fut["low"], kl[(kl.index > (fut.index.max() if len(fut) else t0)) & (kl.index <= t0 + DAY)]])
            up_hit = np.flatnonzero(hi_seq.values >= p0 * 1.03)
            dn_hit = np.flatnonzero(lo_seq.values <= p0 * 0.97)
            fu, fd = (up_hit[0] if len(up_hit) else 10**9), (dn_hit[0] if len(dn_hit) else 10**9)
            first = "위" if fu < fd else ("아래" if fd < fu else "안 터짐")
            rows.append(dict(symbol=s, time=hit, kind=kind, r4=r4, r24=r24, first=first,
                             drop5=r5[hit], sell_x=sell5[hit] / base[hit]))
    return pd.DataFrame(rows), (nev, nflush)

def table(d):
    out = []
    for kind in ["흡수", "반등", "붕괴"]:
        g = d[d.kind == kind] if len(d) else d
        if len(g) == 0:
            out.append(dict(분류=kind, 건수=0)); continue
        q90 = g.r24.quantile(0.9)
        out.append(dict(분류=kind, 건수=len(g),
                        먼저_위3=f"{(g['first'] == '위').mean():.0%}", 먼저_아래3=f"{(g['first'] == '아래').mean():.0%}",
                        평균4h=f"{g.r4.mean():+.2%}", 중앙4h=f"{g.r4.median():+.2%}",
                        평균24h=f"{g.r24.mean():+.2%}", 중앙24h=f"{g.r24.median():+.2%}",
                        상위10퍼제외24h=f"{g.r24[g.r24 < q90].mean():+.2%}" if len(g) >= 10 else "-"))
    return pd.DataFrame(out).fillna("")

if __name__ == "__main__":
    pd.set_option("display.width", 230)
    for folder, label in FOLDERS:
        if not (Path(folder) / "symbols.txt").exists():
            continue
        d, (nev, nflush) = analyze(folder)
        print(f"\n==================== {label} — 과열 사건 {nev} 중 청산 투매 {nflush} ====================")
        if d.empty:
            continue
        print(table(d).to_string(index=False))
        d.to_csv(Path(folder) / "absorption_events.csv", index=False)
    print("\n※ 가설대로면 '흡수'의 먼저_위3 가 가장 높고 중앙값도 + 여야 함. 세 데이터 모두에서 같은 모양인지 확인")
