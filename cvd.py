"""CVD(시장가 매수/매도 우위)로 과열 후 폭발 방향 지속 여부 예측 — 규칙 미리 확정 (2026-11-09)
가설: 과열 후 첫 폭발(±2%)이 시장가 주문에 떠밀린 것이면(체결 우위가 움직인 방향과 같으면) 그 방향으로 계속 간다
규칙:
  1) 기준가 = 과열 신호 15분봉 종가 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만)
  2) 폭발 시작 = 신호 후 24시간 안에 1분봉 고가/저가가 기준가 ±2%에 처음 닿은 1분 (같은 1분에 양쪽이면 제외)
  3) CVD 일치도 = 움직인 방향(+1/-1) × (직전 15분 시장가 매수 - 매도) / (매수 + 매도)   [폭발 시작 1분 포함]
  4) 결과 = 폭발 시작 1분 종가 기준, 이후 24시간 안에 같은 방향 3%가 먼저 → 지속(+1), 반대 3% 먼저 → 반전(-1), 그 외 0
            (15분봉으로 판정, 폭발 시작 이후 첫 15분봉부터)
판정: data_final 에서 순위상관 > 0 + 월 단위 군집 t >= 2.5 (최종 데이터 세 번째 가설 → 기준 상향)
      data, data_holdout 은 참고용
데이터: download_agg1m.py 로 만든 <폴더>/<종목>/agg1m/<날짜>.csv.gz
사용법: python cvd.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목"), ("data_final", "최종 15종목 (판정)")]
BREAK, NEXT, WIN, COOLDOWN = 0.02, 0.03, 15, 16
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
        if not f.exists():
            return None
        parts.append(pd.read_csv(f))
    m = pd.concat(parts, ignore_index=True)
    m["t"] = pd.to_datetime(m["minute"], unit="ms").astype("datetime64[ns]")   # 1분 시작 시각
    return m.sort_values("t").reset_index(drop=True)

def rank_t(d, col):
    d = d.dropna(subset=[col])
    if len(d) < 20:
        return len(d), np.nan, np.nan
    x = d[col].rank().values; y = d.score.rank().values
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = float(np.mean(x * y)); e = y - r * x
    g = pd.Series(x * e).groupby(d.time.dt.to_period("M").values).sum().values
    return len(d), r, r / (np.sqrt((g**2).sum()) / (x**2).sum())

def analyze(folder):
    """폴더 하나의 CVD 사건 표와 체결 데이터 없는 사건 수 반환 (규칙은 위 설명 그대로)"""
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        return pd.DataFrame(), 0
    ev, miss = [], 0
    for s in (root / "symbols.txt").read_text().split():
        if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"]):
            continue
        bt.D = root / s
        k = bt.load()
        kt, kh, kl, kc = k["t"], k["high"].values, k["low"].values, k["close"].values
        for e in episodes(k):
            ts = kt.iloc[e]
            if e + 2 * 96 >= len(kc):
                continue
            m = load_minutes(root / s / "agg1m", [ts.normalize(), (ts + DAY).normalize()])
            if m is None:
                miss += 1; continue
            p0 = kc[e]
            w = m[(m.t >= ts) & (m.t < ts + DAY)]
            up = w.high.values >= p0 * (1 + BREAK)
            dn = w.low.values <= p0 * (1 - BREAK)
            hit = np.flatnonzero(up | dn)
            if not len(hit):
                ev.append(dict(symbol=s, time=ts, kind="안 터짐")); continue
            i = hit[0]
            if up[i] and dn[i]:
                ev.append(dict(symbol=s, time=ts, kind="불명")); continue
            d = 1 if up[i] else -1
            tb = w.t.iloc[i]
            win = m[(m.t > tb - pd.Timedelta(minutes=WIN)) & (m.t <= tb)]
            tot = win.buy.sum() + win.sell.sum()
            imb = (win.buy.sum() - win.sell.sum()) / tot if tot > 0 else np.nan
            entry = w.close.iloc[i]
            # 결과: 폭발 시작 1분이 끝난 뒤 시작하는 첫 15분봉부터 96개
            j0 = int(np.searchsorted(kt.values, (tb + pd.Timedelta(minutes=1)).to_datetime64(), side="left"))
            # kt 는 봉 마감 시각 → 봉 시작 = kt - 15분. 시작이 tb+1분 이후인 첫 봉
            while j0 < len(kt) and kt.iloc[j0] - pd.Timedelta(minutes=15) < tb + pd.Timedelta(minutes=1):
                j0 += 1
            tgt, stp = entry * (1 + d * NEXT), entry * (1 - d * NEXT)
            score = 0
            for j in range(j0, min(j0 + 96, len(kc))):
                a = kh[j] >= tgt if d == 1 else kl[j] <= tgt
                b = kl[j] <= stp if d == 1 else kh[j] >= stp
                if a and b: score = 0; break
                if a: score = 1; break
                if b: score = -1; break
            ev.append(dict(symbol=s, time=ts, kind="위" if d == 1 else "아래", imb=imb, agree=d * imb,
                           score=score, mins_to_break=(tb - ts).total_seconds() / 60))
    return pd.DataFrame(ev), miss

def summarize(df, label, miss):
    br = df[df.kind.isin(["위", "아래"])].dropna(subset=["agree"])
    n, r, t = rank_t(br, "agree")
    q = br.assign(구간=pd.qcut(br.agree.rank(method="first"), 3, labels=["약함/반대", "중간", "강함"]))
    g = q.groupby("구간", observed=True).score.agg(건수="size", 지속=lambda x: (x == 1).mean(), 반전=lambda x: (x == -1).mean())
    row = dict(데이터=label, 사건=len(df), 체결없음=miss, 폭발=len(br),
               폭발까지_중앙=f"{br.mins_to_break.median():.0f}분",
               전체_지속=f"{(br.score == 1).mean():.0%}", 전체_반전=f"{(br.score == -1).mean():.0%}",
               순위상관=round(r, 3), 군집t=round(t, 2),
               **{f"{i} 지속-반전": f"{g.loc[i, '지속'] - g.loc[i, '반전']:+.0%}" for i in g.index})
    return row, r, t

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    rows, final = [], None
    for folder, label in FOLDERS:
        root = Path(folder)
        if not (root / "symbols.txt").exists():
            print(f"[{label}] 없음"); continue
        df, miss = analyze(folder)
        if df.empty:
            print(f"[{label}] 사건 없음 (체결 데이터 없음 {miss}건)"); continue
        row, r, t = summarize(df, label, miss)
        rows.append(row)
        df.to_csv(root / "cvd_events.csv", index=False)
        if folder == "data_final":
            final = (r, t)
    print(pd.DataFrame(rows).to_string(index=False))
    print("※ 가설대로면 순위상관 +, '강함' 구간의 지속-반전이 가장 높아야 함")
    if final is None:
        print("\n최종 판정: 최종 15종목 결과 없음")
    else:
        r, t = final
        print("\n최종 판정 (기준 t >= 2.5):", "통과" if r > 0 and t >= 2.5 else ("방향만 맞음 (약함)" if r > 0 else "실패"))
