"""펀딩비 캐리 (현물 롱 + 선물 숏, 가격 방향 중립) — 규칙 미리 확정
손익 (명목금액 대비, 코인별):
  + 그 주 펀딩비 합계 (선물 숏이 받음, 음수면 냄)
  - 베이시스 변화 (프리미엄 인덱스: 선물이 현물 지수보다 더 비싸지면 숏 쪽 손해)
  - 비용: 새로 들어가거나 빠지는 코인만 (현물 0.1% + 선물 0.05% + 슬리피지 0.02%×2) 편도
규칙:
  - 매주 월요일 00:00 UTC, 지난 7일 펀딩비 합계 상위 3개 (합계 > 0 인 코인만, 바이낸스 현물 상장 코인만)
  - 코인마다 같은 금액, 1주 보유 (다음 주에도 뽑히면 그대로 보유)
자본 대비 수익: 선물 레버리지 L → 필요 자본 = 명목 × (1 + 1/L)
청산 위험: 보유 중 최고가 상승률 >= 1/L - 0.5%(유지증거금) 이면 '청산 위험' (현물 이익을 선물 계좌로 옮기지 않는다고 가정)
           진입 시점 직전 24시간 과열 신호 유무로 나눠서 비교
데이터: 각 폴더 klines_15m.csv, funding.csv, premium_15m.csv (download_premium.py), metrics.csv
판정: 비용 후 주간 순수익 > 0 + t >= 2 를 두 데이터 모두에서 → 통과
사용법: python carry.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import backtest as bt

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
TOP = 3
ONE_WAY = 0.001 + 0.0005 + 0.0002 * 2        # 현물 + 선물 + 슬리피지 양쪽
WEEK, DAY = pd.Timedelta("7D"), 96
LEVS = [1, 2, 3, 5, 10]
MMR = 0.005

def spot_list():
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=30).json()
        return {s["symbol"] for s in r["symbols"] if s.get("status") == "TRADING" and s["quoteAsset"] == "USDT"}
    except Exception as e:
        print("현물 목록을 못 받았어요 — 모든 코인을 대상으로 진행:", e)
        return None

def spot_name(perp):
    return re.sub(r"^1000+", "", perp)            # 1000PEPEUSDT → PEPEUSDT

SPOT = spot_list()
pd.set_option("display.width", 220)
verdict = []
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split()
            if (root / s / "klines_15m.csv").exists() and (root / s / "premium_15m.csv").exists()]
    no_spot = [s for s in syms if SPOT is not None and spot_name(s) not in SPOT]
    syms = [s for s in syms if s not in no_spot]
    D = {}
    for s in syms:
        bt.D = root / s
        k = bt.load()
        rsi = bt.rsi(k["close"], bt.P["rsi_len"])
        sig = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
               & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
        k = k.set_index("t")
        p = pd.read_csv(root / s / "premium_15m.csv")
        prem = pd.Series(p["close"].values, index=pd.to_datetime(p["close_time"] + 1, unit="ms").astype("datetime64[ns]"))
        f = pd.read_csv(root / s / "funding.csv")
        fr = pd.Series(f["last_funding_rate"].values,
                       index=pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[ns]")).sort_index()
        D[s] = dict(close=k["close"], high=k["high"], prem=prem, fr=fr,
                    hot=pd.Series(sig.values, index=k.index).astype(int).rolling(DAY, min_periods=1).sum() > 0)

    start = pd.Timestamp("2023-11-13")
    end = max(d["close"].index.max() for d in D.values())
    rows, pos_rows, held = [], [], set()
    for t0 in pd.date_range(start, end - WEEK, freq="W-MON"):
        t1 = t0 + WEEK
        cand = []
        for s, d in D.items():
            if t0 not in d["close"].index or t1 not in d["close"].index or t0 not in d["prem"].index or t1 not in d["prem"].index:
                continue
            past = d["fr"][(d["fr"].index > t0 - WEEK) & (d["fr"].index <= t0)]
            if past.size < 7:
                continue
            cand.append((s, past.sum()))
        pick = [s for s, v in sorted(cand, key=lambda x: -x[1])[:TOP] if v > 0]
        changed = len(set(pick) ^ held)
        cost = changed * ONE_WAY / max(TOP, 1)       # 포트폴리오 명목 대비 (코인당 1/TOP 비중)
        held = set(pick)
        if not pick:
            rows.append(dict(week=t0, n=0, 펀딩비=0.0, 베이시스=0.0, 비용=-cost, 순수익=-cost)); continue
        fund, basis = [], []
        for s in pick:
            d = D[s]
            fsum = d["fr"][(d["fr"].index > t0) & (d["fr"].index <= t1)].sum()
            db = d["prem"][t1] - d["prem"][t0]
            fund.append(fsum); basis.append(-db)
            maxup = d["high"][(d["high"].index > t0) & (d["high"].index <= t1)].max() / d["close"][t0] - 1
            pos_rows.append(dict(week=t0, symbol=s, maxup=maxup, hot=bool(d["hot"].get(t0, False)), fund=fsum))
        w = 1 / TOP
        rows.append(dict(week=t0, n=len(pick), 펀딩비=w * sum(fund), 베이시스=w * sum(basis),
                         비용=-cost, 순수익=w * (sum(fund) + sum(basis)) - cost))

    wk = pd.DataFrame(rows).set_index("week")
    ps = pd.DataFrame(pos_rows)
    print(f"\n==================== {label} — {len(wk)}주 ====================")
    print(f"현물 없음으로 제외: {', '.join(no_spot) if no_spot else '없음'}  | 대상 {len(syms)}개 | 평균 보유 {wk.n.mean():.1f}개")
    out = []
    for col in ["펀딩비", "베이시스", "비용", "순수익"]:
        x = wk[col]
        t = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else np.nan
        out.append(dict(항목=col, 주평균=f"{x.mean():+.3%}", 연환산_명목대비=f"{x.mean() * 52:+.1%}",
                        주간승률=f"{(x > 0).mean():.0%}" if col != "비용" else "", t값=round(t, 2) if col != "비용" else ""))
    print("[명목금액 대비 손익]")
    print(pd.DataFrame(out).to_string(index=False))
    x = wk.순수익
    eq = x.cumsum()
    print(f"최악의 주 {x.min():+.2%} | 최대낙폭 {(eq - eq.cummax()).min():+.1%} | 샤프 {x.mean() / x.std() * np.sqrt(52):.2f}")
    verdict.append((x.mean(), x.mean() / (x.std() / np.sqrt(len(x)))))

    print("\n[레버리지별 — 자본 대비 연수익 & 청산 위험 (현물 이익을 옮기지 않을 때)]")
    lr = []
    for L in LEVS:
        th = 1 / L - MMR
        liq = ps.maxup >= th
        lr.append(dict(선물레버리지=f"{L}배", 자본대비_연수익=f"{x.mean() * 52 / (1 + 1 / L):+.1%}",
                       청산선=f"+{th:.1%}", 청산위험_포지션=f"{liq.sum()} / {len(ps)} ({liq.mean():.1%})",
                       과열때=f"{liq[ps.hot].mean():.1%}" if ps.hot.any() else "-",
                       평소=f"{liq[~ps.hot].mean():.1%}"))
    print(pd.DataFrame(lr).to_string(index=False))
    q = ps.maxup.quantile([0.5, 0.9, 0.99]).values
    print(f"보유 1주 중 최고 상승률: 중앙 {q[0]:+.1%} / 상위10% {q[1]:+.1%} / 상위1% {q[2]:+.1%} / 최대 {ps.maxup.max():+.1%}")
    print(f"진입 시 과열 신호 있던 포지션: {ps.hot.sum()} / {len(ps)} | 그 포지션들의 주간 펀딩비 평균 "
          f"{ps[ps.hot].fund.mean():+.3%} vs 평소 {ps[~ps.hot].fund.mean():+.3%}")
    wk.to_csv(root / "carry_weeks.csv"); ps.to_csv(root / "carry_positions.csv", index=False)

ok = len(verdict) == 2 and all(m > 0 and t >= 2 for m, t in verdict)
print("\n최종 판정:", "통과" if ok else
      "방향만 맞음 (약함)" if len(verdict) == 2 and all(m > 0 for m, _ in verdict) else "실패")
