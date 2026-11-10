"""펀딩비로 코인 고르기 (시장 중립 롱숏) — 규칙 미리 확정
가설: 펀딩비가 높은(군중이 들뜬) 코인은 다음 주에 상대적으로 덜 오르고, 낮은(식은) 코인은 더 오른다
규칙:
  - 매주 월요일 00:00 UTC 리밸런싱
  - 신호 = 지난 7일 펀딩비 합계 (정산 주기가 8시간·4시간으로 달라도 합계로 비교 가능)
  - 펀딩비 상위 3개 숏 / 하위 3개 롱, 같은 금액 (롱 50% + 숏 50%)
  - 1주 보유. 손익 = 가격 차이 + 펀딩비 수취(숏은 받고 롱은 냄, 양수일 때 기준)
  - 비용: 매주 전부 교체 가정, 편도 0.07% (시장가 0.05% + 슬리피지 0.02%)
  - 그 주 시작 시점에 최소 7일치 펀딩비와 가격이 있는 코인만 후보
판정: 비용 후 주간 수익 > 0 + t >= 2 를 두 데이터 모두에서 → 통과
사용법: python funding_xs.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

FOLDERS = [("data", "기존 15종목"), ("data_holdout", "검증 15종목")]
TOP = 3
COST = 2 * 0.0007          # 포지션 하나 기준 왕복 비용 (매주)
WEEK = pd.Timedelta("7D")

def load_symbol(root, s):
    k = pd.read_csv(root / s / "klines_15m.csv", usecols=["close_time", "close"])
    px = pd.Series(k["close"].values,
                   index=pd.to_datetime(k["close_time"] + 1, unit="ms").astype("datetime64[ns]"))
    f = pd.read_csv(root / s / "funding.csv")
    fr = pd.Series(f["last_funding_rate"].values,
                   index=pd.to_datetime(f["calc_time"], unit="ms").astype("datetime64[ns]")).sort_index()
    return px, fr

def fsum(fr, a, b):
    """(a, b] 구간 펀딩비 합계"""
    return fr[(fr.index > a) & (fr.index <= b)].sum()

pd.set_option("display.width", 220)
verdict = []
for folder, label in FOLDERS:
    root = Path(folder)
    syms = [s for s in (root / "symbols.txt").read_text().split() if (root / s / "klines_15m.csv").exists()]
    data = {s: load_symbol(root, s) for s in syms}
    start = max(pd.Timestamp("2023-11-13"), min(px.index.min() for px, _ in data.values()) + WEEK)
    end = max(px.index.max() for px, _ in data.values())
    mondays = pd.date_range(start.normalize(), end - WEEK, freq="W-MON")

    rows, groups = [], []
    for t0 in mondays:
        t1 = t0 + WEEK
        cand = []
        for s, (px, fr) in data.items():
            if t0 not in px.index or t1 not in px.index:
                continue
            if fr[(fr.index > t0 - WEEK) & (fr.index <= t0)].size < 7 or px.index.min() > t0 - WEEK:
                continue
            cand.append(dict(symbol=s, sig=fsum(fr, t0 - WEEK, t0),
                             ret=px[t1] / px[t0] - 1, fund=fsum(fr, t0, t1)))
        if len(cand) < 2 * TOP:
            continue
        c = pd.DataFrame(cand).sort_values("sig")
        lo, hi = c.head(TOP), c.tail(TOP)
        price = 0.5 * lo.ret.mean() - 0.5 * hi.ret.mean()
        carry = 0.5 * hi.fund.mean() - 0.5 * lo.fund.mean()       # 숏은 펀딩비 받고, 롱은 냄
        rows.append(dict(week=t0, n=len(c), 가격차이=price, 펀딩수취=carry,
                         비용=-COST, 순수익=price + carry - COST))
        c["구간"] = pd.qcut(c.sig.rank(method="first"), 3, labels=["낮음", "중간", "높음"])
        c["week"] = t0
        groups.append(c)

    w = pd.DataFrame(rows).set_index("week")
    g = pd.concat(groups)
    print(f"\n==================== {label} — {len(w)}주, 주당 후보 평균 {w.n.mean():.1f}개 ====================")
    out = []
    for col in ["가격차이", "펀딩수취", "비용", "순수익"]:
        x = w[col]
        t = x.mean() / (x.std() / np.sqrt(len(x))) if x.std() > 0 else np.nan
        out.append(dict(항목=col, 주평균=f"{x.mean():+.3%}", 연환산=f"{x.mean() * 52:+.1%}",
                        주간승률=f"{(x > 0).mean():.0%}" if col != "비용" else "", t값=round(t, 2) if col != "비용" else ""))
    print("[롱 = 펀딩비 하위 3개, 숏 = 상위 3개]")
    print(pd.DataFrame(out).to_string(index=False))
    eq = w.순수익.cumsum()
    print(f"순수익 샤프 {w.순수익.mean() / w.순수익.std() * np.sqrt(52):.2f} | 최대낙폭 {(eq - eq.cummax()).min():+.1%} | "
          f"최악의 주 {w.순수익.min():+.2%}")

    print("\n[펀딩비 3분위별 다음 주 가격 수익률 (주마다 코인을 1/3씩 나눔) — 높음일수록 낮으면 가설 지지]")
    gg = g.groupby("구간", observed=True).agg(코인주=("ret", "size"), 지난주펀딩비=("sig", "mean"),
                                             다음주수익_평균=("ret", "mean"), 다음주수익_중앙=("ret", "median"),
                                             다음주펀딩비=("fund", "mean"))
    for col in ["지난주펀딩비", "다음주펀딩비"]:
        gg[col] = gg[col].map(lambda v: f"{v:+.3%}")
    for col in ["다음주수익_평균", "다음주수익_중앙"]:
        gg[col] = gg[col].map(lambda v: f"{v:+.2%}")
    print(gg.to_string())
    x = w.순수익
    verdict.append((x.mean(), x.mean() / (x.std() / np.sqrt(len(x)))))
    w.to_csv(root / "funding_xs_weeks.csv")

ok = len(verdict) == 2 and all(m > 0 and t >= 2 for m, t in verdict)
print("\n최종 판정:", "통과" if ok else
      "방향만 맞음 (약함)" if len(verdict) == 2 and all(m > 0 for m, _ in verdict) else "실패")
