"""청산 흡수 가설 최종 판정 (2026-11-09 17:1x 등록, data_new · data_new2 를 보기 전에 확정)
근거 (탐색, 결과를 보고 고른 부분 있음):
  - absorption.py: '흡수' 그룹의 4시간 중앙값이 기존 +0.22% / 검증 +0.57% (최종 15종목은 13건으로 −0.94%)
  - 4시간이라는 기간은 탐색 결과를 보고 고른 것 → 통과해도 조심해서 해석
가설: 롱 과열 후 청산 투매를 '흡수'한 경우 (absorption.py 의 정의 그대로), 투매 후 15분 시점 가격 기준 4시간 수익률이 플러스다
규칙: absorption.py 와 완전히 동일 (투매 -1.5%/5분 + 매도 3배 + OI -1%, 직후 15분 저점 유지 + 매도 >= 매수)
비용: 왕복 0.14% 차감한 4시간 수익률로 판정
판정: data_new + data_new2 (46~75위 30종목) 합산
      흡수 그룹 4시간 순수익 평균 > 0 + 월 단위 군집 t >= 2.3 → 통과   (새 데이터에 걸린 두 번째 가설 → 기준 상향)
      두 폴더가 모두 다 받아졌을 때만 판정
사용법: python absorption_final.py   (python download_new2.py → python download_agg1m.py data_new data_new2 후)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import absorption as ab

COST = 0.0014
FOLDERS = [("data_new", "새 15종목 (46~60위)"), ("data_new2", "새 15종목 (61~75위)")]

def ready(folder):
    root = Path(folder)
    if not (root / "symbols.txt").exists():
        return False, "폴더 없음"
    syms = (root / "symbols.txt").read_text().split()
    base = [s for s in syms if all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"])]
    agg = [s for s in base if (root / s / "agg1m").exists() and any((root / s / "agg1m").iterdir())]
    if len(base) < len(syms):
        return False, f"15분봉·펀딩비·OI {len(base)}/{len(syms)}개"
    if len(agg) < len(base) * 0.8:
        return False, f"1분 체결 데이터 {len(agg)}/{len(base)}개 — download_agg1m.py 실행 필요"
    return True, f"{len(syms)}개 준비됨 (1분 데이터 {len(agg)}개)"

pd.set_option("display.width", 230)
parts, ok_all = [], True
for folder, label in FOLDERS:
    ok, msg = ready(folder)
    print(f"[{label}] {msg}")
    if not ok:
        ok_all = False; continue
    d, (nev, nflush) = ab.analyze(folder)
    print(f"  과열 사건 {nev} 중 청산 투매 {nflush}")
    if len(d):
        print(ab.table(d).to_string(index=False))
        parts.append(d.assign(data=label))

if not ok_all:
    print("\n최종 판정: 데이터가 아직 다 준비 안 됨 — 판정 보류")
else:
    d = pd.concat(parts, ignore_index=True)
    g = d[d.kind == "흡수"].dropna(subset=["r4"]).copy()
    g["net4"] = g.r4 - COST
    print(f"\n==================== 판정: 두 폴더 합산 흡수 {len(g)}건 ====================")
    if len(g) < 10:
        print("흡수 사건이 너무 적어 판정 불가")
    else:
        m = g.net4.mean()
        gg = pd.Series(g.net4.values - m).groupby(pd.to_datetime(g.time).dt.to_period("M").values).sum().values
        t = m / (np.sqrt((gg**2).sum()) / len(g))
        print(f"4시간 순수익 평균 {m:+.2%} | 중앙값 {g.net4.median():+.2%} | 승률 {(g.net4 > 0).mean():.0%} | "
              f"먼저 +3% {(g['first'] == '위').mean():.0%} | 군집t {t:.2f}")
        print("최종 판정 (기준 t >= 2.3):", "통과" if m > 0 and t >= 2.3 else ("방향만 맞음 (약함)" if m > 0 else "실패"))
