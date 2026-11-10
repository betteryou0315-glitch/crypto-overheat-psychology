"""2023-11 이전(한 번도 안 본 기간) 데이터가 어디까지 있는지 확인 — 파일은 받지 않고 있는지만 물어봄
대상: data, data_holdout, data_final, data_new, data_new2 의 75종목
확인:
  - 15분봉 (월별): 가장 이른 달
  - OI 들어 있는 metrics (일별): 2021-06 ~ 2023-09 사이 체크포인트
  - 체결 데이터 aggTrades (일별): 2022-06-01, 2023-06-01
결과: 종목별 표 + "2022-01 ~ 2023-10 기간에 15분봉·OI 가 다 있는 종목 수"
사용법: python check_history.py
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
FOLDERS = ["data", "data_holdout", "data_final", "data_new", "data_new2"]
MONTHS = pd.date_range("2019-09-01", "2023-10-01", freq="MS").strftime("%Y-%m").tolist()
METRIC_DAYS = ["2021-06-01", "2021-12-15", "2022-01-15", "2022-06-15", "2023-01-15", "2023-09-15"]
AGG_DAYS = ["2022-06-01", "2023-06-01"]

def exists(url):
    try:
        return requests.head(url, timeout=30, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False

def earliest_kline(s):
    lo, hi = 0, len(MONTHS) - 1
    if not exists(f"{BASE}/monthly/klines/{s}/15m/{s}-15m-{MONTHS[hi]}.zip"):
        return None
    while lo < hi:                                   # 이진 탐색: 상장 이후 계속 있다고 가정
        mid = (lo + hi) // 2
        if exists(f"{BASE}/monthly/klines/{s}/15m/{s}-15m-{MONTHS[mid]}.zip"):
            hi = mid
        else:
            lo = mid + 1
    return MONTHS[lo]

def check(s):
    row = dict(종목=s, 첫15분봉=earliest_kline(s) or "없음")
    for d in METRIC_DAYS:
        row[f"OI {d[:7]}"] = "O" if exists(f"{BASE}/daily/metrics/{s}/{s}-metrics-{d}.zip") else "-"
    for d in AGG_DAYS:
        row[f"체결 {d[:7]}"] = "O" if exists(f"{BASE}/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip") else "-"
    return row

syms, src = [], {}
for f in FOLDERS:
    p = Path(f) / "symbols.txt"
    if p.exists():
        for s in p.read_text().split():
            if s not in src:
                syms.append(s); src[s] = f
print(f"대상 {len(syms)}종목 확인 중...", flush=True)
with ThreadPoolExecutor(12) as ex:
    rows = list(ex.map(check, syms))
df = pd.DataFrame(rows)
df.insert(1, "세트", df.종목.map(src))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.to_string(index=False))

full = df[(df.첫15분봉 != "없음") & (df.첫15분봉 <= "2022-01") & (df["OI 2022-01"] == "O") & (df["OI 2023-09"] == "O")]
print(f"\n2022-01 ~ 2023-10 에 15분봉 + OI 가 있는 종목: {len(full)}개")
print("  세트별:", full.세트.value_counts().to_dict())
print("  종목:", ", ".join(full.종목))
mid = df[(df.첫15분봉 != "없음") & (df.첫15분봉 <= "2023-01") & (df["OI 2023-01"] == "O")]
print(f"2023-01 ~ 2023-10 (10개월) 만이라도 있는 종목: {len(mid)}개")
print("OI(metrics) 가장 이른 체크포인트:", {d: int((df[f'OI {d[:7]}'] == 'O').sum()) for d in METRIC_DAYS})
