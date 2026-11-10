"""오더북 과거 데이터(bookDepth)가 있는지, 어떤 모양인지 확인만 하는 스크립트
- 날짜 몇 개를 찍어보고 받아지는지 확인 (언제부터 데이터가 있는지)
- 받아지면 열 이름, 처음 몇 줄, 기록 간격, 가격 구간(percentage) 종류를 출력
사용법: python check_bookdepth.py
"""
import io, zipfile
import requests
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
SYM = "BTCUSDT"
DATES = ["2023-11-05", "2024-03-01", "2024-06-01", "2025-01-01", "2025-06-01", "2026-10-01"]

sample = None
for d in DATES:
    url = f"{BASE}/{SYM}/{SYM}-bookDepth-{d}.zip"
    r = requests.get(url, timeout=30)
    print(f"{d}: {'있음' if r.status_code == 200 else '없음 (' + str(r.status_code) + ')'}")
    if r.status_code == 200 and sample is None:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        sample = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))

if sample is None:
    print("\nbookDepth 데이터를 찾지 못했어요")
else:
    print("\n열 이름:", list(sample.columns))
    print("\n처음 15줄:")
    print(sample.head(15).to_string(index=False))
    if "percentage" in sample.columns:
        print("\n가격 구간(percentage) 종류:", sorted(sample["percentage"].unique()))
    tcol = sample.columns[0]
    ts = pd.to_datetime(sample[tcol]).drop_duplicates().sort_values()
    print(f"\n하루 기록 시점 수: {len(ts)}  | 간격 중앙값: {ts.diff().median()}")
    print(f"하루 파일 줄 수: {len(sample)}")