"""바이낸스 공개 데이터에 과거 청산 데이터(liquidationSnapshot)가 있는지 확인만 하는 스크립트
사용법: python check_liquidation.py
"""
import io, zipfile
import requests
import pandas as pd

SYM = "BTCUSDT"
DATES = ["2023-11-05", "2024-03-01", "2024-06-01", "2025-01-01", "2026-10-01"]
for kind in ["daily/liquidationSnapshot"]:
    print(f"--- {kind} ---")
    sample = None
    for d in DATES:
        url = f"https://data.binance.vision/data/futures/um/{kind}/{SYM}/{SYM}-liquidationSnapshot-{d}.zip"
        r = requests.get(url, timeout=30)
        print(f"{d}: {'있음' if r.status_code == 200 else '없음 (' + str(r.status_code) + ')'}")
        if r.status_code == 200 and sample is None:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            sample = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    if sample is not None:
        print("\n열 이름:", list(sample.columns))
        print(sample.head(10).to_string(index=False))
        print(f"하루 줄 수: {len(sample)}")