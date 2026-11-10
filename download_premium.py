"""프리미엄 인덱스(선물 vs 현물 지수 괴리율) 15분봉 다운로드
- data/symbols.txt 의 종목 대상 (download.py 를 먼저 실행해둔 상태)
- 저장: data/<종목>/premium_15m.csv
"""
import io, sys, zipfile, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
START, END = dt.date(2023, 11, 1), dt.date(2026, 11, 1)
COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","count","taker_buy_volume","taker_buy_quote_volume","ignore"]

def months():
    d = START
    while d < END:
        yield d.strftime("%Y-%m")
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

def fetch(url):
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    has_header = not raw[0].isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None,
                       names=None if has_header else COLS)

syms = (OUT / "symbols.txt").read_text().split()
for s in syms:
    path = OUT / s / "premium_15m.csv"
    if path.exists():
        print(f"[{s}] 이미 있음, 건너뜀"); continue
    urls = [f"{BASE}/{s}/15m/{s}-15m-{m}.zip" for m in months()]
    with ThreadPoolExecutor(12) as ex:
        res = [r for r in ex.map(fetch, urls) if r is not None]
    if not res:
        print(f"[{s}] 프리미엄 데이터 없음"); continue
    df = pd.concat(res); df.columns = COLS[:len(df.columns)]
    df.drop_duplicates("open_time").sort_values("open_time").to_csv(path, index=False)
    print(f"[{s}] {len(df)}봉 (빠진 달 {len(urls) - len(res)})")