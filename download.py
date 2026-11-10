"""data.binance.vision 에서 USDT 무기한 선물 데이터 3년치 다운로드
- 대상: 현재 24시간 거래대금 상위 N개 종목 (스테이블코인 제외)
- 15분봉 klines (월별) / 펀딩비 fundingRate (월별) / metrics (일별, 5분 단위 OI)
- 저장: data/<종목>/klines_15m.csv, funding.csv, metrics.csv
"""
import io, sys, zipfile, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

TOP_N = 15
BASE = "https://data.binance.vision/data/futures/um"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT.mkdir(exist_ok=True)

END = dt.date(2026, 11, 1)          # 2026-10 까지
START = dt.date(2023, 11, 1)        # 3년

EXCLUDE = {"USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "DEFIUSDT"}
# 바이낸스가 코인으로 분류하지 않았을 때를 대비한 수동 제외 목록 (금/은/주식/ETF)
NON_CRYPTO = {"XAUUSDT", "XAGUSDT", "SOXLUSDT", "SNDKUSDT", "TSLAUSDT", "NVDAUSDT",
              "AAPLUSDT", "MSTRUSDT", "COINUSDT", "HOODUSDT", "QQQUSDT", "SPYUSDT"}

def top_symbols(n):
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()["symbols"]
    coins, skipped = set(), []
    for s in info:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        utype = s.get("underlyingType", "COIN")
        if utype == "COIN" and s["symbol"] not in NON_CRYPTO and s["symbol"] not in EXCLUDE:
            coins.add(s["symbol"])
        else:
            skipped.append(f"{s['symbol']}({utype})")

    r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=30).json()
    r = [x for x in r if x["symbol"] in coins]
    r.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    print("코인 아님으로 제외:", ", ".join(skipped[:30]), "..." if len(skipped) > 30 else "")
    return [x["symbol"] for x in r[:n]]

def already_done(sym):
    d = OUT / sym
    return all((d / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"])

def months():
    d = START
    while d < END:
        yield d.strftime("%Y-%m")
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

def days():
    d = START
    while d < END:
        yield d.strftime("%Y-%m-%d")
        d += dt.timedelta(days=1)

def fetch_csv(url):
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return z.read(z.namelist()[0]).decode()

def grab(urls):
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(fetch_csv, urls))
    return [r for r in res if r is not None], len(urls) - sum(r is not None for r in res)

KCOLS = ["open_time","open","high","low","close","volume","close_time",
         "quote_volume","count","taker_buy_volume","taker_buy_quote_volume","ignore"]

def parse(raw, cols=None):
    has_header = not raw[0].isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None,
                       names=None if has_header else cols)

def download(sym):
    d = OUT / sym
    d.mkdir(exist_ok=True)

    raws, miss = grab([f"{BASE}/monthly/klines/{sym}/15m/{sym}-15m-{m}.zip" for m in months()])
    if not raws:
        print(f"[{sym}] 데이터 없음, 건너뜀"); return
    k = pd.concat([parse(r, KCOLS) for r in raws]); k.columns = KCOLS
    k.drop_duplicates("open_time").sort_values("open_time").to_csv(d/"klines_15m.csv", index=False)

    raws, _ = grab([f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip" for m in months()])
    pd.concat([parse(r) for r in raws]).sort_values("calc_time").to_csv(d/"funding.csv", index=False)

    raws, mmiss = grab([f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{x}.zip" for x in days()])
    pd.concat([parse(r) for r in raws]).sort_values("create_time").to_csv(d/"metrics.csv", index=False)

    print(f"[{sym}] 15분봉 {len(k)}개 (빠진 달 {miss}), metrics 빠진 날 {mmiss}")

if __name__ == "__main__":
    syms = top_symbols(TOP_N)
    print("대상 종목:", ", ".join(syms))
    (OUT / "symbols.txt").write_text("\n".join(syms))   # 백테스트는 이 목록만 사용
    for s in syms:
        if already_done(s):
            print(f"[{s}] 이미 있음, 건너뜀"); continue
        download(s)