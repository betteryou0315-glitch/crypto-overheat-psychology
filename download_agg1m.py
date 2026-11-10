"""체결 데이터(aggTrades)를 과열 사건이 있는 날만 받아서 1분 요약으로 저장
- 대상: data, data_holdout, data_final 의 과열 사건 (펀딩비>=0.02%, RSI>=70, OI 1시간 증가, 연속 신호는 첫 봉만)
- 필요한 날: 신호가 뜬 날 + 다음 날 (신호 후 24시간을 덮기 위해)
- 저장: <폴더>/<종목>/agg1m/<날짜>.csv.gz
    열: minute, open, high, low, close, buy (시장가 매수 금액), sell (시장가 매도 금액), n (체결 수)
- 원본 zip 은 저장하지 않음 (용량 절약). 이미 요약한 날은 건너뜀 → 중간에 멈춰도 다시 실행하면 이어받기
사용법: python download_agg1m.py              (세 폴더 전부)
        python download_agg1m.py data_final   (특정 폴더만)
"""
import io, sys, time, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import backtest as bt

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
FOLDERS = sys.argv[1:] or ["data", "data_holdout", "data_final"]
COOLDOWN = 16
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]

def episodes(k):
    rsi = bt.rsi(k["close"], bt.P["rsi_len"])
    s = ((k["funding"] >= bt.P["funding_min"]) & (rsi >= bt.P["rsi_min"])
         & (k["oi"] > k["oi"].shift(bt.P["oi_lag"]))).fillna(False)
    prev = s.shift(1, fill_value=False).rolling(COOLDOWN, min_periods=1).max().astype(bool)
    return k["t"][(s & ~prev).values]

def needed_days():
    jobs = []
    for folder in FOLDERS:
        root = Path(folder)
        if not (root / "symbols.txt").exists():
            print(f"{folder} 없음 — 건너뜀"); continue
        for s in (root / "symbols.txt").read_text().split():
            if not all((root / s / f).exists() for f in ["klines_15m.csv", "funding.csv", "metrics.csv"]):
                continue
            bt.D = root / s
            k = bt.load()
            days = set()
            for t in episodes(k):
                days.add(t.normalize()); days.add((t + pd.Timedelta("24h")).normalize())
            for d in sorted(days):
                jobs.append((root, s, d.strftime("%Y-%m-%d")))
    return jobs

def summarize(raw):
    first = raw[:100].decode(errors="ignore")
    has_header = not first[0].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=None if has_header else COLS,
                     usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
                     dtype={"price": "float64", "quantity": "float64", "transact_time": "int64"})
    maker = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"]).values
    notional = (df["price"] * df["quantity"]).values
    minute = (df["transact_time"].values // 60000) * 60000
    g = pd.DataFrame({"minute": minute, "price": df["price"].values,
                      "buy": np.where(maker, 0.0, notional), "sell": np.where(maker, notional, 0.0)})
    out = g.groupby("minute").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                  close=("price", "last"), buy=("buy", "sum"), sell=("sell", "sum"),
                                  n=("price", "size")).reset_index()
    return out

def work(job):
    root, s, d = job
    out = root / s / "agg1m" / f"{d}.csv.gz"
    if out.exists():
        return "cached"
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{s}/{s}-aggTrades-{d}.zip", timeout=180)
            break
        except requests.RequestException:
            time.sleep(3)
    else:
        return "error"
    if r.status_code != 200:
        return "missing"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    m = summarize(z.read(z.namelist()[0]))
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False, compression="gzip")
    return "ok"

if __name__ == "__main__":
    jobs = needed_days()
    todo = [j for j in jobs if not (j[0] / j[1] / "agg1m" / f"{j[2]}.csv.gz").exists()]
    print(f"필요한 종목·날짜 {len(jobs)}개 | 이미 있음 {len(jobs) - len(todo)}개 | 새로 받을 것 {len(todo)}개")
    stat, t0 = {}, time.time()
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result() if not f.exception() else "error"
            stat[res] = stat.get(res, 0) + 1
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} 완료 ({el / 60:.1f}분 경과, 남은 예상 {el / i * (len(todo) - i) / 60:.0f}분) {stat}", flush=True)
    print("끝:", stat)
