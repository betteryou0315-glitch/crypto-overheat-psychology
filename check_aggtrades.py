"""체결 데이터(aggTrades) 확인만 하는 스크립트
1) 파일 크기 확인 (다운로드 없이 크기만): 종목 몇 개 × 날짜 몇 개
2) 알트코인 하루치 하나를 받아서 열 이름, 처음 몇 줄, 하루 체결 수, 시장가 매수/매도 비율 출력
3) 과열 사건이 있는 날 수로 전체 다운로드 용량 추정 (data/direction_events.csv 사용)
사용법: python check_aggtrades.py
"""
import io, zipfile
from pathlib import Path
import requests
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
SYMS = ["BTCUSDT", "SOLUSDT", "NEARUSDT", "1000PEPEUSDT"]
DATES = ["2024-03-01", "2025-06-01"]

def url(s, d):
    return f"{BASE}/{s}/{s}-aggTrades-{d}.zip"

print("=== 1) 파일 크기 (압축 상태) ===")
sizes = []
for s in SYMS:
    for d in DATES:
        r = requests.head(url(s, d), timeout=30, allow_redirects=True)
        mb = int(r.headers.get("Content-Length", 0)) / 1e6 if r.status_code == 200 else None
        sizes.append(dict(종목=s, 날짜=d, 상태=r.status_code, 크기MB=round(mb, 1) if mb else "-"))
print(pd.DataFrame(sizes).to_string(index=False))

print("\n=== 2) NEARUSDT 2025-06-01 하루치 내용 ===")
r = requests.get(url("NEARUSDT", "2025-06-01"), timeout=120)
if r.status_code != 200:
    print("받기 실패:", r.status_code)
else:
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    has_header = not first[0].isdigit()
    cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=None if has_header else cols)
    print("헤더 있음" if has_header else "헤더 없음 (열 이름 직접 지정)")
    print("열 이름:", list(df.columns))
    print(df.head(8).to_string(index=False))
    t = pd.to_datetime(df.iloc[:, 5], unit="ms")
    print(f"\n하루 체결 수: {len(df):,} | 시간 범위 {t.min()} ~ {t.max()} | 압축 해제 {len(raw) / 1e6:.0f}MB")
    maker = df.iloc[:, 6].astype(str).str.lower().isin(["true", "1"])
    notional = df.iloc[:, 1] * df.iloc[:, 2]
    print(f"시장가 매수 비중 (금액): {notional[~maker].sum() / notional.sum():.1%}  "
          f"※ is_buyer_maker=True 면 '매도자가 시장가로 친 것' = 시장가 매도")

print("\n=== 3) 필요한 다운로드 용량 추정 ===")
for folder in ["data", "data_holdout", "data_final"]:
    p = Path(folder) / "direction_events.csv"
    if p.exists():
        ev = pd.read_csv(p, parse_dates=["time"])
        days = ev.assign(d=ev.time.dt.date).drop_duplicates(["symbol", "d"])
        print(f"{folder}: 과열 사건 {len(ev)}건 → 종목·날짜 {len(days)}개 (전날 포함하면 최대 약 {len(days) * 2}개)")
    else:
        print(f"{folder}: direction_events.csv 없음")
print("※ 1)의 크기 × 종목·날짜 수로 대략 용량 계산. BTC 같은 대형 코인은 하루치가 매우 큼")
