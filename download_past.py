"""과거 기간(2022-01 ~ 2023-10) 데이터 다운로드 — 한 번도 안 본 기간
- 대상: check_history.py 에서 이 기간에 15분봉 + OI 가 다 있던 28종목
- 저장: data_past/<종목>/ (15분봉, 펀딩비, metrics) — 기존 폴더와 섞이지 않게 분리
다음 단계: python download_agg1m.py data_past  →  python past_verdict.py
사용법: python download_past.py
"""
import sys, datetime as dt
from pathlib import Path

PAST = Path("data_past")
sys.argv = [sys.argv[0], str(PAST)]
import download as dl

dl.START = dt.date(2022, 1, 1)      # download.py 의 기간 설정을 이 스크립트에서만 바꿈
dl.END = dt.date(2023, 11, 1)       # 2023-10 까지 (기존 데이터는 2023-11 부터)

SYMS = ["BTCUSDT", "ETHUSDT", "ZECUSDT", "ONEUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "DOGEUSDT",
        "BNBUSDT", "UNIUSDT", "BCHUSDT", "LINKUSDT", "LITUSDT", "HOTUSDT", "ADAUSDT", "AAVEUSDT",
        "FILUSDT", "XMRUSDT", "AVAXUSDT", "LTCUSDT", "RSRUSDT", "XLMUSDT", "DOTUSDT", "DASHUSDT",
        "GTCUSDT", "TRXUSDT", "1000SHIBUSDT", "HBARUSDT"]

PAST.mkdir(exist_ok=True)
(PAST / "symbols.txt").write_text("\n".join(SYMS))
print(f"과거 기간 {dl.START} ~ {dl.END} (미포함), {len(SYMS)}종목")
for s in SYMS:
    if dl.already_done(s):
        print(f"[{s}] 이미 있음, 건너뜀"); continue
    dl.download(s)
print("끝 — 다음: python download_agg1m.py data_past")
