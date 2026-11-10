"""과열 순간에 옵션(내재 변동성)이 실제 이후 변동성보다 쌌는가? — BTC, ETH
데이터: Deribit DVOL (30일 내재 변동성 지수, 연율 %) — 공개 API, 1시간 간격, 처음 실행 시 다운로드
비교:
  - 내재 변동성(IV) = 신호 시점 DVOL
  - 실제 변동성(RV) = 이후 24시간 15분봉 수익률로 계산한 변동성 (연율 %)
  - 스트래들 손익 근사 (하루짜리 ATM 스트래들을 사서 만기까지 보유):
        이익 = |24시간 수익률| - 옵션값,   옵션값 ≈ 0.798 × IV × √(1/365)
    (수수료·호가 차이, 만기별 IV 차이(30일 지수 vs 하루짜리)는 반영 안 함 → 실제보다 낙관적)
기준선: 모든 시점 평균 (보통 옵션이 비싸서 사면 손해 = 변동성 위험 프리미엄)
판정: 과열 사건의 스트래들 손익이 플러스 + 기준선보다 확실히 높음 → 옵션 전략 가능성 있음
사용법: python options_check.py   (data/direction_events.csv, data/BTCUSDT, data/ETHUSDT 필요)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path("data")
COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
START, END = pd.Timestamp("2023-11-01"), pd.Timestamp("2026-11-01")
API = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
STEP = pd.Timedelta(hours=900)

def get_dvol(cur):
    path = ROOT / f"dvol_{cur}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["t"])
    rows, s = [], START
    while s < END:
        e = min(s + STEP, END)
        r = requests.get(API, params=dict(currency=cur, resolution="3600",
                                          start_timestamp=int(s.timestamp() * 1000),
                                          end_timestamp=int(e.timestamp() * 1000)), timeout=30).json()
        rows += r.get("result", {}).get("data", [])
        s = e
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    df["t"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("t")[["t", "close"]].rename(columns={"close": "dvol"})
    df.to_csv(path, index=False)
    print(f"[{cur}] DVOL {len(df)}개 다운로드 ({df.t.min():%Y-%m-%d} ~ {df.t.max():%Y-%m-%d})")
    return df

def cluster_t(x, times):
    m = x.mean()
    g = pd.Series(x - m).groupby(pd.Series(times).dt.to_period("M").values).sum().values
    return m, m / (np.sqrt((g**2).sum()) / len(x))

ev_all = pd.read_csv(ROOT / "direction_events.csv", parse_dates=["time"])
pd.set_option("display.width", 200)
rows = []
for cur, sym in COINS.items():
    dv = get_dvol(cur)
    k = pd.read_csv(ROOT / sym / "klines_15m.csv")
    k["t"] = pd.to_datetime(k["close_time"] + 1, unit="ms").astype("datetime64[ns]")
    k = k[["t", "close"]].sort_values("t").reset_index(drop=True)
    r = np.log(k["close"]).diff()
    k["rv24"] = r[::-1].rolling(96).std()[::-1].shift(-1) * np.sqrt(96 * 365) * 100   # 이후 24시간
    k["ret24"] = k["close"].shift(-96) / k["close"] - 1
    k = pd.merge_asof(k, dv.astype({"t": "datetime64[ns]"}), on="t", direction="backward",
                      tolerance=pd.Timedelta("2h"))
    k = k.dropna(subset=["rv24", "ret24", "dvol"])
    k["premium"] = 0.798 * (k["dvol"] / 100) * np.sqrt(1 / 365)
    k["pnl"] = k["ret24"].abs() - k["premium"]          # 기초자산 가격 대비 %
    k["rv_iv"] = k["rv24"] / k["dvol"]

    ev = ev_all[ev_all.symbol == sym][["time"]].merge(k, left_on="time", right_on="t")
    base = k.iloc[::16]                                  # 4시간마다 표본 (겹침 줄이기)
    for name, d in [("평소 (4시간마다 표본)", base), ("과열 사건", ev)]:
        m, t = cluster_t(d["pnl"].values, d["t"])
        rows.append(dict(코인=cur, 구분=name, 건수=len(d), IV평균=f"{d.dvol.mean():.0f}%",
                         이후RV평균=f"{d.rv24.mean():.0f}%", RV_IV비율=f"{d.rv_iv.mean():.2f}",
                         옵션값=f"{d.premium.mean():.2%}", 실제변동폭=f"{d.ret24.abs().mean():.2%}",
                         스트래들손익=f"{m:+.2%}", 승률=f"{(d.pnl > 0).mean():.0%}", 군집t=round(t, 2)))
    if len(ev):
        # 과열 사건 vs 평소의 차이
        m_e, m_b = ev.pnl.mean(), base.pnl.mean()
        rows.append(dict(코인=cur, 구분="→ 과열 - 평소 차이", 건수="", 스트래들손익=f"{m_e - m_b:+.2%}"))

print("=== 과열 순간 옵션 가격 vs 실제 이후 변동성 ===")
print("※ RV/IV 비율 > 1 이면 옵션이 쌌던 것 (실제 변동성이 예상보다 큼)")
print("※ 스트래들 손익은 기초자산 가격 대비 %, 수수료 미반영\n")
print(pd.DataFrame(rows).fillna("").to_string(index=False))
