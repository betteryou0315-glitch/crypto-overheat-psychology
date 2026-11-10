"""검증용(홀드아웃) 종목 다운로드: 기존 15종목을 뺀 다음 거래대금 상위 15개 코인
- 저장: data_holdout/<종목>/  (기존 data 폴더와 섞이지 않게 분리)
- BTC 레짐 계산용으로 data/BTCUSDT 를 복사 (분석 대상에는 포함 안 함)
"""
import sys, shutil
from pathlib import Path

MAIN, HOLD = Path("data"), Path("data_holdout")
sys.argv = [sys.argv[0], str(HOLD)]          # download.py 의 저장 폴더를 data_holdout 으로
import download as dl

main_syms = set((MAIN / "symbols.txt").read_text().split())
ranked = dl.top_symbols(60)
hold = [s for s in ranked if s not in main_syms][:15]
print("기존 15종목:", ", ".join(sorted(main_syms)))
print("검증용 15종목:", ", ".join(hold))
(HOLD / "symbols.txt").write_text("\n".join(hold))

for s in hold:
    if dl.already_done(s):
        print(f"[{s}] 이미 있음, 건너뜀"); continue
    dl.download(s)

if not (HOLD / "BTCUSDT").exists():
    shutil.copytree(MAIN / "BTCUSDT", HOLD / "BTCUSDT")
    print("BTCUSDT 복사 완료 (레짐 계산용)")