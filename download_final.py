"""최종 검증용 종목 다운로드: 기존 30종목(data + data_holdout)을 뺀 다음 거래대금 상위 15개 코인
- 저장: data_final/<종목>/  (단 한 번의 최종 검증에만 사용)
- 15분봉, 펀딩비, metrics(OI) + 프리미엄 인덱스
사용법: python download_final.py
"""
import sys, subprocess, os
from pathlib import Path

MAIN, HOLD, FINAL = Path("data"), Path("data_holdout"), Path("data_final")
sys.argv = [sys.argv[0], str(FINAL)]
import download as dl

seen = set((MAIN / "symbols.txt").read_text().split()) | set((HOLD / "symbols.txt").read_text().split())
ranked = dl.top_symbols(90)
final = [s for s in ranked if s not in seen][:15]
print("최종 검증 15종목:", ", ".join(final))
(FINAL / "symbols.txt").write_text("\n".join(final))
for s in final:
    if dl.already_done(s):
        print(f"[{s}] 이미 있음, 건너뜀"); continue
    dl.download(s)

env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
subprocess.run([sys.executable, "download_premium.py", str(FINAL)], env=env)
