"""새 검증용 종목 다운로드: 지금까지 쓴 45종목(data, data_holdout, data_final)을 뺀 다음 거래대금 상위 15개 코인
- 저장: data_new/<종목>/ (15분봉, 펀딩비, metrics) — 두 번째 돌파 가설의 단 한 번 판정에만 사용
사용법: python download_new.py
"""
import sys
from pathlib import Path

USED = [Path("data"), Path("data_holdout"), Path("data_final")]
NEW = Path("data_new")
sys.argv = [sys.argv[0], str(NEW)]
import download as dl

seen = set()
for p in USED:
    if (p / "symbols.txt").exists():
        seen |= set((p / "symbols.txt").read_text().split())
ranked = dl.top_symbols(120)
new = [s for s in ranked if s not in seen][:15]
print(f"이미 쓴 종목 {len(seen)}개 제외 → 새 검증 15종목:", ", ".join(new))
(NEW / "symbols.txt").write_text("\n".join(new))
for s in new:
    if dl.already_done(s):
        print(f"[{s}] 이미 있음, 건너뜀"); continue
    dl.download(s)
print("끝")
