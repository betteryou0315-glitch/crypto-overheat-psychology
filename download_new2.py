"""두 번째 새 검증용 종목 다운로드: 지금까지 쓴 60종목(data, data_holdout, data_final, data_new)을 뺀 다음 15개 코인
- 저장: data_new2/<종목>/ (15분봉, 펀딩비, metrics)
- 청산 흡수 가설 판정에 data_new 와 합쳐서 사용 (표본 늘리기)
사용법: python download_new2.py
"""
import sys
from pathlib import Path

USED = [Path("data"), Path("data_holdout"), Path("data_final"), Path("data_new")]
NEW = Path("data_new2")
sys.argv = [sys.argv[0], str(NEW)]
import download as dl

seen = set()
for p in USED:
    if (p / "symbols.txt").exists():
        seen |= set((p / "symbols.txt").read_text().split())
ranked = dl.top_symbols(150)
new = [s for s in ranked if s not in seen][:15]
print(f"이미 쓴 종목 {len(seen)}개 제외 → 두 번째 새 15종목:", ", ".join(new))
(NEW / "symbols.txt").write_text("\n".join(new))
for s in new:
    if dl.already_done(s):
        print(f"[{s}] 이미 있음, 건너뜀"); continue
    dl.download(s)
print("끝 — 다음: python download_agg1m.py data_new data_new2")
