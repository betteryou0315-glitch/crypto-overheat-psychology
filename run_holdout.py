"""홀드아웃 검증 전체를 순서대로 실행 (명령 하나로)
python run_holdout.py
"""
import os, subprocess, sys

ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}   # 윈도우 cp949 인코딩 오류 방지

STEPS = [
    ["download_holdout.py"],
    ["download_premium.py", "data_holdout"],
    ["direction.py", "data_holdout"],
    ["basis.py", "data_holdout"],
    ["combine.py", "data_holdout"],
    ["regime.py", "data_holdout"],
    ["holdout_test.py"],
]
for step in STEPS:
    print(f"\n########## {' '.join(step)} ##########", flush=True)
    quiet = step[0] in ("direction.py", "basis.py", "combine.py", "regime.py")
    r = subprocess.run([sys.executable, "-W", "ignore", *step],
                       stdout=subprocess.DEVNULL if quiet else None, env=ENV)
    if r.returncode != 0:
        sys.exit(f"{step[0]} 에서 멈춤 (위 오류 메시지 확인)")
    if quiet:
        print("완료")