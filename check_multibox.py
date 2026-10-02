from pathlib import Path

LBL = Path("data/train/labels")
counts = {}
for lf in LBL.glob("r2_*.txt"):
    n = len([l for l in lf.read_text().splitlines() if l.strip()])
    counts[n] = counts.get(n, 0) + 1

for n in sorted(counts):
    print(f"{n} boxes: {counts[n]} images")