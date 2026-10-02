from pathlib import Path

LBL = Path("data/train/labels")

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union else 0

cleaned = 0
for lf in LBL.glob("r2_*.txt"):
    lines = [l for l in lf.read_text().splitlines() if l.strip()]
    rows = [list(map(float, l.split()[1:5])) for l in lines]
    if len(rows) < 2:
        continue

    keep = [True] * len(rows)
    for i in range(len(rows)):
        if not keep[i]:
            continue
        for j in range(i+1, len(rows)):
            if keep[j] and iou(rows[i], rows[j]) > 0.7:
                keep[j] = False  # drop the duplicate, keep the first

    if not all(keep):
        new_lines = [l for l, k in zip(lines, keep) if k]
        lf.write_text("\n".join(new_lines) + "\n")
        cleaned += 1

print(f"Cleaned {cleaned} label files (removed duplicate overlapping boxes)")