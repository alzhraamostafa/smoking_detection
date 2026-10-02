from pathlib import Path
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union else 0

model = YOLO("models/best_v3.pt")
IMG_DIR = Path("data/test/images")
LBL_DIR = Path("data/test/labels")
images = sorted(IMG_DIR.glob("*.jpg"))

# Run inference ONCE at very low confidence, so we have every candidate box
# and its own confidence score, then sweep thresholds against this fixed set
raw_results = model.predict(images, conf=0.05, verbose=False)

# Load ground truth boxes per image (in the same xc,yc,w,h format the labels use)
gt = {}
for img_path in images:
    lbl = LBL_DIR / (img_path.stem + ".txt")
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if line.strip():
                parts = line.split()
                boxes.append(list(map(float, parts[1:5])))
    gt[img_path.stem] = boxes

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
results = []

for conf in thresholds:
    tp, fp, fn = 0, 0, 0
    for img_path, r in zip(images, raw_results):
        w, h = r.orig_shape[1], r.orig_shape[0]
        gt_boxes = gt[img_path.stem][:]  # copy, we'll remove matched ones
        matched = [False] * len(gt_boxes)

        preds = [(float(b.conf[0]), b.xywhn[0].tolist()) for b in r.boxes if float(b.conf[0]) >= conf]

        for pconf, pbox in preds:
            best_iou, best_j = 0, -1
            for j, gbox in enumerate(gt_boxes):
                if matched[j]:
                    continue
                i = iou(pbox, gbox)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= 0.5:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
        fn += matched.count(False)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    results.append((conf, precision, recall, f1))
    print(f"conf={conf:.2f}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")

confs, precs, recs, f1s = zip(*results)
plt.figure(figsize=(7,5))
plt.plot(confs, precs, marker='o', label='Precision')
plt.plot(confs, recs, marker='o', label='Recall')
plt.plot(confs, f1s, marker='o', label='F1')
plt.xlabel('Confidence threshold')
plt.ylabel('Score')
plt.title('Precision / Recall / F1 vs Confidence Threshold (manual)')
plt.legend()
plt.grid(True)
plt.savefig('threshold_sweep.png', dpi=150)
print("Saved chart to threshold_sweep.png")