from pathlib import Path
from ultralytics import YOLO

model = YOLO("models/best_v3.pt")
print("Classes:", model.names)

test_dir = Path("data/test/images")
out_dir = Path("test_predictions")
out_dir.mkdir(exist_ok=True)

# Run on 20 random test images and save annotated results
import random
imgs = list(test_dir.glob("*.jpg"))
random.seed(1)
sample = random.sample(imgs, 20)

results = model.predict(sample, conf=0.4, save=True, project="test_predictions", name="run1", exist_ok=True)

for r in results:
    n_boxes = len(r.boxes)
    print(f"{r.path.split(chr(92))[-1]}: {n_boxes} detection(s)")