import random
from pathlib import Path
import cv2

IMG_DIR = Path("data/train/images")
LBL_DIR = Path("data/train/labels")
OUT_DIR = Path("preview")
OUT_DIR.mkdir(exist_ok=True)

images = list(IMG_DIR.glob("*.jpg"))
for img_path in random.sample(images, 12):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label_path = LBL_DIR / (img_path.stem + ".txt")
    for line in label_path.read_text().splitlines():
        cls, xc, yc, bw, bh = map(float, line.split()[:5])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(OUT_DIR / img_path.name), img)

print("Saved 12 previews in the 'preview' folder")