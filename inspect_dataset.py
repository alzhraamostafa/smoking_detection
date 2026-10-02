from pathlib import Path

ROOT = Path("data")

for split in ["train", "valid", "test"]:
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    if not img_dir.exists():
        print(f"{split}: folder not found")
        continue

    images = [p for p in img_dir.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    labels = list(lbl_dir.glob("*.txt"))

    n_boxes, empty, areas = 0, 0, []
    for lf in labels:
        rows = [l.split() for l in lf.read_text().splitlines() if l.strip()]
        if not rows:
            empty += 1
        for parts in rows:
            n_boxes += 1
            areas.append(float(parts[3]) * float(parts[4]))  # width * height

    print(f"{split}: {len(images)} images, {len(labels)} label files, "
          f"{n_boxes} boxes, {empty} images with no box")
    if areas:
        areas.sort()
        print(f"   median box covers {areas[len(areas)//2] * 100:.2f}% of the image")