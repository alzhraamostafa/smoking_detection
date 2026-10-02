import shutil
from pathlib import Path

SRC = Path("data_reality2")
DST = Path("data")
EXT = {".jpg", ".jpeg", ".png"}

for split in ["train", "valid"]:
    src_img = SRC / split / "images"
    src_lbl = SRC / split / "labels"
    dst_img = DST / split / "images"
    dst_lbl = DST / split / "labels"

    if not src_img.exists():
        print(f"{split}: no source folder, skipping")
        continue

    count = 0
    for img_path in src_img.iterdir():
        if img_path.suffix.lower() not in EXT:
            continue
        label_path = src_lbl / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  WARNING: no label for {img_path.name}, skipping")
            continue

        new_name = f"r2_{img_path.stem}"
        shutil.copy(img_path, dst_img / f"{new_name}{img_path.suffix.lower()}")
        shutil.copy(label_path, dst_lbl / f"{new_name}.txt")
        count += 1

    print(f"{split}: merged {count} images")