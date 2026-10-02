import random
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 4:
    sys.exit("Usage: python add_negatives.py <folder> <count> <name_prefix>")

SRC = Path(sys.argv[1])            # kaggle folder
N = int(sys.argv[2])               # how many images to add
PREFIX = sys.argv[3].lower()       # file names must START with this
DATA = Path("data")
EXT = {".jpg", ".jpeg", ".png"}

imgs = [p for p in SRC.rglob("*")
        if p.suffix.lower() in EXT and p.name.lower().startswith(PREFIX)]
random.seed(42)
random.shuffle(imgs)
imgs = imgs[:N]

print(f"Using {len(imgs)} images. Sample names:")
for p in imgs[:5]:
    print("  ", p.name)

# 70% train, 20% valid, 10% test, same as the original split
cuts = {
    "train": (0, int(len(imgs) * 0.7)),
    "valid": (int(len(imgs) * 0.7), int(len(imgs) * 0.9)),
    "test":  (int(len(imgs) * 0.9), len(imgs)),
}

for split, (a, b) in cuts.items():
    for p in imgs[a:b]:
        name = f"neg_{p.stem}{p.suffix.lower()}"
        shutil.copy(p, DATA / split / "images" / name)
        (DATA / split / "labels" / f"neg_{p.stem}.txt").write_text("")  # empty = no cigarette
    print(f"{split}: added {b - a}")