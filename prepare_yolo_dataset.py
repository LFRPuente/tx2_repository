from __future__ import annotations

import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset"
OUT = ROOT / "dataset_yolo11"
SEED = 42
VAL_RATIO = 0.2


def has_boxes(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def split_items(items: list[Path]) -> tuple[list[Path], list[Path]]:
    rng = random.Random(SEED)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * VAL_RATIO)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def copy_pair(image_path: Path, split: str) -> None:
    label_path = SOURCE / "labels" / f"{image_path.stem}.txt"
    dst_img = OUT / "images" / split / image_path.name
    dst_lbl = OUT / "labels" / split / f"{image_path.stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dst_img)
    if label_path.exists():
        shutil.copy2(label_path, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")


def main() -> None:
    images = sorted((SOURCE / "images").glob("*.jpg"))
    positives = [p for p in images if has_boxes(SOURCE / "labels" / f"{p.stem}.txt")]
    negatives = [p for p in images if p not in positives]

    pos_train, pos_val = split_items(positives)
    neg_train, neg_val = split_items(negatives)
    train = sorted(pos_train + neg_train)
    val = sorted(pos_val + neg_val)

    if OUT.exists():
        shutil.rmtree(OUT)
    for item in train:
        copy_pair(item, "train")
    for item in val:
        copy_pair(item, "val")

    yaml = OUT / "tx2_tubes.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {OUT.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: tubo",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"source_images={len(images)} positives={len(positives)} negatives={len(negatives)}")
    print(f"train={len(train)} val={len(val)}")
    print(f"yaml={yaml}")


if __name__ == "__main__":
    main()
