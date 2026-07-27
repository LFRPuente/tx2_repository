from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "dataset_pieces"
DEFAULT_OUTPUT = ROOT / "dataset_pieces_yolo11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a stratified YOLO train/validation dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-name", default="piece")
    parser.add_argument("--yaml-name", default="tx2_pieces.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def has_boxes(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def split_items(
    items: list[Path],
    rng: random.Random,
    val_ratio: float,
) -> tuple[list[Path], list[Path]]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def copy_pair(source: Path, output: Path, image_path: Path, split: str) -> None:
    label_path = source / "labels" / f"{image_path.stem}.txt"
    dst_img = output / "images" / split / image_path.name
    dst_lbl = output / "labels" / split / f"{image_path.stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dst_img)
    if label_path.exists():
        shutil.copy2(label_path, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")


def validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Source and output directories must be different.")
    if not (source / "images").is_dir() or not (source / "labels").is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {source}")
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("The generated dataset output must remain inside the repository.") from exc
    if output == ROOT.resolve():
        raise ValueError("The repository root cannot be used as the generated dataset output.")
    return source, output


def main() -> None:
    args = parse_args()
    source, output = validate_paths(args.source, args.output)
    val_ratio = float(args.val_ratio)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    images = sorted((source / "images").glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No annotated JPG frames were found under {source / 'images'}")
    positives = [path for path in images if has_boxes(source / "labels" / f"{path.stem}.txt")]
    negatives = [path for path in images if path not in positives]

    rng = random.Random(int(args.seed))
    pos_train, pos_val = split_items(positives, rng, val_ratio)
    neg_train, neg_val = split_items(negatives, rng, val_ratio)
    train = sorted(pos_train + neg_train)
    val = sorted(pos_val + neg_val)

    if output.exists():
        shutil.rmtree(output)
    for item in train:
        copy_pair(source, output, item, "train")
    for item in val:
        copy_pair(source, output, item, "val")

    yaml_path = output / str(args.yaml_name)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {args.class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    box_count = sum(
        len(
            [
                line
                for line in (source / "labels" / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        for image in positives
    )
    print(
        f"source_images={len(images)} positives={len(positives)} "
        f"negatives={len(negatives)} boxes={box_count}"
    )
    print(f"train={len(train)} val={len(val)}")
    print(f"yaml={yaml_path}")


if __name__ == "__main__":
    main()
