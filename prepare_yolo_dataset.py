from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "dataset_pieces"
DEFAULT_OUTPUT = ROOT / "dataset_pieces_yolo11_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a stratified YOLO train/validation dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-name", default="piece")
    parser.add_argument("--yaml-name", default="tx2_pieces.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-gap-frames",
        type=int,
        default=90,
        help="Keep annotations from the same video and nearby source frames in one split.",
    )
    return parser.parse_args()


def has_boxes(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def annotation_position(source: Path, image_path: Path) -> tuple[str, int]:
    metadata_path = source / "labels" / f"{image_path.stem}.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_data = metadata.get("source") or {}
        video_name = str(source_data.get("video_name") or source_data.get("video") or image_path.stem)
        frame_idx = int(source_data.get("source_frame_idx", metadata.get("frame_idx", 0)))
        return video_name, frame_idx
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return image_path.stem, 0


def temporal_groups(
    source: Path,
    items: list[Path],
    max_gap_frames: int,
) -> list[list[Path]]:
    by_video: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for image_path in items:
        video_name, frame_idx = annotation_position(source, image_path)
        by_video[video_name].append((frame_idx, image_path))

    groups: list[list[Path]] = []
    for entries in by_video.values():
        current: list[Path] = []
        previous_frame: int | None = None
        for frame_idx, image_path in sorted(entries, key=lambda item: (item[0], item[1].name)):
            if previous_frame is None or frame_idx - previous_frame <= max_gap_frames:
                current.append(image_path)
            else:
                groups.append(current)
                current = [image_path]
            previous_frame = frame_idx
        if current:
            groups.append(current)
    return groups


def split_groups(
    groups: list[list[Path]],
    rng: random.Random,
    val_ratio: float,
    required_val_items: set[Path] | None = None,
) -> tuple[list[Path], list[Path]]:
    shuffled = groups[:]
    rng.shuffle(shuffled)
    item_count = sum(len(group) for group in shuffled)
    target_val_count = max(1, round(item_count * val_ratio)) if item_count > 1 else 0
    val_groups: list[list[Path]] = []
    train_groups: list[list[Path]] = []
    val_count = 0
    for group in shuffled:
        if val_count < target_val_count and len(shuffled) - len(val_groups) > 1:
            val_groups.append(group)
            val_count += len(group)
        else:
            train_groups.append(group)
    if not train_groups and val_groups:
        train_groups.append(val_groups.pop())
    required = required_val_items or set()
    if required and not any(image in required for group in val_groups for image in group):
        train_candidate = next(
            (group for group in sorted(train_groups, key=len) if any(image in required for image in group)),
            None,
        )
        val_candidate = next(
            (group for group in sorted(val_groups, key=len) if not any(image in required for image in group)),
            None,
        )
        if train_candidate is not None and val_candidate is not None:
            train_groups.remove(train_candidate)
            val_groups.remove(val_candidate)
            train_groups.append(val_candidate)
            val_groups.append(train_candidate)
    train = sorted(image for group in train_groups for image in group)
    val = sorted(image for group in val_groups for image in group)
    return train, val


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
    max_gap_frames = max(0, int(args.group_gap_frames))
    groups = temporal_groups(source, images, max_gap_frames)
    train, val = split_groups(groups, rng, val_ratio, required_val_items=set(negatives))

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
    print(
        f"groups={len(groups)} group_gap_frames={max_gap_frames} "
        f"train={len(train)} val={len(val)}"
    )
    print(f"yaml={yaml_path}")


if __name__ == "__main__":
    main()
