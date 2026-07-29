from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset_pieces_yolo11_v3" / "tx2_pieces.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "detect" / "runs_tx2"
CURRENT_MODEL = DEFAULT_PROJECT / "yolo11n_pieces_v2" / "weights" / "best.pt"
DEFAULT_BASE_MODEL = str(CURRENT_MODEL) if CURRENT_MODEL.exists() else "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TX2 individual-piece YOLO11 detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--name", default="yolo11n_pieces_v3")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared piece dataset not found: {data_path}. "
            "Run prepare_yolo_dataset.py after annotating individual pieces."
        )

    from ultralytics import YOLO

    model = YOLO(str(args.base_model))
    result = model.train(
        data=str(data_path),
        epochs=int(args.epochs),
        patience=int(args.patience),
        batch=int(args.batch),
        imgsz=int(args.imgsz),
        device=str(args.device),
        workers=int(args.workers),
        project=str(DEFAULT_PROJECT),
        name=str(args.name),
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        seed=42,
        deterministic=True,
        amp=True,
        close_mosaic=10,
    )
    save_dir = Path(str(result.save_dir))
    print(f"run={save_dir}")
    print(f"best={save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
