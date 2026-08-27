from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v3" / "weights" / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the TX2 YOLO model to TensorRT FP16.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--workspace", type=float, default=1.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    engine_path = model_path.with_suffix(".engine")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
    if engine_path.exists() and not args.force:
        print(f"TensorRT engine already exists: {engine_path}")
        return

    from ultralytics import YOLO

    exported = YOLO(str(model_path)).export(
        format="engine",
        imgsz=args.imgsz,
        batch=1,
        dynamic=True,
        quantize=16,
        simplify=False,
        workspace=args.workspace,
        device=args.device,
        verbose=False,
    )
    print(f"TensorRT engine: {Path(exported).resolve()}")


if __name__ == "__main__":
    main()
