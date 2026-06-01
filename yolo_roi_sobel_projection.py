from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = Path(r"C:\Users\luis_\Downloads\20260508_000307_7F66.mkv")
DEFAULT_HOMOGRAPHY = ROOT / "outputs" / "homography_selection.json"
DEFAULT_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "yolo_sobel_projection"


@dataclass
class ProjectionConfig:
    roi_pad_x: int = 12
    roi_pad_y: int = 16
    min_yolo_conf: float = 0.25
    clahe_clip: float = 2.0
    blur_ksize: tuple[int, int] = (31, 1)
    sobel_ksize: int = 3
    bin_width: int = 6
    profile_smooth: int = 9
    score_keep_percentile: float = 35.0
    line_inlier_tol: float = 8.0
    min_points: int = 12
    max_abs_slope: float = 0.35
    edge_percentile: float = 88.0
    edge_band_start: float = 0.0
    edge_band_end: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta ROI con YOLO y proyecta edges Sobel Y en X para estimar una linea."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-sec", type=float, default=80.0)
    parser.add_argument("--end-sec", type=float, default=125.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--roi-pad-x", type=int, default=12)
    parser.add_argument("--roi-pad-y", type=int, default=16)
    parser.add_argument("--edge-band-start", type=float, default=0.0)
    parser.add_argument("--edge-band-end", type=float, default=1.0)
    parser.add_argument("--score-percentile", type=float, default=35.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args()


def load_homography(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["homography_matrix"], dtype=np.float64)
    output_size = tuple(int(v) for v in payload["output_size"])
    return matrix, output_size


def read_frame_at(cap: cv2.VideoCapture, second: float) -> tuple[int, np.ndarray]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(0, min(int(round(second * fps)), max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"No se pudo leer frame {frame_idx} @ {second:.3f}s")
    return frame_idx, frame


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def clamp_roi(box: np.ndarray, image_shape: tuple[int, int, int], cfg: ProjectionConfig) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = box.astype(float).tolist()
    x0 = int(np.floor(x0 - cfg.roi_pad_x))
    y0 = int(np.floor(y0 - cfg.roi_pad_y))
    x1 = int(np.ceil(x1 + cfg.roi_pad_x))
    y1 = int(np.ceil(y1 + cfg.roi_pad_y))
    x0 = max(0, min(x0, width - 2))
    y0 = max(0, min(y0, height - 2))
    x1 = max(x0 + 2, min(x1, width - 1))
    y1 = max(y0 + 2, min(y1, height - 1))
    return x0, y0, x1, y1


def choose_best_box(result, cfg: ProjectionConfig) -> tuple[np.ndarray, float] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    candidates = []
    for box, score in zip(xyxy, conf):
        if float(score) < cfg.min_yolo_conf:
            continue
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        candidates.append((area * float(score), box, float(score)))
    if not candidates:
        return None
    _rank, box, score = max(candidates, key=lambda item: item[0])
    return box, score


def edge_response_from_roi(roi: np.ndarray, cfg: ProjectionConfig) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, cfg.blur_ksize, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=cfg.sobel_ksize)
    edge = np.abs(grad_y)
    return gray, edge


def project_edge_line(edge: np.ndarray, cfg: ProjectionConfig) -> dict[str, object]:
    height, width = edge.shape
    y_start = int(round(cfg.edge_band_start * height))
    y_end = int(round(cfg.edge_band_end * height))
    y_start = max(0, min(y_start, height - 2))
    y_end = max(y_start + 2, min(y_end, height))

    xs: list[int] = []
    ys: list[int] = []
    scores: list[float] = []

    half = max(1, cfg.bin_width // 2)
    for x in range(half, max(half + 1, width - half), cfg.bin_width):
        x0 = max(0, x - half)
        x1 = min(width, x + half + 1)
        stripe = edge[y_start:y_end, x0:x1]
        if stripe.size == 0:
            continue
        profile = smooth_1d(stripe.mean(axis=1), cfg.profile_smooth)
        rel_y = int(np.argmax(profile))
        xs.append(x)
        ys.append(y_start + rel_y)
        scores.append(float(profile[rel_y]))

    if len(xs) < cfg.min_points:
        return {
            "is_valid": False,
            "line": None,
            "points": [],
            "confidence": 0.0,
            "crm_px": float("inf"),
            "threshold": float("nan"),
        }

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    score_threshold = float(np.percentile(score_arr, cfg.score_keep_percentile))
    score_mask = score_arr >= score_threshold

    if int(score_mask.sum()) < cfg.min_points:
        return {
            "is_valid": False,
            "line": None,
            "points": list(zip(xs, ys, scores)),
            "confidence": 0.0,
            "crm_px": float("inf"),
            "threshold": score_threshold,
        }

    slope, intercept = np.polyfit(x_arr[score_mask], y_arr[score_mask], 1, w=score_arr[score_mask])
    residual = np.abs(y_arr - (slope * x_arr + intercept))
    inliers = score_mask & (residual <= cfg.line_inlier_tol)

    if int(inliers.sum()) >= cfg.min_points:
        slope, intercept = np.polyfit(x_arr[inliers], y_arr[inliers], 1, w=score_arr[inliers])
        residual = np.abs(y_arr - (slope * x_arr + intercept))
        inliers = score_mask & (residual <= cfg.line_inlier_tol)

    crm_px = float(np.sqrt(np.mean(residual[inliers] ** 2))) if int(inliers.sum()) else float("inf")
    confidence = float(inliers.sum()) / float(len(x_arr))
    confidence *= float(np.median(score_arr[inliers])) / (float(score_arr.mean()) + 1e-6) if int(inliers.sum()) else 0.0
    is_valid = int(inliers.sum()) >= cfg.min_points and abs(float(slope)) <= cfg.max_abs_slope

    return {
        "is_valid": bool(is_valid),
        "line": (float(slope), float(intercept)),
        "points": list(zip(xs, ys, scores, inliers.tolist())),
        "confidence": float(confidence),
        "crm_px": crm_px,
        "threshold": score_threshold,
    }


def make_edge_overlay(edge: np.ndarray, cfg: ProjectionConfig) -> np.ndarray:
    norm = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    threshold = np.percentile(edge, cfg.edge_percentile)
    mask = (edge >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    heat[mask > 0] = (0, 255, 255)
    return heat


def draw_detection(
    rectified: np.ndarray,
    roi_px: tuple[int, int, int, int] | None,
    yolo_conf: float,
    projection: dict[str, object] | None,
    cfg: ProjectionConfig,
    frame_idx: int,
    time_sec: float,
) -> np.ndarray:
    overlay = rectified.copy()
    status = "NO ROI"
    color = (0, 0, 255)

    if roi_px is not None:
        x0, y0, x1, y1 = roi_px
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 160, 0), 2)
        status = "NO LINE"
        color = (0, 165, 255)

        if projection is not None:
            for point in projection["points"]:
                px, py, _score, *rest = point
                is_inlier = bool(rest[0]) if rest else False
                pcolor = (0, 255, 0) if is_inlier else (0, 180, 255)
                cv2.circle(overlay, (x0 + int(px), y0 + int(py)), 2, pcolor, -1)

            if projection["line"] is not None:
                slope, intercept = projection["line"]
                ly = y0 + slope * 0.0 + intercept
                ry = y0 + slope * (x1 - x0 - 1) + intercept
                color = (0, 255, 0) if projection["is_valid"] else (0, 165, 255)
                status = "OK" if projection["is_valid"] else "LINEA DEBIL"
                cv2.line(overlay, (x0, int(round(ly))), (x1, int(round(ry))), color, 3)

            label = "YOLO {:.2f} | conf {:.2f} | crm {:.1f}px".format(
                yolo_conf,
                projection["confidence"],
                projection["crm_px"],
            )
            cv2.putText(
                overlay,
                label,
                (x0 + 4, max(22, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    header = f"frame={frame_idx} t={time_sec:.2f}s | {status}"
    cv2.putText(
        overlay,
        header,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return overlay


def detect_frame(
    model: YOLO,
    raw_frame: np.ndarray,
    homography_matrix: np.ndarray,
    output_size: tuple[int, int],
    cfg: ProjectionConfig,
    imgsz: int,
    conf: float,
    frame_idx: int,
    time_sec: float,
) -> dict[str, object]:
    rectified = cv2.warpPerspective(raw_frame, homography_matrix, output_size)
    yolo_result = model.predict(rectified, imgsz=imgsz, conf=conf, verbose=False)[0]
    best = choose_best_box(yolo_result, cfg)
    if best is None:
        overlay = draw_detection(rectified, None, 0.0, None, cfg, frame_idx, time_sec)
        return {
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "has_roi": False,
            "is_valid": False,
            "overlay": overlay,
            "edge_overlay": None,
        }

    box, yolo_conf = best
    roi_px = clamp_roi(box, rectified.shape, cfg)
    x0, y0, x1, y1 = roi_px
    roi = rectified[y0:y1, x0:x1]
    _gray, edge = edge_response_from_roi(roi, cfg)
    projection = project_edge_line(edge, cfg)
    overlay = draw_detection(rectified, roi_px, yolo_conf, projection, cfg, frame_idx, time_sec)
    edge_overlay = make_edge_overlay(edge, cfg)

    left_y = right_y = float("nan")
    slope = intercept = float("nan")
    if projection["line"] is not None:
        slope, intercept = projection["line"]
        left_y = float(y0 + intercept)
        right_y = float(y0 + slope * (x1 - x0 - 1) + intercept)

    return {
        "frame_idx": frame_idx,
        "time_sec": time_sec,
        "has_roi": True,
        "is_valid": bool(projection["is_valid"]),
        "yolo_conf": float(yolo_conf),
        "roi_x0": x0,
        "roi_y0": y0,
        "roi_x1": x1,
        "roi_y1": y1,
        "slope_roi": float(slope),
        "intercept_roi": float(intercept),
        "left_y_full": left_y,
        "right_y_full": right_y,
        "edge_confidence": float(projection["confidence"]),
        "crm_px": float(projection["crm_px"]),
        "overlay": overlay,
        "edge_overlay": edge_overlay,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "frame_idx",
        "time_sec",
        "has_roi",
        "is_valid",
        "yolo_conf",
        "roi_x0",
        "roi_y0",
        "roi_x1",
        "roi_y1",
        "slope_roi",
        "intercept_roi",
        "left_y_full",
        "right_y_full",
        "edge_confidence",
        "crm_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    cfg = ProjectionConfig(
        min_yolo_conf=args.conf,
        roi_pad_x=args.roi_pad_x,
        roi_pad_y=args.roi_pad_y,
        edge_band_start=args.edge_band_start,
        edge_band_end=args.edge_band_end,
        score_keep_percentile=args.score_percentile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    edges_dir = args.output_dir / "edges"
    samples_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    homography_matrix, output_size = load_homography(args.homography)
    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.video}")

    rows: list[dict[str, object]] = []
    seconds = np.arange(args.start_sec, args.end_sec + 1e-6, args.step_sec)
    video_writer = None
    fps = 1.0 / args.step_sec if args.step_sec > 0 else 1.0

    for idx, second in enumerate(seconds):
        frame_idx, raw_frame = read_frame_at(cap, float(second))
        result = detect_frame(
            model,
            raw_frame,
            homography_matrix,
            output_size,
            cfg,
            args.imgsz,
            args.conf,
            frame_idx,
            float(second),
        )
        rows.append(result)

        overlay = result["overlay"]
        if idx < args.sample_limit:
            stem = f"frame_{frame_idx:06d}_{second:07.2f}s"
            cv2.imwrite(str(samples_dir / f"{stem}.jpg"), overlay)
            if result.get("edge_overlay") is not None:
                cv2.imwrite(str(edges_dir / f"{stem}_edges.jpg"), result["edge_overlay"])

        if args.save_video:
            if video_writer is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(args.output_dir / "overlay.mp4"), fourcc, fps, (w, h))
            video_writer.write(overlay)

        status = "OK" if result.get("is_valid") else "NO"
        print(f"{idx + 1:03d}/{len(seconds):03d} frame={frame_idx} t={second:.2f}s {status}")

    cap.release()
    if video_writer is not None:
        video_writer.release()
    write_csv(args.output_dir / "projection_lines.csv", rows)
    print(f"CSV: {args.output_dir / 'projection_lines.csv'}")
    print(f"Samples: {samples_dir}")


if __name__ == "__main__":
    main()
