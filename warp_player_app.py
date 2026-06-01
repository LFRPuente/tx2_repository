from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


DEFAULT_VIDEO_DIR = Path(r"C:\Users\luis_\OneDrive\Desktop\tx1")
DEFAULT_POINTS = np.array(
    [
        [586.0, 110.0],  # P1
        [944.0, 534.0],  # P2
        [662.0, 609.0],  # P3
        [341.0, 165.0],  # P4
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona un video y reproduce la vista warp rectificada."
    )
    parser.add_argument("--video", type=Path, help="Ruta del video a abrir.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Ancho maximo de la ventana de salida.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=950,
        help="Alto maximo de la ventana de salida.",
    )
    return parser.parse_args()


def pick_video_file() -> Path | None:
    root = Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Selecciona un video",
        initialdir=str(DEFAULT_VIDEO_DIR),
        filetypes=[
            ("Videos", "*.mkv *.mp4 *.avi *.mov *.m4v"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def compute_destination_points(src: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    tl, tr, br, bl = src
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = max(int(round(max(width_top, width_bottom))), 2)
    height = max(int(round(max(height_left, height_right))), 2)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return dst, (width, height)


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Se requieren exactamente 4 puntos.")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def compute_scale(frame: np.ndarray, max_width: int, max_height: int) -> tuple[float, tuple[int, int]]:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    return scale, (int(round(width * scale)), int(round(height * scale)))


def draw_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(
            vis,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 30
    return vis


@dataclass
class WarpConfig:
    max_width: int = 1600
    max_height: int = 950


def build_warp_from_frame(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    ordered = order_points(DEFAULT_POINTS)
    dst, size = compute_destination_points(ordered)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return matrix, size


def run(video_path: Path, config: WarpConfig) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"No se pudo leer el primer frame: {video_path}")

    matrix, warp_size = build_warp_from_frame(first_frame)
    scale, preview_size = compute_scale(first_frame, config.max_width, config.max_height)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    delay_ms = max(1, int(round(1000.0 / fps)))

    cv2.namedWindow("Warp", cv2.WINDOW_NORMAL)
    paused = False
    frame_index = 1

    # Procesar y mostrar el primer frame ya leído.
    while True:
        warped = cv2.warpPerspective(first_frame, matrix, warp_size)
        warped = draw_text(
            warped,
            [
                "Warp player",
                "espacio: pausa | n: frame siguiente | q/esc: salir",
                f"frame={frame_index}",
            ],
        )
        warped_preview = cv2.resize(warped, compute_scale(warped, config.max_width, config.max_height)[1])
        cv2.imshow("Warp", warped_preview)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("n") and paused:
            ok, first_frame = cap.read()
            if not ok:
                break
            frame_index += 1
        elif not paused:
            ok, first_frame = cap.read()
            if not ok:
                break
            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    video_path = args.video if args.video is not None else pick_video_file()
    if video_path is None:
        print("No se selecciono ningun video.")
        return
    if not video_path.exists():
        raise RuntimeError(f"No existe el video: {video_path}")

    run(video_path, WarpConfig(max_width=args.max_width, max_height=args.max_height))


if __name__ == "__main__":
    main()
