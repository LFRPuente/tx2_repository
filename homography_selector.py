from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = Path(
    r"C:\Users\luis_\OneDrive\Desktop\tx2_cv\20260407_191100_61A4_B8A44FEF1AB4\20260407_19\20260407_191100_E055.mkv"
)
DEFAULT_SECOND = 16.0
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\luis_\OneDrive\Desktop\tx2_cv\outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona 4 puntos y genera una homografia hacia un rectangulo."
    )
    parser.add_argument("--image", type=Path, help="Imagen de referencia a usar.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Video a usar como fuente.")
    parser.add_argument(
        "--second",
        type=float,
        default=DEFAULT_SECOND,
        help="Segundo del video a extraer cuando no se pasa --image.",
    )
    parser.add_argument(
        "--dest-width",
        type=int,
        default=0,
        help="Ancho del rectangulo de salida. Si es 0 se calcula automaticamente.",
    )
    parser.add_argument(
        "--dest-height",
        type=int,
        default=0,
        help="Alto del rectangulo de salida. Si es 0 se calcula automaticamente.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Ancho maximo de visualizacion en pantalla.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=950,
        help="Alto maximo de visualizacion en pantalla.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para JSON y previews.",
    )
    return parser.parse_args()


def load_reference_image(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise RuntimeError(f"No se pudo abrir la imagen: {args.image}")
        return image, str(args.image)

    if not args.video.exists():
        raise RuntimeError(f"No existe el video: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(args.second) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame en {args.second:.3f} s")
    return frame, f"{args.video} @ {args.second:.3f}s"


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Se requieren exactamente 4 puntos.")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def destination_from_source(
    ordered_src: np.ndarray, dest_width: int = 0, dest_height: int = 0
) -> tuple[np.ndarray, tuple[int, int]]:
    tl, tr, br, bl = ordered_src
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = int(round(max(width_top, width_bottom))) if dest_width <= 0 else int(dest_width)
    height = int(round(max(height_left, height_right))) if dest_height <= 0 else int(dest_height)
    width = max(width, 2)
    height = max(height, 2)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return dst, (width, height)


def compute_display_scale(
    image: np.ndarray, max_width: int, max_height: int
) -> tuple[float, tuple[int, int]]:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    display_size = (int(round(width * scale)), int(round(height * scale)))
    return scale, display_size


def add_text_block(image: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = image.copy()
    x = 15
    y = 25
    for line in lines:
        cv2.putText(
            vis,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 28
    return vis


def draw_selection(
    base_image: np.ndarray, points: list[tuple[float, float]], selected: bool
) -> np.ndarray:
    vis = base_image.copy()
    for idx, point in enumerate(points, start=1):
        px = (int(round(point[0])), int(round(point[1])))
        cv2.circle(vis, px, 12, (0, 255, 0), -1)
        cv2.putText(
            vis,
            str(idx),
            (px[0] + 14, px[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"({px[0]}, {px[1]})",
            (px[0] + 14, px[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        poly = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], selected and len(points) == 4, (0, 0, 255), 3)

    return vis


def draw_grid(base_image: np.ndarray, step_x: int = 400, step_y: int = 200) -> np.ndarray:
    vis = base_image.copy()
    height, width = vis.shape[:2]
    for x in range(0, width, step_x):
        cv2.line(vis, (x, 0), (x, height - 1), (0, 255, 255), 1)
        cv2.putText(vis, str(x), (x + 8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    for y in range(0, height, step_y):
        cv2.line(vis, (0, y), (width - 1, y), (255, 255, 0), 1)
        cv2.putText(vis, str(y), (12, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


@dataclass
class AppState:
    image: np.ndarray
    source_label: str
    args: argparse.Namespace
    points: list[tuple[float, float]]
    display_scale: float
    display_size: tuple[int, int]

    def __post_init__(self) -> None:
        self.image_height, self.image_width = self.image.shape[:2]
        self.output_dir = self.args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.warp_result: np.ndarray | None = None
        self.ordered_points: np.ndarray | None = None
        self.dst_points: np.ndarray | None = None
        self.matrix: np.ndarray | None = None
        self.output_size: tuple[int, int] | None = None
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.dragging = False
        self.drag_anchor: tuple[int, int] | None = None
        self.drag_pan_anchor: tuple[float, float] | None = None

    def get_view_size(self) -> tuple[int, int]:
        view_width = max(50, int(round(self.image_width / self.zoom)))
        view_height = max(50, int(round(self.image_height / self.zoom)))
        return min(view_width, self.image_width), min(view_height, self.image_height)

    def clamp_pan(self) -> None:
        view_width, view_height = self.get_view_size()
        max_pan_x = max(0.0, self.image_width - view_width)
        max_pan_y = max(0.0, self.image_height - view_height)
        self.pan_x = min(max(self.pan_x, 0.0), max_pan_x)
        self.pan_y = min(max(self.pan_y, 0.0), max_pan_y)

    def display_to_image(self, display_x: int, display_y: int) -> tuple[float, float]:
        view_width, view_height = self.get_view_size()
        x_ratio = display_x / max(self.display_size[0], 1)
        y_ratio = display_y / max(self.display_size[1], 1)
        orig_x = self.pan_x + x_ratio * view_width
        orig_y = self.pan_y + y_ratio * view_height
        orig_x = min(max(orig_x, 0.0), self.image_width - 1.0)
        orig_y = min(max(orig_y, 0.0), self.image_height - 1.0)
        return orig_x, orig_y

    def add_point_from_display(self, display_x: int, display_y: int) -> None:
        if len(self.points) >= 4:
            return
        orig_x, orig_y = self.display_to_image(display_x, display_y)
        self.points.append((orig_x, orig_y))
        self.refresh_warp()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
        self.refresh_warp()

    def reset(self) -> None:
        self.points.clear()
        self.refresh_warp()

    def zoom_at(self, factor: float, display_x: int | None = None, display_y: int | None = None) -> None:
        old_zoom = self.zoom
        new_zoom = min(max(self.zoom * factor, 1.0), 12.0)
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        if display_x is None or display_y is None:
            display_x = self.display_size[0] // 2
            display_y = self.display_size[1] // 2

        anchor_x, anchor_y = self.display_to_image(display_x, display_y)
        self.zoom = new_zoom
        view_width, view_height = self.get_view_size()
        x_ratio = display_x / max(self.display_size[0], 1)
        y_ratio = display_y / max(self.display_size[1], 1)
        self.pan_x = anchor_x - x_ratio * view_width
        self.pan_y = anchor_y - y_ratio * view_height
        self.clamp_pan()

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan_x += dx
        self.pan_y += dy
        self.clamp_pan()

    def start_drag(self, display_x: int, display_y: int) -> None:
        self.dragging = True
        self.drag_anchor = (display_x, display_y)
        self.drag_pan_anchor = (self.pan_x, self.pan_y)

    def drag_to(self, display_x: int, display_y: int) -> None:
        if not self.dragging or self.drag_anchor is None or self.drag_pan_anchor is None:
            return
        view_width, view_height = self.get_view_size()
        dx = (display_x - self.drag_anchor[0]) * view_width / max(self.display_size[0], 1)
        dy = (display_y - self.drag_anchor[1]) * view_height / max(self.display_size[1], 1)
        self.pan_x = self.drag_pan_anchor[0] - dx
        self.pan_y = self.drag_pan_anchor[1] - dy
        self.clamp_pan()

    def stop_drag(self) -> None:
        self.dragging = False
        self.drag_anchor = None
        self.drag_pan_anchor = None

    def refresh_warp(self) -> None:
        self.warp_result = None
        self.ordered_points = None
        self.dst_points = None
        self.matrix = None
        self.output_size = None
        if len(self.points) != 4:
            return

        ordered = order_points(np.asarray(self.points, dtype=np.float32))
        dst_points, output_size = destination_from_source(
            ordered, self.args.dest_width, self.args.dest_height
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst_points)
        warp = cv2.warpPerspective(self.image, matrix, output_size)

        self.ordered_points = ordered
        self.dst_points = dst_points
        self.matrix = matrix
        self.output_size = output_size
        self.warp_result = warp

    def render_source(self) -> np.ndarray:
        base = draw_grid(self.image)
        vis = draw_selection(base, self.points, selected=len(self.points) == 4)
        view_width, view_height = self.get_view_size()
        x0 = int(round(self.pan_x))
        y0 = int(round(self.pan_y))
        x1 = x0 + view_width
        y1 = y0 + view_height
        vis = vis[y0:y1, x0:x1]
        info = [
            f"Fuente: {self.source_label}",
            "Click izquierdo: agregar punto",
            "Rueda o +/-: zoom | arrastre derecho: pan",
            "Flechas/WASD: mover | u: deshacer | r: reiniciar | s: guardar | q: salir",
            f"Puntos: {len(self.points)}/4 | zoom: {self.zoom:.2f}x",
        ]
        if self.output_size is not None:
            info.append(f"Rectangulo destino: {self.output_size[0]} x {self.output_size[1]}")
        vis = add_text_block(vis, info)
        return cv2.resize(vis, self.display_size, interpolation=cv2.INTER_AREA)

    def render_warp(self) -> np.ndarray:
        if self.warp_result is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder = add_text_block(
                placeholder,
                [
                    "Warp preview",
                    "Selecciona 4 puntos",
                    "El orden puede ser cualquiera",
                ],
            )
            return placeholder

        preview = self.warp_result.copy()
        preview = add_text_block(
            preview,
            [
                "Warp preview",
                "s guarda JSON + previews + matriz",
            ],
        )
        scale, size = compute_display_scale(preview, self.args.max_width, self.args.max_height)
        return cv2.resize(preview, size, interpolation=cv2.INTER_AREA)

    def save_outputs(self) -> Path:
        if self.warp_result is None or self.matrix is None or self.ordered_points is None:
            raise RuntimeError("Todavia no hay una homografia valida para guardar.")

        json_path = self.output_dir / "homography_selection.json"
        source_path = self.output_dir / "homography_selection_source.jpg"
        warp_path = self.output_dir / "homography_selection_warp.jpg"

        source_vis = draw_selection(self.image, list(map(tuple, self.ordered_points.tolist())), selected=True)
        cv2.imwrite(str(source_path), source_vis)
        cv2.imwrite(str(warp_path), self.warp_result)

        payload = {
            "source": self.source_label,
            "ordered_source_points": self.ordered_points.tolist(),
            "destination_points": self.dst_points.tolist(),
            "output_size": list(self.output_size),
            "homography_matrix": self.matrix.tolist(),
            "source_preview": str(source_path),
            "warp_preview": str(warp_path),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path


def run_app(args: argparse.Namespace) -> None:
    image, source_label = load_reference_image(args)
    display_scale, display_size = compute_display_scale(image, args.max_width, args.max_height)
    state = AppState(
        image=image,
        source_label=source_label,
        args=args,
        points=[],
        display_scale=display_scale,
        display_size=display_size,
    )

    window_source = "Homography Source"
    window_warp = "Homography Warp"

    cv2.namedWindow(window_source, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_warp, cv2.WINDOW_NORMAL)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_point_from_display(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.start_drag(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.drag_to(x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            state.stop_drag()
        elif event == cv2.EVENT_MOUSEWHEEL:
            state.zoom_at(1.2 if _flags > 0 else 1 / 1.2, x, y)

    cv2.setMouseCallback(window_source, on_mouse)

    while True:
        cv2.imshow(window_source, state.render_source())
        cv2.imshow(window_warp, state.render_warp())
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            state.undo()
        elif key == ord("r"):
            state.reset()
        elif key == ord("s"):
            try:
                json_path = state.save_outputs()
                print(f"Homografia guardada en: {json_path}")
            except RuntimeError as exc:
                print(exc)
        elif key in (ord("+"), ord("=")):
            state.zoom_at(1.2)
        elif key in (ord("-"), ord("_")):
            state.zoom_at(1 / 1.2)
        elif key in (ord("w"), ord("W"), 2490368):
            state.pan_by(0, -40 / state.zoom)
        elif key == 2621440:
            state.pan_by(0, 40 / state.zoom)
        elif key in (ord("a"), ord("A"), 2424832):
            state.pan_by(-40 / state.zoom, 0)
        elif key in (ord("d"), ord("D"), 2555904):
            state.pan_by(40 / state.zoom, 0)

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_app(args)


if __name__ == "__main__":
    main()
