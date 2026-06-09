# TX2 Computer Vision

Tools and MVP applications for TX2 piece-front measurement from video.

## Main Pieces

- `homography_web_app.py`: Flask tool for homography, YOLO annotation, measurement calibration, Sobel front detection, and frame review.
- `mvp_react_app/`: React/Vite MVP showing the simplified measurement view with the real video overlay and a diagram.
- `yolo_roi_sobel_projection.py`: YOLO ROI plus Sobel Y projection analysis.
- `tools/plc_timestamp_probe.ipynb`: notebook to probe PLC/OPC UA timestamp tags from the VPN.
- `tools/plc_vision_plain_test.py`: notebook-free PLC/OPC UA smoke test for `MeasureLength` and `VisionWD`.
- `tools/plc_cut_sync_monitor.py`: synchronizes reads to `VisionWD` changes and logs `MeasureLength` edges for cut/measure timing.
- `dataset/` and `dataset_yolo11/`: curated YOLO annotation datasets.
- `runs/detect/runs_tx2/yolo11n_tubos_v1/weights/best.pt`: trained model used by the local app.

Generated videos, local caches, `node_modules`, previews, and redundant checkpoints are intentionally ignored.

## Python Setup

```powershell
py -m pip install -r requirements.txt
```

## Run The Full Local Tool

```powershell
.\run_homography_web_app.ps1
```

Then open:

```text
http://127.0.0.1:5050
```

## Run The React MVP

```powershell
.\run_mvp_react_app.ps1
```

Then open:

```text
http://127.0.0.1:5173
```

## PLC Timestamp Notebook

Open:

```text
tools/plc_timestamp_probe.ipynb
```

Default TX2 endpoint:

```text
opc.tcp://10.14.6.48:49320
```

## PLC Plain Tests

Run a read-only smoke test against Kepware/OPC UA:

```powershell
python tools\plc_vision_plain_test.py --samples 100 --interval 0.07
```

Run a cut/measure sync monitor. It watches `VisionWD` and reads `MeasureLength`
on watchdog changes, then writes JSON and CSV reports under `outputs/`:

```powershell
python tools\plc_cut_sync_monitor.py --duration 120 --poll-interval 0.01
```

Default tags:

```text
ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength
ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD
```
