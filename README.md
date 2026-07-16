# TX2 Computer Vision

Tools and MVP applications for TX2 piece-front measurement from video.

## Main Pieces

- `homography_web_app.py`: Flask tool for homography, YOLO annotation, measurement calibration, Sobel front detection, and frame review.
- `live_mvp_app.py`: live Flask MVP with AXIS video, PLC-triggered recording, measurement processing, and clip history.
- `mvp_react_app/`: React/Vite MVP showing the simplified measurement view with the real video overlay and a diagram.
- `yolo_roi_sobel_projection.py`: YOLO ROI plus Sobel Y projection analysis.
- `tools/plc_timestamp_probe.ipynb`: notebook to probe PLC/OPC UA timestamp tags from the VPN.
- `tools/plc_vision_plain_test.py`: notebook-free PLC/OPC UA smoke test for `MeasureLength` and `VisionWD`.
- `tools/plc_cut_sync_monitor.py`: synchronizes reads to `VisionWD` changes and logs `MeasureLength` edges for cut/measure timing.
- `tools/axis_camera_viewer.py`: opens the live AXIS camera stream through RTSP.
- `tools/axis_live_processor.py`: runs the live AXIS stream through the same homography, YOLO, Sobel, and measurement pipeline used by the app.
- `tools/plc_triggered_video_recorder.py`: records AXIS RTSP clips when the PLC cut/measure tag changes.
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

## Run The Live MVP

Configure the AXIS credentials in the terminal that will launch the app:

```powershell
$env:AXIS_USER="your-user"
$env:AXIS_PASSWORD="your-password"
.\run_live_mvp_app.ps1
```

Then open:

```text
http://127.0.0.1:8767
```

The Live MVP provides a light interface with the live camera view and
measurement diagram. It keeps the AXIS stream at `1920x1080` and targets 10 FPS
for capture, processing, browser updates, and saved clips. It connects to the
PLC through OPC UA and starts a clip when `MeasureLength` changes from `False`
to `True`. The next signal closes that clip and starts the following one; if no
new signal arrives, recording stops after 10 seconds.

Open the saved clip history at:

```text
http://127.0.0.1:8767/history
```

Each history entry contains the MP4, PLC event metadata, processing snapshots,
and the overlays produced while the clip was recorded. The app retains the 100
most recent clips and removes older clip artifacts automatically.

The live frame buffer is capped to avoid retaining several gigabytes of raw
images. Clips are streamed directly to disk and resampled to the configured
output FPS, so their playback duration matches the PLC recording window. If a
second PLC event arrives while a clip is active, the current clip ends at that
event boundary and the next clip begins there, so saved recordings do not
overlap.

Live MVP data is stored directly under:

```text
outputs/live_plc_clips/<date>/
```

Storage currently uses MP4, JSON, and JPEG files. SQLite is not used.

## Next Steps On The TX2 Server

Run the following preparation commands from the repository root:

```powershell
git pull
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Before the live test, confirm that these calibration and model files are the
ones intended for the production camera:

```text
outputs/homography_selection.json
outputs/roi_selection.json
outputs/table_measurement_calibration.json
runs/detect/runs_tx2/yolo11n_tubos_v1/weights/best.pt
```

Use this checklist for the on-machine validation:

- [ ] Set `AXIS_USER` and `AXIS_PASSWORD`, then run `run_live_mvp_app.ps1`.
- [ ] Open `http://127.0.0.1:8767` and confirm the original camera image remains at its native resolution.
- [ ] Confirm YOLO detects the package in the rectified image and Sobel Y runs only inside the selected YOLO ROI.
- [ ] Confirm the detected front is horizontal and its projection spans the full original image.
- [ ] Confirm the reference line, distance to reference, and total measurement are correct in inches.
- [ ] Confirm OPC UA connects to `opc.tcp://10.14.6.48:49320` and `VisionWD` keeps changing.
- [ ] Trigger a `MeasureLength` rising edge and verify that one clip of at most 10 seconds appears in `/history`.
- [ ] Trigger two events less than 10 seconds apart and verify that the first clip ends when the second begins, with no overlap.
- [ ] Verify each sidecar JSON contains the PLC source timestamp, watchdog value, frame timestamps, and processing snapshots.
- [ ] Verify each saved MP4 keeps the camera resolution and reports at most 100 frames at 10 FPS.
- [ ] Leave the app running for at least 30 minutes and confirm the frame buffer stays capped and process memory does not grow continuously.

After the live validation, decide the production host binding, Windows service
or scheduled-task setup, log retention, clip retention, and final storage path.
The current PLC tag `MeasureLength` is used as a Boolean trigger; locating a
separate numeric PLC length tag remains necessary if the calculated measurement
must also be compared with or written back to the PLC.

## AXIS Live Camera

The AXIS cameras at `10.14.115.74` and `10.14.115.75` expose the standard
RTSP endpoint, but require Digest authentication.

Open a raw live view:

```powershell
$env:AXIS_USER="user"
$env:AXIS_PASSWORD="password"
python tools\axis_camera_viewer.py --ip 10.14.115.74
```

Run the live stream through the TX2 processing pipeline:

```powershell
$env:AXIS_USER="user"
$env:AXIS_PASSWORD="password"
python tools\axis_live_processor.py --ip 10.14.115.74 --output-dir outputs
```

If inference falls behind the stream, process every third frame:

```powershell
python tools\axis_live_processor.py --ip 10.14.115.74 --output-dir outputs --process-every 3
```

## PLC Triggered Video Clips

Record the live AXIS stream around each PLC cut/measure event. The script keeps
a pre-roll buffer, waits for the configured OPC UA tag edge, then writes an MP4
clip plus a JSON sidecar with PLC and frame timestamps.

```powershell
$env:AXIS_USER="your-user"
$env:AXIS_PASSWORD="your-password"
python tools\plc_triggered_video_recorder.py --ip 10.14.115.241 --edge rising --pre-seconds 3 --post-seconds 3
```

Clips are saved under:

```text
outputs/plc_cut_clips/
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
