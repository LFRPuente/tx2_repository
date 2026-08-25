# TX2 Computer Vision

Tools and MVP applications for TX2 piece-front measurement from video.

The full implementation runbook for moving the calibration tool, red exclusion
zones, individual-piece measurement, SQLite persistence, and operator
corrections into the real-time MVP is:

[`LIVE_MVP_INTEGRATION_README.md`](LIVE_MVP_INTEGRATION_README.md)

The current deployment decision is Windows + IIS, with SQLite during the
initial internal rollout and a future migration to an external Microsoft SQL
Server. This supersedes the older PostgreSQL deployment target. See:

[`docs/WINDOWS_IIS_DEPLOYMENT_PLAN.md`](docs/WINDOWS_IIS_DEPLOYMENT_PLAN.md)

The production backend is served by Waitress on loopback. The internal URL
reserved for the IIS publication is
`https://tx2-measurement.barnstxprod.local`; corporate DNS, an internal-CA
certificate, IIS/ARR, and the Windows service account must be supplied by IT
before that hostname is reachable from VPN clients.

## Main Pieces

- `homography_web_app.py`: Flask tool for homography, YOLO annotation, measurement calibration, Sobel front detection, and frame review.
- `live_mvp_app.py`: Live MVP backend with AXIS video, PLC-triggered recording, measurement processing, APIs, and clip history.
- `templates/`: Flask HTML views for Live, History, homography, ROI selection, and annotation.
- `static/css/` and `static/js/`: presentation and browser behavior for every Flask app, kept outside the Python backends.
- `mvp_react_app/`: React/Vite MVP showing the simplified measurement view with the real video overlay and a diagram.
- `yolo_roi_sobel_projection.py`: YOLO ROI plus Sobel Y projection analysis.
- `tools/plc_timestamp_probe.ipynb`: notebook to probe PLC/OPC UA timestamp tags from the VPN.
- `tools/plc_vision_plain_test.py`: notebook-free PLC/OPC UA smoke test for `MeasureLength` and `VisionWD`.
- `tools/plc_cut_sync_monitor.py`: synchronizes reads to `VisionWD` changes and logs `MeasureLength` edges for cut/measure timing.
- `tools/axis_camera_viewer.py`: opens the live AXIS camera stream through RTSP.
- `tools/axis_live_processor.py`: runs the live AXIS stream through the same homography, YOLO, Sobel, and measurement pipeline used by the app.
- `tools/plc_triggered_video_recorder.py`: records AXIS RTSP clips when the PLC cut/measure tag changes.
- `dataset/` and `dataset_yolo11/`: legacy package-front YOLO datasets.
- `dataset_pieces/`: source annotations with one box per measurable piece.
- `training_videos/`: curated native-resolution AXIS clips stored with Git LFS.
- `prepare_yolo_dataset.py`: creates the piece train/validation split.
- `train_yolo11_pieces.py`: trains the individual-piece YOLO11 detector.
- `runs/detect/runs_tx2/yolo11n_pieces_v3/weights/best.pt`: current individual-piece model.
- `runs/detect/runs_tx2/yolo11n_tubos_v1/weights/best.pt`: legacy fallback model.

Generated videos, local caches, `node_modules`, previews, and redundant checkpoints are intentionally ignored.
Only reviewed raw clips under `training_videos/` are versioned.

## Individual Piece Detection

The target pipeline detects and measures every visible piece independently:

1. YOLO returns one box per piece.
2. Boxes are ordered from left to right and assigned `piece_id`.
3. Sobel Y runs only in the lower front band of each box.
4. Each detected front is forced to be horizontal and limited to that piece.
5. Each piece is measured independently against the shared reference line.

YOLO predictions pass through geometric rules learned from the saved annotations:

- Strongly overlapping boxes are treated as duplicate detections; the highest-confidence box is kept.
- Severe width or height outliers are removed using the median piece profile.
- A large internal gap is considered a missing piece only when normal spacing exists on both sides (`1, 2 -- gap -- 3, 4`).
- A box inferred from a gap is accepted automatically only when Sobel confirms a valid horizontal front inside it.

The previous package annotations are not compatible with this detector and must
not be mixed into `dataset_pieces/`. In the annotation view, draw one box around
each visible piece whose front edge can be measured. Include the full visible
piece down to its lower front edge. Do not draw one box around the whole bundle.
Frames with no measurable pieces remain valid negative examples.

Use **Run model** in the annotation view to test the current checkpoint. Model
boxes appear as a separate dashed overlay and never replace saved annotations.
The side panel reports detections, matches, and annotated pieces missed by the
model.

Legacy frames can be exposed as pending annotation candidates by launching the
app with `--legacy-candidates-dir dataset`. Their old package boxes are never
loaded. Do not enable those candidates while annotating a different source
video, because frame indexes would refer to unrelated images.

Run the annotation app and save the new labels under `dataset_pieces/`:

```powershell
.\run_homography_web_app.ps1
```

After annotating a representative set of frames:

```powershell
python prepare_yolo_dataset.py
python train_yolo11_pieces.py
```

Curated RAW clips can be downloaded on a training computer with:

```powershell
git lfs install
git lfs pull --include="training_videos/**"
```

See [`training_videos/README.md`](training_videos/README.md) for clip hashes,
capture metadata, and commands for opening a clip in the annotation tool.

The v3 training command starts from the v2 individual-piece checkpoint when
available and never overwrites previous versions. Only reviewed annotations
are included. The default command uses `imgsz=1280`; the current v3 checkpoint
was trained explicitly at `imgsz=960` on CUDA.

The trained weights are written to:

```text
runs/detect/runs_tx2/yolo11n_pieces_v3/weights/best.pt
```

Current `v3` checkpoint (2026-07-29):

- Source snapshot: 89 frames, 632 piece boxes, and 6 negative frames.
- The snapshot includes 21 new 4K frames with 189 manually reviewed boxes.
- Temporal-group split: 71 train / 18 validation, with no source-frame pairs
  closer than 90 frames divided across the two splits.
- Base checkpoint: `yolo11n_pieces_v2/weights/best.pt`.
- Best epoch: 47 (early stopping completed at epoch 77).
- Independent `best.pt` validation: precision `0.940`, recall `1.000`,
  mAP50 `0.988`, and mAP50-95 `0.737`.
- Held-out 4K slice at confidence `0.25`: 63/63 boxes found, one false
  positive, precision `0.984`, recall `1.000`, and mean matched IoU `0.895`.
- `best.pt` SHA-256:
  `4B2F6B6292DED79BDA00043BFA6BE9DE1091513FCB342E4485903BD87EC3BCB2`.

Both MVP launchers select that model automatically when it exists. The
individual checkpoint is present in the current repository. A launcher only
falls back to `v2`, then `v1`, and finally the legacy package model if newer
checkpoints are missing.

Vision API responses now include:

- `pieces`: ordered piece boxes, Sobel result, measurement, confidence, and validity.
- `measurement_summary`: detected/valid counts plus minimum, maximum, and average measurements.

The singular `sobel` and `measurement` fields remain temporarily available for
older clients and refer to the first valid piece.

In either supported database backend, one PLC measurement event owns zero or
more piece measurement rows. `measurement_event.detected_piece_count` stores
the detected quantity and `valid_piece_count` stores the validated quantity.
Automatic and operator-entered values remain separate for every piece.

## Python Setup

```powershell
py -m pip install -r requirements.txt
```

## Run The Full Local Tool

```powershell
.\run_homography_web_app.ps1
```

The calibration and annotation launcher reads temporary raw clips produced by
the Live MVP:

```text
outputs/live_plc_clips/<YYYY-MM-DD>/*_raw.mp4
```

It searches recursively, ignores processed MP4s when raw clips exist, skips a
new RAW clip while its MP4 container is still being finalized, and loads only
complete clips matching the resolution and nominal rounded FPS of the newest
complete recording. Small container-reported FPS variations are grouped
together without mixing older Full HD material with the new `2880x2160`
source. The backend exposes the selected files as one continuous timeline
while retaining the source video name, source frame index, and source
timestamp in each annotation.

If the Live folder contains only incomplete RAW files, the launcher falls back
to the reviewed clips in `training_videos/` instead of failing at startup.

When the camera resolution, zoom, or position changes, recreate and validate
these files from a native `2880x2160` raw clip before accepting measurements:

```text
outputs/homography_selection.json
outputs/table_measurement_calibration.json
```

The annotation history creates evenly spaced pending candidates for each
selected raw clip. A candidate is excluded from the training dataset until it
is reviewed and saved. Save frames without boxes as negative examples; draw
one tight box per visible piece on positive frames.

### Spatial scale map

The Measurements view builds a local conversion map from known physical
references without warping the rectified image:

1. Add vertical known-length segments near the left, center, and right sides of
   the working ROI. Add references at multiple heights only when scale also
   changes from top to bottom.
2. Add horizontal references when the X or free manual ruler must be accurate.
3. Click `Guardar y crear mapa`.
4. Inspect the colored overlay, the Y scale range, and the calibrated coverage.
5. Validate the resulting piece measurements in Player.

Each nearly vertical segment contributes to `scaleY(x,y)` and each nearly
horizontal segment contributes to `scaleX(x,y)`. Two or more references along
one direction use clamped linear interpolation. References distributed in both
image dimensions use inverse-distance interpolation. Diagonal references are
ignored because they do not identify an individual axis reliably.

Piece measurements integrate `scaleY` from the saved horizontal reference to
the Sobel front. The X, Y, and free manual rulers integrate the corresponding
local axis scales along their complete paths. Measurements outside calibrated
coverage are explicitly marked as extrapolated and use the nearest boundary
scale.

The map is stored as `scale_map` in
`outputs/table_measurement_calibration.json`. Homography, rectified images,
YOLO boxes, exclusion zones, and training images remain unchanged, so updating
the map does not require retraining YOLO.

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

Configure the AXIS credentials in the terminal that will launch the app. The
current internal rollout uses the local SQLite database
`outputs/tx2_live_mvp.sqlite3`:

```powershell
$env:AXIS_USER="your-user"
$env:AXIS_PASSWORD="your-password"
.\run_live_mvp_app.ps1
```

On GPU hosts, create `.venv-gpu` with a CUDA-enabled PyTorch build. The
launcher prefers `.venv-gpu\Scripts\python.exe` automatically and resolves
`-Device auto` to `cuda:0` when CUDA is available:

```powershell
.\run_live_mvp_app.ps1 -Device auto
```

Use the single-process Waitress launcher for the IIS backend:

```powershell
.\run_live_mvp_production.ps1 -Device auto
```

It binds to `127.0.0.1:8767` by default. Verify it locally with
`http://127.0.0.1:8767/api/health`. The versioned IIS reverse-proxy
configuration is under `deployment/`.

For the temporary direct-IP deployment on the current VPN host, set the
machine-specific address outside the repository and restart production:

```powershell
[Environment]::SetEnvironmentVariable(
    "TX2_LISTEN_ADDRESS", "10.14.6.84", "User"
)
.\run_live_mvp_production.ps1 -Device auto
```

The Live MVP is then served at `http://10.14.6.84:8767`. An administrator must
allow inbound TCP `8767` on the Domain firewall profile, restricted to the
approved VPN address range. This direct mode does not provide IIS Windows
Authentication or TLS, so it is intended only for the internal pilot. Remove
`TX2_LISTEN_ADDRESS` to return to the loopback-only IIS backend.

The selected device, GPU name, PyTorch version, and CUDA runtime are exposed
under `processor` in `/api/live/status`. Use `-Device cpu` only for an explicit
CPU fallback.

Import existing clip sidecars before the first SQLite launch:

```powershell
python tools\import_live_sidecars_to_sqlite.py `
  --output-dir .\outputs `
  --model .\runs\detect\runs_tx2\yolo11n_pieces_v3\weights\best.pt
```

The import is idempotent. The background reconciler also registers legacy
sidecars that have not yet been associated with the selected backend.

Do not configure `TX2_POSTGRES_DSN` for the agreed Windows deployment. The
legacy PostgreSQL repository remains in the codebase for compatibility and
tests, but the next database target is an external Microsoft SQL Server.

Then open:

```text
http://127.0.0.1:8767
```

The Live MVP provides a light interface with the live camera view and
measurement diagram. It requests the AXIS stream at its configured
`2880x2160` resolution. The browser receives a separate 10 FPS fragmented MP4
whose H.264 packets are copied without decoding or re-encoding. A 10 FPS NVDEC
camera buffer feeds PLC inference and processed clips, so the browser stream
does not double YOLO/Sobel work or JPEG encoding. A short FFmpeg startup probe,
per-frame MP4 fragmentation, playback-rate catch-up without MP4 seeks, and
three-sample status hysteresis keep the operator view close to real time without
false disconnect flashes or invented frames above the camera's 10 FPS limit.
YOLO runs on CUDA and
processed clips are written through an asynchronous NVIDIA NVENC queue. The
homography intentionally stays in OpenCV CPU: on the deployed L40S host it is
faster than transferring the full 2880x2160 frame to CUDA and back before YOLO.
Live inference uses
the individual-piece model at an initial confidence of `0.10`, applies the same
geometric rules as the offline tool, filters boxes against configured exclusion
zones before Sobel, and retains the full `box_rules` diagnostics. Measurements
are shown in compact sixteenth-inch format such as `40' 9 1/16"`.

### Live performance baseline

Benchmark from 2026-08-24 using a retained `2880x2160` clip, YOLO v3 at
`imgsz=960`, and the installed NVIDIA L40S:

| Path | Before | Current |
|---|---:|---:|
| Vision frame, median | 110.32 ms | 68.88 ms |
| Vision frame, p95 | 137.26 ms | 89.40 ms |
| Analysis + video submission, median | sequential | 69.08 ms |
| Processing throughput | below 10 FPS | 13.28 FPS |
| End-to-end throughput including MP4 flush | below 10 FPS | 11.12 FPS |

The previous path encoded a full JPEG for every processed frame and wrote H.264
synchronously. The current path creates JPEG evidence only for the canonical
PLC measurement frame, caches frame metadata for overlapping clips, and
overlaps NVENC with the next frame's analysis. `/api/live/status` exposes the
selected encoder, cache hit/miss counts, and the latest per-stage durations.

It connects to the PLC through OPC UA and records one 8-second clip when
`MeasureLength` changes from `False` to `True`. The recording is provisional
until the window closes: if YOLO did not detect any piece during those 8
seconds, the processed MP4, temporary raw MP4, processing captures, sidecar,
and pending database event are discarded.

Temporary raw capture is enabled by `--save-raw-clips`. For a camera source,
the app opens a separate `2880x2160`, 10 FPS RTSP stream and copies its H.264
packets directly to `<clip>_raw.mp4` without decoding or overlays. YOLO remains
at 10 FPS. A camera capability probe on 2026-08-24 confirmed that the deployed
capture mode rejects requested rates above 10 FPS at this resolution. Changing
the sensor capture mode can alter framing and requires homography/calibration
validation first. Remove the flag and the three `--raw-*` launcher arguments
when this temporary data collection is complete.

The automatic per-piece measurements for a retained event come from the camera
frame immediately preceding the PLC signal. At that instant, the
Live camera perimeter turns green and the same green perimeter is embedded in
the processed MP4 for 0.8 seconds. The sidecar records the configured delay,
the actual selected-frame offset, and the marked video frame range.

The database reconciler also verifies retained sidecar event IDs against the
active database. A sidecar marked as synced is imported again if its database
row is missing, keeping History aligned with the 100 retained clips.

The deployed `outputs/table_measurement_calibration.json` currently contains no
`exclusion_zones`. The filtering and overlay support are active in code, but
remain inert until zones are saved from the calibration tool. Do not restore
coordinates from an older calibration without validating them against the
deployed homography.

Open the saved clip history at:

```text
http://127.0.0.1:8767/history
```

Each history entry contains a browser-compatible H.264 MP4, PLC event metadata,
processing snapshots, and the overlays produced while the clip was recorded.
The MP4 itself contains the processed live camera view with the YOLO-derived
reference and front overlays. While temporary raw capture is active, a
`Raw clip` action opens the synchronized camera-only MP4.
The Processing evidence section shows only the canonical `PLC + 2.0 seconds`
camera frame with a green perimeter. All processing snapshots remain available
in the sidecar JSON and selected database for audit, but are not rendered as a
gallery. The app retains the 100 most recent events and removes all processed,
raw, sidecar, and evidence artifacts belonging to older events automatically.

The selected database is the History system of record. Each PLC signal creates an
idempotent event, snapshots the active model/homography/calibration hashes,
stores all processing snapshots, selects one canonical snapshot, and persists
the per-piece automatic measurements from the `PLC + 2.0 seconds` snapshot.
Operator corrections preserve the
automatic value and create immutable audit revisions with optimistic
concurrency checks. The SQLite schema keeps the event, piece, asset and revision
boundaries needed for a later SQL Server adapter without changing the Live or
History API.

The live frame buffer is capped to avoid retaining several gigabytes of images.
Processed clips are streamed directly to disk and resampled to the configured
output FPS. Temporary raw clips are copied directly from H.264, so their
playback duration and native frame rate come from the camera stream. If a
second PLC event arrives while another clip is active, both recording windows
are preserved as separate clips.

Live MVP data is stored directly under:

```text
outputs/live_plc_clips/<date>/
```

MP4 and JPEG assets remain on disk; SQLite stores event, snapshot, piece, asset
and audit metadata. Sidecars record `db_sync_backend` so the
reconciler can idempotently register legacy clips and move between backends.
If the configured database is temporarily unavailable after startup, the MP4
and sidecar are kept with `db_sync_status: pending` and a background reconciler
retries them. For an explicit camera/PLC simulation only:

```powershell
.\run_live_mvp_app.ps1 -DatabaseDisabled
```

The Live and History pages visibly report simulation mode and operator
corrections are disabled.

## Database And Deployment Direction

SQLite is the active backend for the initial internal rollout. PostgreSQL is no
longer the agreed production destination; its repository and migration tools
remain available only as legacy code. The planned production path is Windows,
IIS as the internal HTTPS reverse proxy, one application process, and a future
external Microsoft SQL Server. Keep MP4/JPEG assets on disk and migrate only
event, piece, measurement, asset-path and audit metadata.

## Next Steps On The TX2 Server

Run the following preparation commands from the repository root:

```powershell
git pull
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Activate the latest YOLO checkpoint

A GitHub push does not update the running TX2 server by itself. `git pull`
downloads the checkpoint and launcher changes, but the existing Python process
keeps its previously loaded YOLO weights in memory until the MVP is restarted.

After pulling, verify that the expected `v3` checkpoint exists and matches the
trained artifact:

```powershell
$modelPath = "runs\detect\runs_tx2\yolo11n_pieces_v3\weights\best.pt"
$expectedHash = "4B2F6B6292DED79BDA00043BFA6BE9DE1091513FCB342E4485903BD87EC3BCB2"

if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Missing YOLO checkpoint: $modelPath"
}

$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelPath).Hash
if ($actualHash -ne $expectedHash) {
    throw "Unexpected YOLO checkpoint hash: $actualHash"
}

Write-Host "YOLO v3 checkpoint verified: $actualHash"
```

Stop the currently running MVP with `Ctrl+C`, or stop the Windows service or
scheduled task that owns it, and then start it again:

```powershell
.\run_live_mvp_app.ps1
```

At startup, confirm that the console reports
`yolo11n_pieces_v3\weights\best.pt` and the SHA-256 shown above. Both launchers
prioritize `v3`; they fall back to `v2`, then `v1`, and finally the legacy
package model.

Before the live test, confirm that these calibration and model files are the
ones intended for the production camera:

```text
outputs/homography_selection.json
outputs/roi_selection.json
outputs/table_measurement_calibration.json
runs/detect/runs_tx2/yolo11n_pieces_v3/weights/best.pt
```

Use this checklist for the on-machine validation:

- [ ] Set `AXIS_USER` and `AXIS_PASSWORD`, then run `run_live_mvp_app.ps1`.
- [ ] Confirm `outputs\tx2_live_mvp.sqlite3` is writable by the Windows account running the MVP.
- [ ] Confirm `/api/live/status` reports `nvidia_nvenc`, CUDA YOLO and a healthy SQLite backend.
- [ ] Back up SQLite with the service stopped or through the SQLite backup API; include retained video assets.
- [ ] Open `http://127.0.0.1:8767` and confirm the original camera image remains at its native resolution.
- [ ] Confirm YOLO detects each piece independently in the rectified image and Sobel Y runs only inside each piece ROI.
- [ ] Confirm boxes with more than 20% overlap in a saved red zone are discarded before Sobel.
- [ ] Confirm every detected piece front is horizontal and remains associated with its own piece.
- [ ] Confirm the reference distance is correct and total lengths use the compact `40' 9 1/16"` format.
- [ ] Confirm OPC UA connects to `opc.tcp://10.14.6.48:49320` and `VisionWD` keeps changing.
- [ ] Trigger a `MeasureLength` rising edge and verify that exactly one 8-second clip appears in `/history`.
- [ ] Trigger two events less than eight seconds apart and verify that neither event is lost.
- [ ] Verify each sidecar JSON contains the PLC source timestamp, watchdog value, frame timestamps, and processing snapshots.
- [ ] Verify each saved MP4 keeps the camera resolution and reports 80 frames at 10 FPS for an 8-second window.
- [ ] Verify SQLite stores one event with zero or more per-piece automatic measurements.
- [ ] Verify History can save and clear an operator measurement without changing the automatic value.
- [ ] Verify every operator change creates an immutable audit revision.
- [ ] Confirm the reconciler restores a retained sidecar event missing from SQLite.
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
