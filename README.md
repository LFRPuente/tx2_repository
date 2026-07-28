# TX2 Computer Vision

Tools and MVP applications for TX2 piece-front measurement from video.

The full implementation runbook for moving the calibration tool, red exclusion
zones, individual-piece measurement, PostgreSQL persistence, and operator
corrections into the real-time MVP is:

[`LIVE_MVP_INTEGRATION_README.md`](LIVE_MVP_INTEGRATION_README.md)

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
- `dataset/` and `dataset_yolo11/`: legacy package-front YOLO datasets.
- `dataset_pieces/`: source annotations with one box per measurable piece.
- `prepare_yolo_dataset.py`: creates the piece train/validation split.
- `train_yolo11_pieces.py`: trains the individual-piece YOLO11 detector.
- `runs/detect/runs_tx2/yolo11n_tubos_v1/weights/best.pt`: legacy fallback model.

Generated videos, local caches, `node_modules`, previews, and redundant checkpoints are intentionally ignored.

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

The trained weights are written to:

```text
runs/detect/runs_tx2/yolo11n_pieces_v1/weights/best.pt
```

Both MVP launchers select that model automatically when it exists. The
individual checkpoint is present in the current repository. A launcher only
falls back to the legacy package model, with an explicit warning, if that file
is missing.

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

The current offline source is the extracted July 24 recording playlist:

```text
C:\Users\luis_\Downloads\20260724_10\
```

The folder contains 12 consecutive `1920x1080`, 30 FPS MKV files. The backend
exposes them as one continuous timeline while retaining the source video name,
source frame index, and source timestamp in each annotation. Because all 12
recordings share the calibrated camera view, they reuse:

```text
outputs/homography_selection.json
outputs/table_measurement_calibration.json
```

The raw recordings are extracted from
`C:\Users\luis_\Downloads\20260724_10.zip` and remain outside Git.

The annotation history initially shows 12 evenly spaced pending candidates per
video, for 144 candidates across the playlist. A candidate is excluded from the
training dataset until it is reviewed and saved. Save frames without boxes as
negative examples; draw one tight box per visible piece on positive frames.

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

Configure the AXIS credentials in the terminal that will launch the app.
While PostgreSQL is pending, the launcher automatically uses the local SQLite
database `outputs/tx2_live_mvp.sqlite3`:

```powershell
$env:AXIS_USER="your-user"
$env:AXIS_PASSWORD="your-password"
.\run_live_mvp_app.ps1
```

Import existing clip sidecars before the first SQLite launch:

```powershell
python tools\import_live_sidecars_to_sqlite.py `
  --output-dir .\outputs `
  --model .\runs\detect\runs_tx2\yolo11n_pieces_v1\weights\best.pt
```

The import is idempotent. The background reconciler also registers legacy
sidecars that have not yet been associated with the selected backend.

When PostgreSQL is available, set the DSN before launching. A configured
PostgreSQL connection is never allowed to fail over silently to SQLite:

```powershell
$env:TX2_POSTGRES_DSN="host=127.0.0.1 port=5432 dbname=tx2_vision user=tx2_vision_app connect_timeout=5"
.\run_live_mvp_app.ps1
```

Then open:

```text
http://127.0.0.1:8767
```

The Live MVP provides a light interface with the live camera view and
measurement diagram. It keeps the AXIS stream at `1920x1080` and targets 10 FPS
for capture, processing, browser updates, and saved clips. Live inference uses
the individual-piece model at an initial confidence of `0.10`, applies the same
geometric rules as the offline tool, filters boxes against configured exclusion
zones before Sobel, and retains the full `box_rules` diagnostics. Measurements
are shown in compact sixteenth-inch format such as `40' 9 1/16"`.

It connects to the PLC through OPC UA and records one 8-second clip when
`MeasureLength` changes from `False` to `True`. The recording is provisional
until the window closes: if YOLO did not detect any piece during those 8
seconds, the MP4, processing captures, sidecar, and pending database event are
discarded.

The automatic per-piece measurements for a retained event come from the first
processed frame at or after `PLC signal + 2.0 seconds`. At that instant, the
Live camera perimeter turns green and the same green perimeter is embedded in
the processed MP4 for 0.8 seconds. The sidecar records the configured delay,
the actual selected-frame offset, and the marked video frame range.

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
reference and front overlays; it is not the raw camera feed.
The Processing evidence section shows only the canonical `PLC + 2.0 seconds`
camera frame with a green perimeter. All processing snapshots remain available
in the sidecar JSON and selected database for audit, but are not rendered as a
gallery. The app retains the 100 most recent clips and removes older clip
artifacts and their database events automatically.

The selected database is the History system of record. Each PLC signal creates an
idempotent event, snapshots the active model/homography/calibration hashes,
stores all processing snapshots, selects one canonical snapshot, and persists
the per-piece automatic measurements from the `PLC + 2.0 seconds` snapshot.
Operator corrections preserve the
automatic value and create immutable audit revisions with optimistic
concurrency checks. SQLite mirrors the PostgreSQL entities so it can be migrated
later without changing the Live or History API.

The live frame buffer is capped to avoid retaining several gigabytes of raw
images. Clips are streamed directly to disk and resampled to the configured
output FPS, so their playback duration matches the PLC recording window. If a
second PLC event arrives while another clip is active, both recording windows
are preserved as separate clips.

Live MVP data is stored directly under:

```text
outputs/live_plc_clips/<date>/
```

MP4 and JPEG assets remain on disk; SQLite or PostgreSQL stores event, snapshot,
piece, asset and audit metadata. Sidecars record `db_sync_backend` so the
reconciler can idempotently register legacy clips and move between backends.
If the configured database is temporarily unavailable after startup, the MP4
and sidecar are kept with `db_sync_status: pending` and a background reconciler
retries them. For an explicit camera/PLC simulation only:

```powershell
.\run_live_mvp_app.ps1 -DatabaseDisabled
```

The Live and History pages visibly report simulation mode and operator
corrections are disabled.

## PostgreSQL Server Setup

PostgreSQL remains the production target. SQLite is the approved temporary
local backend while the Windows service is pending. Follow the PostgreSQL
installation, security, credential, verification, and backup instructions in
[`docs/postgresql_server_setup.md`](docs/postgresql_server_setup.md).

The guide prepares the database service, `tx2_vision` database, and restricted
`tx2_vision_app` login. After the service, password file, and
`TX2_POSTGRES_DSN` are ready, apply the versioned schema:

```powershell
python tools\apply_postgres_migrations.py
```

Validate all existing sidecars without changing them:

```powershell
python tools\migrate_live_sidecars_to_postgres.py `
  --dry-run `
  --output-dir .\outputs `
  --model .\runs\detect\runs_tx2\yolo11n_pieces_v1\weights\best.pt
```

Then run the idempotent import by removing `--dry-run`. The import enriches
legacy sidecars with event/configuration identifiers, registers disk assets by
relative path, and can be rerun safely. The schema and lifecycle contract are
specified in
[`LIVE_MVP_INTEGRATION_README.md`](LIVE_MVP_INTEGRATION_README.md).

After SQLite has been used, validate and migrate all events and immutable
operator revisions:

```powershell
python tools\migrate_sqlite_to_postgres.py --dry-run
python tools\migrate_sqlite_to_postgres.py
```

The migration is idempotent. It preserves event IDs, per-piece automatic
measurements, current operator overrides, revision numbers and audit
timestamps. Keep the SQLite file until the PostgreSQL event and revision counts
have been verified.

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
runs/detect/runs_tx2/yolo11n_pieces_v1/weights/best.pt
```

Use this checklist for the on-machine validation:

- [ ] Set `AXIS_USER` and `AXIS_PASSWORD`, then run `run_live_mvp_app.ps1`.
- [ ] Confirm `TX2_POSTGRES_DSN` and `%APPDATA%\postgresql\pgpass.conf` belong to the Windows account running the MVP.
- [ ] Run `tools\apply_postgres_migrations.py` and confirm the schema checksum is recorded.
- [ ] Run the sidecar migrator in `--dry-run`, then import the existing clips.
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
- [ ] Verify PostgreSQL stores one event with zero or more per-piece automatic measurements.
- [ ] Verify History can save and clear an operator measurement without changing the automatic value.
- [ ] Verify every operator change creates an immutable audit revision.
- [ ] Stop PostgreSQL during a test event, confirm the sidecar becomes `pending`, restart PostgreSQL, and confirm reconciliation changes it to `synced`.
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
