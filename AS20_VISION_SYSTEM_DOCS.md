# AS20 TX2 Vision System Documentation

Last updated: 2026-07-01

This document explains how the TX2 vision tools in this repository connect to
the PLC, read cut/measure events, process the AXIS camera stream, and save
event clips.

## System Overview

The system has four main parts:

1. PLC/OPC UA connection through Kepware.
2. AXIS camera RTSP capture.
3. Vision processing with homography, YOLO, Sobel edge detection, and measurement calibration.
4. PLC-triggered event recording to MP4 plus JSON metadata.

The PLC tags are read-only from the Python tools in this repository. The tools
do not write values back to the PLC.

## Important Files

| File | Purpose |
| --- | --- |
| `tools/plc_vision_plain_test.py` | Read-only PLC smoke test. Reads `MeasureLength`, reads `VisionWD`, then samples `VisionWD` to confirm it changes. |
| `tools/plc_cut_sync_monitor.py` | Watches `VisionWD`; every watchdog tick reads `MeasureLength` and logs rising/falling/change events to JSON and CSV. |
| `tools/plc_triggered_video_recorder.py` | Records AXIS camera clips when `MeasureLength` changes on the selected edge. Saves MP4 and JSON sidecar files. |
| `tools/axis_camera_viewer.py` | Opens a raw AXIS RTSP stream in an OpenCV window. |
| `tools/axis_live_processor.py` | Runs the live AXIS stream through homography, YOLO, Sobel, and measurement overlay. |
| `homography_web_app.py` | Main local Flask tool for homography selection, YOLO review, Sobel detection, and measurement calibration. |
| `outputs/homography_selection.json` | Saved perspective transform used by the live processor. |
| `outputs/table_measurement_calibration.json` | Measurement calibration used to convert pixel position to inches. |
| `runs/detect/runs_tx2/yolo11n_tubos_v1/weights/best.pt` | YOLO model used by the TX2 app. |

## PLC Connection

Default OPC UA endpoint:

```text
opc.tcp://10.14.6.48:49320
```

Default PLC nodes:

```text
MeasureLength:
ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength

VisionWD:
ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD
```

`VisionWD` is used as a watchdog/heartbeat. It changes continuously while the
PLC/vision interface is alive. The recorder and monitor use this tag as the
timing driver: when `VisionWD` changes, the tools immediately read
`MeasureLength`.

`MeasureLength` is treated as the cut/measure event tag. When it changes from
`False` to `True`, the tools call that a `rising` edge. When it changes from
`True` to `False`, they call that a `falling` edge.

## PLC Smoke Test

Use this to confirm the PLC endpoint is reachable and the watchdog is changing:

```powershell
python tools\plc_vision_plain_test.py --samples 100 --interval 0.07
```

What it does:

1. Connects to the OPC UA endpoint.
2. Reads `MeasureLength` once.
3. Reads `VisionWD` once.
4. Samples `VisionWD` repeatedly.
5. Writes a JSON report under `outputs/`.

Result meanings:

| Result | Meaning |
| --- | --- |
| `PASS` | Nodes were readable and `VisionWD` changed. |
| `WARN` | Nodes were readable, but `VisionWD` did not change in the sample window. |
| `FAIL` | One or more node reads failed. |

Default output name:

```text
outputs/plc_vision_plain_test_<utc>.json
```

## PLC Cut Sync Monitor

Use this when validating whether the cutter/PLC is sending events:

```powershell
python tools\plc_cut_sync_monitor.py --duration 120 --poll-interval 0.01
```

What it does:

1. Connects to OPC UA.
2. Polls `VisionWD`.
3. When `VisionWD` changes, reads `MeasureLength`.
4. Records each watchdog tick.
5. Records each detected event edge.
6. Saves JSON and CSV reports.

Default output files:

```text
outputs/plc_cut_sync_monitor_<utc>.json
outputs/plc_cut_sync_monitor_<utc>.csv
```

Useful flags:

```powershell
python tools\plc_cut_sync_monitor.py `
  --duration 45 `
  --poll-interval 0.01 `
  --output outputs\plc_cut_signal_check_YYYYMMDD_HHMMSS
```

If the report says `EVENTS_FOUND`, the PLC signal changed during the test.

## AXIS Camera Connection

The tools connect to the AXIS camera through RTSP:

```text
rtsp://<user>:<password>@<ip>/axis-media/media.amp?videocodec=<codec>
```

Known camera used for event recording:

```text
10.14.115.241
```

Do not hard-code credentials in scripts or docs. Use environment variables:

```powershell
$env:AXIS_USER="your_user"
$env:AXIS_PASSWORD="your_password"
```

Supported codec values:

| Codec | Notes |
| --- | --- |
| `h264` | More compressed; can be efficient but may be less stable through some OpenCV/FFmpeg paths. |
| `jpeg` | MJPEG style stream; often more stable for the current live/recorder workflow. |

Open a raw viewer:

```powershell
python tools\axis_camera_viewer.py --ip 10.14.115.241 --codec jpeg
```

## Live Vision App

Run the camera through the TX2 processing pipeline:

```powershell
python tools\axis_live_processor.py `
  --ip 10.14.115.241 `
  --codec jpeg `
  --output-dir outputs
```

What the live processor does:

1. Opens the AXIS RTSP stream.
2. Loads `outputs/homography_selection.json`.
3. Warps each processed frame into the calibrated/rectified view.
4. Runs YOLO using `runs/detect/runs_tx2/yolo11n_tubos_v1/weights/best.pt`.
5. Chooses the strongest tube ROI.
6. Runs Sobel projection inside the ROI to find the front edge.
7. Loads `outputs/table_measurement_calibration.json`.
8. Converts the detected edge to a calibrated measurement.
9. Displays an OpenCV overlay.

Keyboard controls:

| Key | Action |
| --- | --- |
| `q` or `Esc` | Exit. |
| `v` | Toggle original/rectified view. |
| `s` | Save snapshot, if `--snapshot-dir` was provided. |

If inference cannot keep up, process fewer frames:

```powershell
python tools\axis_live_processor.py `
  --ip 10.14.115.241 `
  --codec jpeg `
  --output-dir outputs `
  --process-every 3
```

## PLC-Triggered Video Recorder

The recorder saves short clips around each PLC event.

Typical command:

```powershell
$env:AXIS_USER="your_user"
$env:AXIS_PASSWORD="your_password"

python tools\plc_triggered_video_recorder.py `
  --ip 10.14.115.241 `
  --codec jpeg `
  --edge rising `
  --pre-seconds 3 `
  --post-seconds 3 `
  --output-dir outputs\plc_cut_clips\2026-07-01 `
  --max-clips 50
```

How it works:

1. Starts an RTSP camera reader thread.
2. Keeps recent frames in an in-memory pre-roll buffer.
3. Starts a PLC monitor loop.
4. Reads `VisionWD` repeatedly.
5. When `VisionWD` changes, reads `MeasureLength`.
6. Detects the configured event edge.
7. Pulls frames from before and after the PLC event.
8. Writes an MP4 file.
9. Writes a JSON sidecar with PLC timestamps, camera metadata, and frame counts.
10. Stops automatically when `--max-clips` is reached.

Output file pattern:

```text
cut_<index>_<utc_timestamp>_<edge>.mp4
cut_<index>_<utc_timestamp>_<edge>.json
```

Example:

```text
cut_0001_20260630_011103_423333Z_rising.mp4
cut_0001_20260630_011103_423333Z_rising.json
```

The MP4 contains the video clip. The JSON sidecar contains:

| JSON field | Meaning |
| --- | --- |
| `event_index` | Event number within that recorder process. |
| `event_edge` | `rising`, `falling`, or `changed`. |
| `event_value` | PLC event node value at detection time. |
| `previous_event_value` | Previous value used to detect the edge. |
| `event_read_utc` | UTC time when Python read the event node. |
| `event_source_timestamp` | OPC UA source timestamp from the PLC/Kepware data value. |
| `watchdog_value` | `VisionWD` value that triggered the synchronized read. |
| `watchdog_source_timestamp` | OPC UA source timestamp for `VisionWD`. |
| `camera_ip` | Camera IP used for recording. |
| `pre_seconds` | Requested pre-roll duration. |
| `post_seconds` | Requested post-roll duration. |
| `frames_written` | Number of frames written to the clip. |
| `video_fps` | FPS used by OpenCV `VideoWriter`. |
| `first_frame_utc` | UTC timestamp of first saved frame. |
| `last_frame_utc` | UTC timestamp of last saved frame. |
| `video_path` | Full output path of the MP4. |

Important behavior:

- `--max-clips` counts existing `.mp4` files in the output folder.
- If the folder already has at least that many MP4 files, the recorder exits immediately.
- If the recorder is restarted, the event index starts again at `cut_0001`, but file timestamps still keep names unique.
- The recorder writes MP4 using the `mp4v` codec through OpenCV.

## Running Recorder In The Background On Windows

Example PowerShell pattern:

```powershell
$repo = "C:\Users\ven.luis.puente\Documents\IES-TX2\tx2_repository"
$python = "C:\Users\ven.luis.puente\AppData\Local\Programs\Python\Python314\python.exe"
$day = Get-Date -Format "yyyy-MM-dd"
$out = Join-Path $repo (Join-Path "outputs\plc_cut_clips" $day)
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:AXIS_USER="your_user"
$env:AXIS_PASSWORD="your_password"

$args = @(
  "tools\plc_triggered_video_recorder.py",
  "--ip","10.14.115.241",
  "--codec","jpeg",
  "--edge","rising",
  "--pre-seconds","3",
  "--post-seconds","3",
  "--output-dir",$out,
  "--max-clips","50"
)

Start-Process `
  -FilePath $python `
  -ArgumentList $args `
  -WorkingDirectory $repo `
  -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $out "recorder_stdout.log") `
  -RedirectStandardError (Join-Path $out "recorder_stderr.log")
```

Verify it is running:

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -like "python*" -and
    $_.CommandLine -like "*plc_triggered_video_recorder.py*"
  } |
  Select-Object ProcessId, CommandLine
```

Check latest clips:

```powershell
Get-ChildItem outputs\plc_cut_clips -Recurse -Filter *.mp4 |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 FullName, LastWriteTime, Length
```

## Operational Checklist

Before production recording:

1. Confirm PLC connection:

```powershell
python tools\plc_vision_plain_test.py --samples 30 --interval 0.07
```

2. Confirm cutter signal:

```powershell
python tools\plc_cut_sync_monitor.py --duration 45 --poll-interval 0.01
```

3. Confirm camera stream:

```powershell
python tools\axis_camera_viewer.py --ip 10.14.115.241 --codec jpeg
```

4. Start event recorder:

```powershell
python tools\plc_triggered_video_recorder.py `
  --ip 10.14.115.241 `
  --codec jpeg `
  --edge rising `
  --pre-seconds 3 `
  --post-seconds 3 `
  --output-dir outputs\plc_cut_clips\2026-07-01 `
  --max-clips 50
```

5. Confirm MP4 and JSON files are being created.

## Troubleshooting

### No PLC events are saved

Run the sync monitor and check whether `EVENTS_FOUND` appears:

```powershell
python tools\plc_cut_sync_monitor.py --duration 60 --poll-interval 0.01
```

If no events are found:

- Confirm `VisionWD` is changing.
- Confirm `MeasureLength` is the correct PLC tag.
- Confirm the selected edge matches the PLC behavior. Try `--edge any` during testing.

### Recorder starts and exits immediately

Check whether the output folder already reached `--max-clips`.

```powershell
(Get-ChildItem outputs\plc_cut_clips\2026-07-01 -Filter *.mp4).Count
```

Use a new output folder or a larger `--max-clips`.

### Camera does not open

Check:

- Camera IP.
- Network/VPN route.
- AXIS username/password.
- Codec. Try `--codec jpeg` if `h264` is unstable.

### Live processor says homography or calibration is missing

Confirm these files exist:

```text
outputs/homography_selection.json
outputs/table_measurement_calibration.json
```

Run the homography/calibration app if they need to be regenerated:

```powershell
.\run_homography_web_app.ps1
```

## Data Locations

| Path | Contents |
| --- | --- |
| `outputs/plc_vision_plain_test_*.json` | PLC smoke test reports. |
| `outputs/plc_cut_sync_monitor_*.json` | Detailed monitor reports. |
| `outputs/plc_cut_sync_monitor_*.csv` | Monitor rows for spreadsheet review. |
| `outputs/plc_cut_clips/` | Event MP4 clips and JSON sidecars. |
| `outputs/live_app/` | Optional live app logs, if launched with redirected output. |
| `outputs/homography_selection.json` | Homography used by live processing. |
| `outputs/table_measurement_calibration.json` | Measurement calibration. |

## Dependencies

Python dependencies are listed in `requirements.txt`:

```text
asyncua
flask
matplotlib
numpy
opencv-python
pillow
ultralytics
```

Install them with:

```powershell
python -m pip install -r requirements.txt
```

## Safety Notes

- The PLC tools in this repository are read-only.
- Do not commit AXIS credentials.
- Keep production recordings in dated output folders.
- Use `--max-clips` to avoid filling the disk.
- Review MP4 clips and JSON sidecars together; the JSON is the source for PLC timestamps.
