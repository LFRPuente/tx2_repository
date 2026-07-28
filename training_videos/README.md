# Curated RAW Training Clips

These clips are direct H.264 copies from the AXIS camera. They contain no YOLO
overlay and are tracked with Git LFS.

## Download

```powershell
git lfs install
git lfs pull --include="training_videos/**"
```

## Open A Clip

The camera stream is variable frame rate, so open one clip at a time when its
effective FPS differs from the other files:

```powershell
python homography_web_app.py `
  --video training_videos/2026-07-28/<clip-name>.mp4 `
  --output-dir outputs `
  --dataset-dir dataset_pieces `
  --model runs/detect/runs_tx2/yolo11n_pieces_v1/weights/best.pt
```

## 2026-07-28

All five clips are `2880x2160`, approximately eight seconds long, and were
saved only after their recording sidecars and MP4 containers were complete.

| Clip | Effective FPS | Frames | Max pieces | Size | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `live_0074_20260728_234942_998171Z_rising_raw.mp4` | 15.00 | 120 | 3 | 22.47 MiB | `4e602ed2df56b52c3e4b40ad142e3e5f52e0f789ebebc72512d74d3ac36f1092` |
| `live_0073_20260728_234936_989874Z_rising_raw.mp4` | 27.56 | 220 | 2 | 38.80 MiB | `c07a3babd0503030387dd4146dda6211a742aaaf007900264c9a70258875474d` |
| `live_0072_20260728_234837_785004Z_rising_raw.mp4` | 25.57 | 205 | 9 | 35.53 MiB | `fa578dd2b112252e86894b9227ce8dfd9653770503e5cf060d294c40555f3da7` |
| `live_0071_20260728_234821_360159Z_rising_raw.mp4` | 25.43 | 203 | 10 | 36.00 MiB | `72d280943d3108c962b94b6699354e1c575b67ae04ece14e21d36562acb712b2` |
| `live_0070_20260728_234805_331601Z_rising_raw.mp4` | 28.43 | 227 | 9 | 39.96 MiB | `d99ed3504697084d097cf4c40b87933b49e7762ac423597661fc89d3cb6624f9` |
