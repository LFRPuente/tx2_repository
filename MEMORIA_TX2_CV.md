# Memoria de trabajo - TX2 CV

Fecha de actualizacion: 2026-05-11

## Estado general

Workspace actual:

```text
C:\Users\luis_\OneDrive\Desktop\tx2_cv
```

Este directorio no esta montado sobre un repositorio Git. `git status` falla con:

```text
fatal: not a git repository (or any of the parent directories): .git
```

El contexto anterior `MEMORIA_TX1_TUBOS.md` tenia referencias a `tx1`. Para este proyecto, el contexto valido es `tx2_cv`.

## Objetivo del proyecto

Proyecto de vision por computadora para trabajar sobre videos de un paquete de tubos. El flujo actual es:

1. Tomar un frame de referencia del video.
2. Seleccionar 4 puntos para generar una homografia.
3. Rectificar la zona del paquete de tubos.
4. Detectar una linea limite del paquete dentro de una ROI.
5. Generar overlays, CSV de mediciones y muestras visuales para calibracion.

## Datos principales

Video principal usado por scripts y notebook:

```text
C:\Users\luis_\OneDrive\Desktop\tx2_cv\20260407_191100_61A4_B8A44FEF1AB4\20260407_19\20260407_191100_E055.mkv
```

Otros archivos de captura:

```text
C:\Users\luis_\OneDrive\Desktop\tx2_cv\20260407_191100_61A4_B8A44FEF1AB4\20260407_19\20260407_191602_5246.mkv
C:\Users\luis_\OneDrive\Desktop\tx2_cv\20260407_191100_61A4_B8A44FEF1AB4\recording.xml
```

Frame de referencia usado por el selector/notebook:

```text
16.0 s
```

## Homografia guardada

Archivo canonico de homografia:

```text
C:\Users\luis_\OneDrive\Desktop\tx2_cv\outputs\homography_selection.json
```

Origen:

```text
C:\Users\luis_\OneDrive\Desktop\tx2_cv\20260407_191100_61A4_B8A44FEF1AB4\20260407_19\20260407_191100_E055.mkv @ 16.000s
```

Puntos fuente ordenados:

```python
[
    [1210.6868896484375, 793.9533081054688],
    [2782.1767578125, 938.3800048828125],
    [2697.136962890625, 1724.7066650390625],
    [1158.003173828125, 1624.453369140625],
]
```

Salida rectificada:

```text
1578 x 832
```

Archivos asociados:

```text
outputs\homography_selection_source.jpg
outputs\homography_selection_warp.jpg
```

## Scripts

### homography_selector.py

Selector interactivo OpenCV para escoger 4 puntos sobre una imagen o frame de video y guardar:

- `outputs\homography_selection.json`
- `outputs\homography_selection_source.jpg`
- `outputs\homography_selection_warp.jpg`

Defaults importantes:

- Video: `20260407_191100_E055.mkv`
- Segundo: `16.0`
- Output: `outputs`

Lanzador:

```text
run_homography_selector.ps1
```

### line_measure_app.py

App OpenCV para reproducir video y detectar una linea horizontal/inclinada dentro de una ROI.

Usa por default:

```text
outputs\homography_selection.json
```

Configuracion principal del detector:

```python
process_width = 1366
roi_norm_original = (0.384, 0.013, 0.659, 0.677)
roi_norm_rectified = (0.00, 0.08, 1.00, 0.62)
search_band_rectified = (0.45, 0.88)
num_sample_points = 26
max_abs_slope = 0.20
min_confidence = 0.30
max_crm_px = 4.0
pre_sobel_blur_ksize = (31, 1)
```

El detector:

- aplica homografia si existe el JSON;
- redimensiona para procesamiento;
- toma una ROI normalizada;
- convierte a gris;
- aplica CLAHE;
- aplica blur;
- calcula Sobel Y absoluto;
- muestrea columnas;
- ajusta recta con filtros de score, cluster, inliers, pendiente, confianza y CRM.

Lanzador:

```text
run_line_measure_app.ps1
```

### warp_player_app.py

Reproductor OpenCV que muestra el video rectificado.

Atencion: este archivo parece tener valores legacy/desactualizados frente a `tx2_cv`:

- `DEFAULT_VIDEO_DIR` apunta a `C:\Users\luis_\OneDrive\Desktop\tx1`
- `DEFAULT_POINTS` usa puntos `[586, 110]`, `[944, 534]`, `[662, 609]`, `[341, 165]`
- esos puntos no coinciden con `outputs\homography_selection.json`

Lanzadores:

```text
run_warp_player_app.ps1
run_warp_player_app.bat
```

Si se quiere usar este reproductor para TX2, conviene actualizarlo para cargar `outputs\homography_selection.json` o reemplazar sus puntos hardcodeados.

## Notebook

Notebook principal:

```text
tube_horizontal_limit_detection.ipynb
```

Titulo interno:

```text
Deteccion del limite inclinado del paquete de tubos
```

El notebook trabaja con:

- `VIDEO_PATH` al video `20260407_191100_E055.mkv`
- `REFERENCE_SECOND = 16.0`
- `ROI_NORM_ORIGINAL = (0.384, 0.013, 0.659, 0.677)`
- `ROI_NORM_RECTIFIED = (0.00, 0.08, 1.00, 0.62)`
- `WRITE_OVERLAY_VIDEO = True`
- `OVERLAY_VIDEO_PATH = outputs\tube_limit_line_overlay.mp4`
- `CSV_PATH = outputs\tube_limit_line.csv`
- `PREVIEW_TIMES = [14.0, 16.0, 18.0, 20.0, 22.0]`

Funciones principales del notebook:

- abrir video;
- leer frame por segundo;
- aplicar homografia si esta disponible;
- convertir ROI normalizada a pixeles;
- construir respuesta de borde con Sobel Y;
- mostrar debug;
- generar grid de previews;
- procesar video completo/parcial;
- exportar CSV y overlay MP4.

## Outputs actuales

Directorio:

```text
outputs
```

Archivos:

```text
homography_selection.json
homography_selection_source.jpg
homography_selection_warp.jpg
tube_limit_line.csv
tube_limit_line_overlay.mp4
```

CSV actual:

```text
frame_index,time_sec,slope_roi,intercept_roi,left_y_full,right_y_full,confidence,crm_px,is_valid
```

El CSV empieza en frame `360`, tiempo `12.0 s`.

## Carpetas de muestras y debug

Carpetas de imagenes generadas:

```text
sample_overlays
horizontal_samples
horizontal_samples2
slanted_samples
slanted_samples_median
rectified_band_samples
diag_2150
```

Imagenes sueltas importantes:

```text
frame_16s.jpg
frame_16s_grid.jpg
homography_quad_preview.jpg
homography_rectified_preview.jpg
band_homography_preview.jpg
band_homography_quad.jpg
debug_horizontal.jpg
debug_horizontal_segment.jpg
debug_peaks.jpg
debug_peaks_tight.jpg
debug_slanted_line.jpg
debug_slanted_line_median.jpg
```

## Peso del proyecto

Inventario al actualizar esta memoria:

```text
2705 archivos
aprox. 551 MB
```

Distribucion principal:

```text
.uv-cache-check                    ~334 MB
20260407_191100_61A4_B8A44FEF1AB4  ~154 MB
outputs                              ~7 MB
imagenes de muestras/debug          ~15 MB aprox.
```

La carpeta `.uv-cache-check` parece ser cache/dependencias de `uv` y no codigo fuente del proyecto, pero forma parte del directorio actual.

## Dependencias implicitas

Los scripts importan:

```text
opencv-python / cv2
numpy
tkinter
```

El notebook tambien usa librerias tipicas de analisis/visualizacion, como `matplotlib` y CSV/Pathlib.

No hay archivo `requirements.txt`, `pyproject.toml` ni entorno formal detectado en este directorio.

## Estado recomendado para continuar

1. Para calibrar homografia nueva: ejecutar `run_homography_selector.ps1`.
2. Para medir linea con la homografia guardada: ejecutar `run_line_measure_app.ps1`.
3. Para reproducir warp en TX2: primero actualizar `warp_player_app.py`, porque conserva defaults de `tx1`.
4. Para experimentacion y regenerar outputs: usar `tube_horizontal_limit_detection.ipynb`.

## Nota sobre TX1

Referencias a:

```text
C:\Users\luis_\OneDrive\Desktop\tx1
three_view_web_app.py
tube_start_detection_steps.ipynb
20260417_010008_BA92.mkv
```

pertenecen al contexto anterior de TX1, no al estado actual de `tx2_cv`, salvo que se copien explicitamente a este workspace.
