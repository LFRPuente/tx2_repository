# AS20 VisionSystem OPC Probe Bundle

Portable bundle para probar, desde una maquina dentro de la VPN, los tags de
Kepware/OPC UA del sistema de vision.

## Contexto rapido

Este bundle nace de la revision de la app `BilletScanning` y de la captura de
Kepware en el server `BRJTXQMOSTX2OPC.BARNSTXPROD.LOCAL`.

La app existente no se conecta directo al PLC. El camino es:

```text
Python / Notebook -> OPC UA -> Kepware -> ControlLogix PLC
```

Para esta prueba no estamos leyendo el grupo `TX2MH > Scanning`. Estamos
leyendo el grupo que viste en Kepware:

```text
Connectivity > ControlLogix > AS20 > VisionSystem
```

El objetivo es validar dos cosas:

- Que `VisionWD` cambie rapido para confirmar que el PLC/sistema de vision esta
  vivo.
- Que `MeasureLength` exista y podamos leer su valor/timestamps OPC UA para
  evaluar si sirve como disparo o medicion para sincronizar con el video.

Mas detalle en `CONTEXT.md`.

## Endpoint

```text
opc.tcp://10.14.6.48:49320
```

Alternativa si DNS resuelve dentro de la VPN:

```text
opc.tcp://BRJTXQMOSTX2OPC.BARNSTXPROD.LOCAL:49320
```

## Tags configurados

```text
ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength
ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD
```

`VisionWD` sirve para validar que el PLC/sistema de vision esta vivo. El
notebook trae una celda que lo lee cada 70 ms, cerca del ritmo de actualizacion
que viste en planta.

`MeasureLength` se lee directo del path `AS20 > VisionSystem`. En la captura de
Kepware aparece como Boolean, asi que conviene confirmar si ese tag es el valor
real de longitud o solo un bit/trigger de medicion.

## Como correrlo

Desde PowerShell:

```powershell
cd C:\ruta\al\bundle\as20_vision_opc_probe
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m notebook plc_timestamp_probe.ipynb
```

Tambien puedes ejecutar:

```powershell
.\open_notebook.ps1
```

## Monitoreo sincronizado con VisionWD

Para monitorear usando `VisionWD` como tick, corre:

```powershell
python watchdog_sync_probe.py --seconds 120 --poll-seconds 0.069
```

Esto no escribe al PLC. Solo lee `VisionWD`, registra una fila cuando cambia y
lee `MeasureLength` en ese mismo tick observado. La salida tambien se guarda en:

```text
watchdog_sync_log.csv
```

Si necesitas capturar menos saltos de watchdog, usa un poll mas rapido pero
todavia razonable:

```powershell
python watchdog_sync_probe.py --seconds 120 --poll-seconds 0.035
```

## Archivos

```text
plc_timestamp_probe.ipynb  Notebook principal
watchdog_sync_probe.py      Monitor anclado a cambios de VisionWD
requirements.txt           Dependencias minimas
open_notebook.ps1          Crea venv, instala dependencias y abre Jupyter
CONTEXT.md                 Contexto tecnico de la prueba
```
