# AS20 VisionSystem OPC Probe Bundle

Portable bundle para probar, desde una maquina dentro de la VPN, los tags de
Kepware/OPC UA del sistema de vision.

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

