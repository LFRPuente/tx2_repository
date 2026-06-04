# Contexto Tecnico - AS20 VisionSystem OPC Probe

## De donde sale esta prueba

Estamos preparando el MVP de vision para TX2. La parte de CV ya calcula la linea
frontal de las piezas usando YOLO + Sobel Y dentro del ROI. Para poder comparar
o sincronizar esa medicion con planta, necesitamos leer senales del PLC por OPC
UA.

En el repo `BilletScanning` se reviso la app de OPC UA. La arquitectura existente
usa un microservicio FastAPI con `asyncua`, pero ese microservicio tampoco habla
directo al PLC. Se conecta a Kepware, y Kepware se conecta al ControlLogix.

Ruta conceptual:

```text
CV / Notebook / Backend
  -> OPC UA endpoint de Kepware
  -> KEPServerEX
  -> ControlLogix
  -> AS20 > VisionSystem
```

## Server y endpoint

La captura muestra KEPServerEX conectado al runtime en:

```text
BRJTXQMOSTX2OPC.BARNSTXPROD.LOCAL
```

El endpoint usado por la app de TX2 en `BilletScanning` es:

```text
opc.tcp://10.14.6.48:49320
```

Si el DNS resuelve dentro de la VPN, tambien puede probarse:

```text
opc.tcp://BRJTXQMOSTX2OPC.BARNSTXPROD.LOCAL:49320
```

## Tags que queremos probar

En Kepware la ruta visible es:

```text
Connectivity > ControlLogix > AS20 > VisionSystem
```

Los tags mostrados son:

```text
MeasureLength
VisionWD
```

Con la convencion usada por Kepware en el proyecto, los NodeId esperados son:

```text
ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength
ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD
```

## Como interpretar VisionWD

`VisionWD` es el watchdog. Sirve para validar vida/comunicacion del PLC o del
sistema de vision. En la captura aparece con scan rate de 50 ms y se comento que
el PLC lo actualiza alrededor de cada 69 ms.

En el notebook hay una celda que lo lee cada 70 ms:

```python
FAST_WATCH_SECONDS = 0.07
SAMPLES = 100
```

Si el valor cambia entre muestras, la comunicacion esta viva. Si no cambia,
puede haber problema de conexion, de path, de permisos, de scan rate, o el
watchdog puede estar detenido.

## Como interpretar MeasureLength

`MeasureLength` es el tag que suena asociado a la medicion de longitud, pero en
la captura aparece como `Boolean`. Eso es importante.

Puede significar una de estas cosas:

- Es un bit de solicitud/estado de medicion.
- Es un trigger que indica que se debe medir o que la medicion esta lista.
- El valor numerico real de longitud esta en otro tag.
- El tipo de dato en Kepware necesita revisarse.

Por eso el notebook imprime:

```text
value
source_timestamp
server_timestamp
status
```

Para sincronizar contra video, el mejor caso seria tener un tag numerico de
longitud y/o un timestamp/evento de medicion. `VisionWD` solo confirma vida; no
marca necesariamente el instante fisico de la medicion.

