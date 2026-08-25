# TX2 Live MVP: plan de despliegue en Windows con IIS

Estado: backend productivo listo; infraestructura IIS/DNS pendiente de TI
Fecha: 2026-08-24  
Rama revisada: `main`

## 1. Objetivo

Publicar el TX2 Live MVP dentro de la red de Nucor desde el servidor Windows,
sin exponer directamente Flask ni sus puertos internos. La primera etapa debe
conservar el procesamiento que ya funciona: camara AXIS, PLC OPC UA, YOLO,
Sobel, grabacion, History y correcciones del operador.

Este documento registra las decisiones tomadas y el orden recomendado para
seguir. No es todavia una receta de instalacion terminada: varios puntos
requieren cambios de codigo, informacion de TI y validacion en el servidor.

## 2. Decisiones actuales

| Tema | Decision |
|---|---|
| Sistema operativo | Windows |
| Alcance de red | Solo red interna de Nucor |
| Puerta web | IIS |
| Nginx | No se usara en este despliegue Windows |
| Aplicacion actual | Flask en `live_mvp_app.py` |
| Base de datos actual | SQLite |
| Archivo SQLite | `outputs/tx2_live_mvp.sqlite3` |
| Base de datos futura | Microsoft SQL Server en otro servidor |
| PostgreSQL | Ya no es el destino acordado |
| Django | Posible evolucion futura; no existe en el MVP actual |
| Docker | Evaluacion posterior; no es requisito para la primera publicacion |
| Procesamiento de vision | Un solo proceso propietario de camara, PLC y GPU |

Las secciones antiguas de `README.md`, `LIVE_MVP_INTEGRATION_README.md` y
`docs/postgresql_server_setup.md` que declaran PostgreSQL como destino de
produccion quedan superadas por esta decision. No se debe instalar PostgreSQL
en el servidor TX2 siguiendo esas instrucciones.

## 3. Estado real del repositorio

Actualmente el Live MVP:

- usa Flask;
- conserva `run_live_mvp_app.ps1` para desarrollo y diagnostico;
- arranca en produccion con `run_live_mvp_production.ps1` y Waitress;
- escucha solamente en `127.0.0.1:8767`;
- posee un runtime explicito que inicia y detiene camara, procesador, recorder,
  PLC, reconciliador y base de datos dentro de un solo proceso;
- guarda el historial temporal en SQLite;
- conserva videos e imagenes en disco y solo guarda sus rutas y metadatos en
  la base de datos;
- incluye `deployment/iis/web.config` y `deployment/configure_iis.ps1` para el
  reverse proxy IIS;
- expone `/api/health` sin rutas, secretos ni datos sensibles;
- no tiene `Dockerfile` ni archivo de Docker Compose;
- no tiene autenticacion corporativa implementada en Flask;
- no tiene un adaptador para SQL Server.

El codigo ya contiene persistencia para SQLite y PostgreSQL. PostgreSQL no se
debe confundir con SQL Server: el adaptador actual `DatabaseRepository` usa
`psycopg` y no puede conectarse a Microsoft SQL Server cambiando solamente una
cadena de conexion.

Configuracion live importante en el launcher actual:

```text
Web local:       http://127.0.0.1:8767
URL objetivo:    https://tx2-measurement.barnstxprod.local
Camara default:  10.14.115.241
Resolucion:      2880x2160
YOLO/Sobel:      10 FPS objetivo
Live/Raw:        10 FPS (maximo a 2880x2160 aceptado por la camara)
PLC OPC UA:      opc.tcp://10.14.6.48:49320
Trigger:         ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength
Watchdog:        ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD
Grabacion:       8 segundos por evento
Medicion:        frame inmediatamente anterior/alineado a la senal PLC
```

La camara fue consultada por VAPIX y probada por RTSP el 2026-08-24. El capture
mode vigente admite `2880x2160 @ 10 FPS`; solicitudes de 11, 12, 15, 20, 25 o
30 FPS devuelven `400 Bad Request`. No se cambia el capture mode porque puede
alterar el encuadre y obligaria a recalibrar homografia y mediciones.

## 4. Arquitectura recomendada para la primera etapa

```text
PC dentro de Nucor
        |
        | HTTPS 443
        v
+--------------------------+
| IIS en Windows           |
| - certificado interno    |
| - autenticacion Windows  |
| - restriccion de red     |
| - logs de acceso         |
+------------+-------------+
             |
             | proxy local
             v
+--------------------------+
| Servidor WSGI productivo |
| 127.0.0.1:8767           |
+------------+-------------+
             |
             v
+--------------------------+
| Flask TX2 Live MVP       |
| - API y UI               |
| - History                |
| - runtime de vision      |
+------+----------+--------+
       |          |
       v          v
 SQLite local   Videos/imagenes en disco
       |
       +---- futuro: migracion controlada a SQL Server externo
```

IIS y Flask no cumplen la misma funcion:

- IIS es la entrada de red, termina HTTPS, autentica y reenvia solicitudes.
- El servidor WSGI mantiene Flask funcionando de forma productiva.
- Flask contiene la aplicacion y sus endpoints.
- El runtime de vision procesa la camara, PLC, YOLO y Sobel una sola vez.

IIS debe ser el unico servicio visible desde la red. El puerto `8767` debe
seguir enlazado a `127.0.0.1` y no debe abrirse en Windows Firewall.

## 5. Backend productivo implementado

No se publica `app.run` como servidor de produccion. La Fase 1 quedo
implementada con:

1. `LiveMvpRuntime` como propietario del ciclo de vida de camara, procesador,
   recorder, PLC, reconciliador y base de datos.
2. Inicio unico y apagado ordenado, incluyendo espera de grabaciones activas.
3. `run_live_mvp_production.py` como punto de entrada WSGI Waitress.
4. `run_live_mvp_production.ps1` como launcher productivo.
5. Waitress fijado en `requirements.txt`.
6. Binding exclusivo en `127.0.0.1:8767` con ocho threads.
7. Endpoint `/api/health` que solo informa checks booleanos.
8. `run_live_mvp_app.ps1` conservado para desarrollo y diagnostico.

Comando productivo:

```powershell
.\run_live_mvp_production.ps1 -Device auto
```

No se deben configurar varios workers o replicas del Live MVP actual. Dos
procesos abririan dos conexiones a la camara y al PLC, cargarian YOLO dos veces
y podrian crear eventos duplicados.

## 6. Servicio de Windows

Waitress no mantiene la aplicacion encendida despues de reiniciar Windows por
si solo. El comando de produccion debe quedar registrado como un servicio de
Windows con la herramienta aprobada por TI.

El servicio debe:

- arrancar automaticamente;
- ejecutarse con una cuenta de servicio dedicada;
- reiniciarse ante una falla no controlada;
- usar el directorio correcto del repositorio como working directory;
- cargar las credenciales AXIS desde un mecanismo protegido;
- tener permisos de lectura sobre modelo y calibraciones;
- tener permisos de escritura sobre `outputs/` y logs;
- no tener permisos administrativos innecesarios;
- detenerse de forma ordenada antes de actualizar el codigo.

No se deben guardar `AXIS_USER`, `AXIS_PASSWORD`, cadenas de base de datos ni
certificados privados dentro de Git.

## 7. Configuracion IIS propuesta

Componentes que TI debe aprobar e instalar:

- rol Web Server (IIS);
- HTTPS;
- Windows Authentication;
- URL Rewrite;
- Application Request Routing (ARR) para reverse proxy.

Documentacion de referencia de Microsoft:

- [Reverse proxy con URL Rewrite y ARR](https://learn.microsoft.com/en-us/iis/extensions/url-rewrite-module/reverse-proxy-with-url-rewrite-v2-and-application-request-routing)
- [Windows Authentication en IIS](https://learn.microsoft.com/en-us/iis/configuration/system.webServer/security/authentication/windowsAuthentication/)

Configuracion objetivo:

```text
URL publica interna: https://tx2-measurement.barnstxprod.local/
DNS interno:         tx2-measurement.barnstxprod.local -> 10.14.6.84
Binding IIS:         443 con certificado de la CA interna
Backend local:       http://127.0.0.1:8767/
Anonymous auth:      deshabilitada
Windows auth:        habilitada
Firewall:            443 solo desde redes aprobadas de Nucor
```

Configuracion versionada:

```text
deployment/iis/web.config
deployment/configure_iis.ps1
```

Cuando TI haya instalado los componentes y entregado el certificado y las
redes VPN autorizadas, ejecutar en PowerShell elevado:

```powershell
.\deployment\configure_iis.ps1 `
  -CertificateThumbprint "<thumbprint-CA-interna>" `
  -AllowedRemoteAddress "<subred-VPN-aprobada>"
```

Auditoria del host del 2026-08-24:

- IP del servidor: `10.14.6.84`;
- el nombre objetivo aun no resuelve en DNS;
- IIS, WAS, URL Rewrite y ARR no estan instalados;
- no hay certificado de maquina para el nombre objetivo;
- la sesion actual no tiene elevacion administrativa.

Por esos cuatro bloqueos de infraestructura, el script queda listo e
idempotente pero no se puede activar el HTTPS corporativo desde esta sesion.
No se debe sustituir esta configuracion con un hosts file ni exponer Waitress
en `0.0.0.0`.

Antes de activar Windows Authentication se debe obtener el grupo de Active
Directory autorizado. La aplicacion tambien debe dejar de confiar en el nombre
de operador enviado libremente por el navegador. IIS debe transmitir la
identidad autenticada mediante un header controlado, y Flask debe aceptar ese
header solamente cuando la solicitud proviene del proxy local.

Se deben validar especificamente:

- reproduccion de MP4 y solicitudes HTTP Range;
- polling de `/api/live/frame`;
- respuesta de `/api/live/status`;
- carga de `/history` y sus assets;
- tiempos de espera de ARR durante video;
- headers de proxy y direccion IP original;
- tamano de respuestas y consumo de red con varios usuarios.

## 8. SQLite durante la primera etapa

SQLite es la base activa por ahora:

```text
outputs/tx2_live_mvp.sqlite3
```

La implementacion actual habilita foreign keys, busy timeout y modo WAL. Para
que esta etapa sea confiable:

- ejecutar una sola instancia del Live MVP;
- almacenar la base en disco local estable, no dentro de Git;
- dar acceso solo a la cuenta del servicio;
- monitorear espacio libre del disco;
- respaldar tambien los videos y sidecars relacionados;
- no copiar un archivo SQLite activo como si fuera un archivo cualquiera;
- hacer el respaldo con el servicio detenido o usando la API de backup de
  SQLite;
- probar una restauracion antes de considerar completo el despliegue.

SQLite es suficiente para el piloto y para pocos usuarios porque la escritura
principal ocurre por eventos del PLC. No se considera la base final cuando se
requiera integracion corporativa, reporteo o acceso concurrente mayor.

## 9. Migracion futura a SQL Server

SQL Server estara en otro servidor de Nucor. Por lo tanto:

- SQL Server no se instalara en el servidor TX2;
- SQL Server no formara parte de Docker;
- IIS no se conectara a SQL Server;
- la aplicacion sera el unico componente que consulte y escriba la base;
- videos e imagenes permaneceran en almacenamiento de archivos;
- SQL Server guardara eventos, piezas, mediciones, revisiones, timestamps,
  hashes y rutas de assets.

Informacion que se debe obtener de TI:

- nombre DNS del servidor e instancia;
- puerto TCP real;
- nombre de la base;
- metodo de autenticacion permitido;
- cuenta de servicio o login de aplicacion;
- certificado y requisitos de cifrado;
- reglas de firewall;
- politica de respaldos y retencion;
- ambiente de desarrollo o pruebas separado de produccion.

Trabajo de software requerido:

1. Definir un esquema SQL Server equivalente al esquema SQLite vigente.
2. Implementar un repositorio SQL Server con el mismo contrato que usa el Live
   MVP para eventos, snapshots, piezas, assets y revisiones.
3. Instalar Microsoft ODBC Driver 18 en el host o imagen que ejecute Python.
4. Elegir `pyodbc`/SQLAlchemy si Flask continua, o `mssql-django` si se migra a
   Django.
5. Agregar migraciones versionadas para SQL Server.
6. Crear una herramienta idempotente `SQLite -> SQL Server`.
7. Preservar IDs, timestamps UTC, medidas automaticas, correcciones del
   operador y numero de revision.
8. Validar conteos y hashes antes del cambio definitivo.
9. Hacer el corte con la aplicacion detenida para evitar escrituras durante la
   migracion final.
10. Conservar SQLite como respaldo de solo lectura hasta aprobar SQL Server.

Cuando se configure SQL Server, un error de conexion no debe cambiar
silenciosamente a SQLite. La aplicacion debe reportar el error y conservar los
sidecars pendientes para reconciliacion, siguiendo el principio de seguridad
ya usado por el backend PostgreSQL anterior.

Entidades que deben preservarse:

```text
vision_configuration
measurement_event
analysis_snapshot
piece_measurement
piece_measurement_revision
event_asset
```

No se deben guardar MP4, JPEG ni base64 dentro de SQL Server.

## 10. Django: alcance futuro

Django puede ser una mejor capa web definitiva cuando se necesiten usuarios,
roles, permisos, ORM, migraciones, panel administrativo y auditoria. Sin
embargo, cambiar Flask por Django no mejora YOLO, Sobel ni el acceso al PLC.

La migracion correcta seria incremental:

1. Separar primero el runtime de vision de Flask.
2. Mantener un solo worker de vision.
3. Permitir que la capa web consulte resultados sin cargar otra copia de YOLO.
4. Migrar History, usuarios y API a Django en una entrega independiente.
5. Conectar Django a SQL Server cuando el servidor y el esquema esten listos.

No se recomienda mezclar en una sola puesta en produccion IIS, Docker, Django y
SQL Server. Cada cambio debe poder probarse y revertirse por separado.

## 11. Docker en Windows

Docker sigue siendo util para empaquetar versiones de Python, CUDA, YOLO y
OpenCV, pero todavia no esta implementado en este repositorio. En el servidor
Windows se debe tomar la decision solo despues de probar:

- compatibilidad de Windows y WSL2;
- acceso NVIDIA desde un contenedor;
- `nvidia-smi` dentro del contenedor;
- acceso RTSP a la camara;
- acceso OPC UA al PLC;
- escritura persistente de SQLite, videos y calibraciones;
- arranque automatico sin una sesion interactiva;
- licencia y soporte aprobados por TI.

Si se aprueba Docker, IIS permanecera en el host Windows:

```text
Red Nucor -> IIS host -> 127.0.0.1:8767 -> contenedor de aplicacion
```

Los puertos se publicaran solo en loopback y los siguientes directorios seran
volumenes persistentes:

```text
outputs/
runs/detect/.../weights/
configuraciones de homografia y medicion
logs/
```

La primera publicacion no debe bloquearse por Docker. El orden recomendado es
estabilizar IIS + servicio Windows + SQLite y despues evaluar el contenedor con
una prueba separada de GPU.

## 12. Fases recomendadas

### Fase 0: respaldar y establecer linea base

- [ ] Respaldar `outputs/`, SQLite, calibraciones y modelo activo.
- [ ] Registrar commit y SHA-256 del modelo desplegado.
- [ ] Ejecutar todas las pruebas existentes.
- [ ] Validar camara, PLC, una grabacion y una correccion en History.

### Fase 1: servidor Flask productivo

- [x] Separar el ciclo de vida del runtime de vision.
- [x] Agregar punto de entrada WSGI productivo.
- [x] Agregar Waitress y launcher de produccion.
- [x] Garantizar una sola instancia de vision.
- [x] Probar inicio, error y apagado ordenado.

### Fase 2: servicio Windows

- [ ] Crear cuenta de servicio dedicada.
- [ ] Configurar secretos fuera de Git.
- [ ] Registrar el servicio con inicio automatico.
- [ ] Definir logs, rotacion y recuperacion ante falla.
- [ ] Probar reinicio completo del servidor.

### Fase 3: IIS interno

- [x] Versionar `web.config` y script idempotente de configuracion IIS.
- [x] Reservar `tx2-measurement.barnstxprod.local` como nombre objetivo.
- [ ] Instalar IIS, URL Rewrite, ARR y Windows Authentication.
- [ ] Crear DNS interno y certificado aprobado.
- [ ] Configurar reverse proxy hacia `127.0.0.1:8767`.
- [ ] Restringir firewall y grupo de Active Directory.
- [ ] Integrar la identidad autenticada con las revisiones del operador.
- [ ] Probar video, History, API y varios usuarios.

### Fase 4: operacion temporal con SQLite

- [ ] Automatizar backup consistente de SQLite y assets.
- [ ] Probar restauracion.
- [ ] Monitorear almacenamiento, memoria, GPU y estado PLC/camara.
- [ ] Documentar actualizacion y rollback del modelo YOLO.

### Fase 5: SQL Server externo

- [ ] Recibir datos de conexion y seguridad de TI.
- [ ] Crear esquema y repositorio SQL Server.
- [ ] Crear migrador idempotente desde SQLite.
- [ ] Validar en una base de pruebas.
- [ ] Ejecutar corte controlado y conservar respaldo SQLite.

### Fase 6: decisiones posteriores

- [ ] Evaluar Docker con una prueba real de GPU en Windows.
- [ ] Evaluar separacion definitiva del vision worker.
- [ ] Evaluar Django para usuarios, permisos, admin y API.
- [ ] Evaluar transporte de video mas eficiente si aumenta la concurrencia.

## 13. Criterios de aceptacion para publicar en Nucor

- [ ] La URL HTTPS funciona desde una PC autorizada de Nucor.
- [ ] Una PC no autorizada no puede entrar.
- [ ] El puerto `8767` no es accesible remotamente.
- [ ] IIS registra usuario, hora, ruta y resultado HTTP.
- [ ] Solo existe un proceso propietario de camara, PLC y YOLO.
- [ ] `/api/live/status` informa camara, PLC, procesador y base saludables.
- [ ] La imagen live conserva la resolucion y mediciones esperadas.
- [ ] El trigger PLC crea exactamente un evento y su clip de 8 segundos.
- [ ] History reproduce video usando HTTP Range.
- [ ] La correccion conserva la medida automatica y crea revision de auditoria.
- [ ] La identidad de la revision proviene de IIS, no de texto libre.
- [ ] SQLite y los assets pueden respaldarse y restaurarse.
- [ ] Reiniciar Windows recupera el servicio automaticamente.
- [ ] Una prueba de usuarios concurrentes no duplica inferencia ni eventos.

## 14. Datos pendientes de TI

- version exacta de Windows Server;
- nombre DNS interno que recibira la aplicacion;
- certificado de la CA interna;
- grupo de Active Directory autorizado;
- cuenta de servicio para el Live MVP;
- ubicacion definitiva de videos, backups y logs;
- politica de firewall entre usuarios, TX2, camara y PLC;
- disponibilidad y soporte de Docker/WSL2 en ese servidor;
- datos del SQL Server futuro y metodo de autenticacion.

## 15. Acciones que no se deben realizar

- No exponer `app.run` ni el puerto `8767` a la red.
- No instalar Nginx junto con IIS sin una necesidad nueva y documentada.
- No instalar PostgreSQL siguiendo la documentacion anterior.
- No guardar secretos en el repositorio.
- No ejecutar dos procesos de vision para atender mas usuarios.
- No guardar videos dentro de SQLite o SQL Server.
- No borrar SQLite despues de migrar hasta validar conteos y auditoria.
- No combinar en una sola entrega la publicacion IIS, Django, Docker y SQL
  Server.

## 16. Proximo paso operativo

TI debe crear el registro DNS, emitir el certificado, instalar IIS/ARR/URL
Rewrite/Windows Authentication y entregar la cuenta de servicio y subred VPN
autorizada. Con esos datos se ejecuta `deployment/configure_iis.ps1`, se
registra el launcher Waitress como servicio Windows y se valida HTTPS desde una
segunda computadora conectada a la VPN. Ninguno de esos pasos requiere cambiar
el algoritmo de vision, la resolucion, la homografia ni el modelo YOLO.
