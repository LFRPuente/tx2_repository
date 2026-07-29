# PostgreSQL Setup On The TX2 Windows Server

This guide prepares PostgreSQL on the same Windows server that runs the TX2
Live MVP. PostgreSQL is the only database for the application; SQLite is not
part of the design.

This procedure creates the PostgreSQL service, an application database, and a
restricted login. It does not create the application tables yet. Tables and
migrations will be added when database persistence is integrated into the app.

## 1. Install PostgreSQL

1. Sign in to the TX2 server with a Windows administrator account.
2. Download a supported 64-bit PostgreSQL release approved by NSTX IT from the
   [official Windows installer page](https://www.postgresql.org/download/windows/).
3. Install PostgreSQL Server and Command Line Tools. pgAdmin is optional.
4. Keep TCP port `5432` unless it conflicts with an approved server service.
5. Set a strong password for the `postgres` administrator and store it in the
   approved password manager. Do not put it in this repository.
6. Keep the database data directory outside the Git repository.

Open a new elevated PowerShell terminal and locate the installed tools:

```powershell
$Psql = (Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName
$PgDump = Join-Path (Split-Path $Psql) "pg_dump.exe"

& $Psql --version
$PgService = Get-Service | Where-Object Name -Like "postgresql*" |
  Select-Object -First 1
Set-Service -Name $PgService.Name -StartupType Automatic
Start-Service $PgService.Name
Get-Service $PgService.Name
```

The PostgreSQL Windows service must report `Running` and use automatic startup.

## 2. Create The Database And Application Login

Connect with the administrator account:

```powershell
& $Psql -h 127.0.0.1 -p 5432 -U postgres -d postgres
```

Run the following commands in `psql`. The `\password` command prompts securely
and avoids placing the application password in shell history:

```sql
SET password_encryption = 'scram-sha-256';

CREATE ROLE tx2_vision_app
    WITH LOGIN
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOREPLICATION;

\password tx2_vision_app

CREATE DATABASE tx2_vision
    WITH OWNER = tx2_vision_app
    ENCODING = 'UTF8';

\connect tx2_vision

ALTER DATABASE tx2_vision SET timezone TO 'UTC';
ALTER ROLE tx2_vision_app SET timezone TO 'UTC';

REVOKE CONNECT ON DATABASE tx2_vision FROM PUBLIC;
GRANT CONNECT ON DATABASE tx2_vision TO tx2_vision_app;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE, CREATE ON SCHEMA public TO tx2_vision_app;

\quit
```

Use the `tx2_vision_app` login from the application. Never run the application
as the `postgres` administrator.

## 3. Restrict PostgreSQL To The Local Server

Because the MVP and PostgreSQL run on the same machine, PostgreSQL should only
listen on the loopback interface. Find the active configuration files:

```powershell
& $Psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -Atc "SHOW config_file"
& $Psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -Atc "SHOW hba_file"
```

In `postgresql.conf`, verify:

```ini
listen_addresses = 'localhost'
port = 5432
password_encryption = 'scram-sha-256'
```

Place these entries before broader matching entries in `pg_hba.conf`:

```text
# TYPE  DATABASE    USER            ADDRESS         METHOD
host    tx2_vision  tx2_vision_app  127.0.0.1/32    scram-sha-256
host    tx2_vision  tx2_vision_app  ::1/128         scram-sha-256
```

Do not add `0.0.0.0/0`, do not use `trust`, and do not create an inbound Windows
Firewall rule for port `5432`. Restart the PostgreSQL service after changing
`postgresql.conf`:

```powershell
$PgService = Get-Service | Where-Object Name -Like "postgresql*" |
  Select-Object -First 1
Restart-Service $PgService.Name
Get-Service $PgService.Name
```

## 4. Configure The Application Account

Perform this section while signed in as the same dedicated Windows account that
will run the MVP or its scheduled task. Store the database password in that
account's PostgreSQL password file, not in source control:

```powershell
$PgPassDirectory = Join-Path $env:APPDATA "postgresql"
$PgPassFile = Join-Path $PgPassDirectory "pgpass.conf"
$WindowsAccount = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

New-Item -ItemType Directory -Force $PgPassDirectory | Out-Null
Set-Content -Path $PgPassFile -Value "127.0.0.1:5432:tx2_vision:tx2_vision_app:REPLACE_WITH_APP_PASSWORD"
& icacls $PgPassFile /inheritance:r /grant:r "${WindowsAccount}:M"
```

Replace the placeholder before testing. If the password contains `:` or `\`,
escape that character with `\` in `pgpass.conf`.

Reserve the following connection string for the database integration. It omits
the password because PostgreSQL reads it from `pgpass.conf`:

```powershell
$env:TX2_POSTGRES_DSN = "host=127.0.0.1 port=5432 dbname=tx2_vision user=tx2_vision_app connect_timeout=5"

# Optional persistence for the current Windows account. New processes see it.
[Environment]::SetEnvironmentVariable(
  "TX2_POSTGRES_DSN",
  $env:TX2_POSTGRES_DSN,
  "User"
)
```

The Live MVP reads `TX2_POSTGRES_DSN` at startup. The password remains outside
the DSN and is resolved through `pgpass.conf`.

## 5. Verify The Installation

Test TCP connectivity and then connect as the application login:

```powershell
Test-NetConnection 127.0.0.1 -Port 5432

& $Psql `
  -h 127.0.0.1 `
  -p 5432 `
  -U tx2_vision_app `
  -d tx2_vision `
  -c "SELECT current_database(), current_user, current_setting('TimeZone'), now();"
```

Expected results:

- `TcpTestSucceeded` is `True` locally.
- The database is `tx2_vision`.
- The user is `tx2_vision_app`.
- The timezone is `UTC`.
- No password is requested after `pgpass.conf` is configured correctly.

Since PostgreSQL is restricted to `localhost`, a connection test to the TX2
server's network IP on port `5432` should not succeed.

## 6. Prepare Backups

Choose an approved backup directory outside the repository. The following
command creates a PostgreSQL custom-format backup and can later be placed in a
Windows scheduled task:

```powershell
$BackupDirectory = "D:\TX2Backups\PostgreSQL"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force $BackupDirectory | Out-Null

& $PgDump `
  -h 127.0.0.1 `
  -p 5432 `
  -U tx2_vision_app `
  -d tx2_vision `
  --format=custom `
  --file (Join-Path $BackupDirectory "tx2_vision_$Timestamp.dump")
```

Define retention, backup frequency, and a restore test with NSTX IT before the
database becomes the production system of record.

## Ready For Application Integration

The server is ready for the next development step when all of these are true:

- PostgreSQL starts automatically and remains `Running`.
- `tx2_vision_app` can connect to `tx2_vision` locally.
- PostgreSQL does not accept network connections on the server's LAN address.
- The database and application roles use UTC.
- Credentials and backups are outside the Git repository.
- The `TX2_POSTGRES_DSN` variable is available to the Windows account that will
  run the MVP.

After that, apply the versioned application schema and validate the existing
sidecars from the repository root:

```powershell
python tools\apply_postgres_migrations.py

python tools\migrate_live_sidecars_to_postgres.py `
  --dry-run `
  --output-dir .\outputs `
  --model .\runs\detect\runs_tx2\yolo11n_pieces_v1\weights\best.pt
```

Remove `--dry-run` only after reviewing the validation result. The import is
idempotent and one PLC event can reference multiple independently measured
pieces. Production Live startup intentionally fails until the database is
reachable and all required tables exist.
