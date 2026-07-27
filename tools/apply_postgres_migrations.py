from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIGRATIONS = ROOT / "db" / "migrations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply versioned TX2 PostgreSQL migrations.")
    parser.add_argument("--dsn", default=os.environ.get("TX2_POSTGRES_DSN", ""))
    parser.add_argument("--migrations-dir", type=Path, default=DEFAULT_MIGRATIONS)
    return parser.parse_args()


def migration_statements(sql: str) -> list[str]:
    statements = []
    for statement in sql.split(";"):
        value = statement.strip()
        if not value or value.upper() in {"BEGIN", "COMMIT"}:
            continue
        statements.append(value)
    return statements


def main() -> int:
    args = parse_args()
    if not args.dsn.strip():
        print("ERROR: TX2_POSTGRES_DSN is required.", file=sys.stderr)
        return 2
    try:
        import psycopg
    except ImportError:
        print("ERROR: PostgreSQL support is not installed. Install requirements.txt.", file=sys.stderr)
        return 2

    migrations = sorted(args.migrations_dir.glob("*.sql"))
    if not migrations:
        print(f"ERROR: No migrations found in {args.migrations_dir}", file=sys.stderr)
        return 2

    try:
        with psycopg.connect(args.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migration (
                        filename text PRIMARY KEY,
                        sha256 text NOT NULL,
                        applied_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            connection.commit()

            for path in migrations:
                sql = path.read_text(encoding="utf-8")
                digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT sha256 FROM schema_migration WHERE filename = %s",
                        (path.name,),
                    )
                    row = cursor.fetchone()
                if row:
                    if row[0] != digest:
                        raise RuntimeError(
                            f"Applied migration {path.name} no longer matches its recorded checksum"
                        )
                    print(f"SKIP {path.name}")
                    continue

                with connection.transaction():
                    with connection.cursor() as cursor:
                        for statement in migration_statements(sql):
                            cursor.execute(statement)
                        cursor.execute(
                            "INSERT INTO schema_migration (filename, sha256) VALUES (%s, %s)",
                            (path.name, digest),
                        )
                print(f"APPLIED {path.name}")
    except Exception as exc:
        print(f"ERROR: PostgreSQL migration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
