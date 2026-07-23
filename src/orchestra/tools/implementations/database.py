"""Read-only Postgres query tool. Rejects anything that isn't a SELECT."""

import os
import re

_WRITE_KEYWORDS = re.compile(
    r"^\s*(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b",
    re.IGNORECASE,
)


def _connection_string() -> str:
    return (
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ.get('POSTGRES_DB', 'orchestra')} "
        f"user={os.environ.get('POSTGRES_USER', 'orchestra')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'orchestra')}"
    )


def database_query(sql: str, params: list | None = None) -> dict:
    stripped = sql.strip()
    if _WRITE_KEYWORDS.match(stripped) or not stripped.lower().startswith("select"):
        raise ValueError("database_query only permits read-only SELECT statements.")

    import psycopg

    with psycopg.connect(_connection_string()) as conn, conn.cursor() as cur:
        cur.execute(stripped, params or [])
        columns = [col.name for col in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    return {"rows": rows, "row_count": len(rows)}
