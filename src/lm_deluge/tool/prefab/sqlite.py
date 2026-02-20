from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from .. import Tool

SQLITE_DESCRIPTION = """Interact with a SQLite database using progressive disclosure.

Use this tool to:
- inspect available tables/views and basic shape of the database
- inspect schema metadata for a specific table
- run SQL queries with bounded result sets

This is optimized for LLM workflows where you want schema discovery first, then
targeted queries, instead of dumping full datasets into context.
"""

SqliteCommand = Literal["query", "list_tables", "describe_table"]
QueryOutputFormat = Literal["json", "yaml", "csv", "tsv"]

ALL_COMMANDS: tuple[SqliteCommand, ...] = (
    "query",
    "list_tables",
    "describe_table",
)


@dataclass(frozen=True)
class InferredColumn:
    name: str
    sqlite_type: Literal["INTEGER", "REAL", "TEXT", "JSONB"]


class SqliteParams(BaseModel):
    """Schema describing SQLite tool calls."""

    command: SqliteCommand = Field(
        description="Operation to run (query, list_tables, describe_table)."
    )
    sql: Optional[str] = Field(
        default=None,
        description="SQL query to execute when command='query'.",
    )
    query_params: list[Any] | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional positional list or named dict of bound parameters for SQL "
            "placeholders."
        ),
    )
    output_format: QueryOutputFormat = Field(
        default="json",
        description="Output format for query results (json, yaml, csv, tsv).",
    )
    max_rows: int = Field(
        default=200,
        ge=1,
        le=10_000,
        description="Maximum number of rows returned for command='query'.",
    )
    allow_write: bool = Field(
        default=False,
        description=(
            "Allow non-read SQL statements. Disabled by default. If manager is "
            "read_only, writes are always blocked."
        ),
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Table name for command='describe_table'.",
    )
    include_views: bool = Field(
        default=True,
        description="When listing objects, include views in addition to tables.",
    )
    include_row_counts: bool = Field(
        default=False,
        description="When listing objects, include COUNT(*) for each table/view.",
    )
    sample_rows: int = Field(
        default=0,
        ge=0,
        le=100,
        description=(
            "When describing a table, optionally include up to this many sample "
            "rows."
        ),
    )


def _delete_file_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _quote_identifier(identifier: str) -> str:
    name = identifier.strip()
    if not name:
        raise ValueError("Identifier cannot be empty")
    return '"' + name.replace('"', '""') + '"'


def _normalize_result_value(value: Any) -> Any:
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return value


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


class SqliteManager:
    """SQLite prefab manager focused on schema-first, targeted querying."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        tool_name: str = "sqlite",
        read_only: bool = True,
        _delete_on_close: bool = False,
    ):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database does not exist: {self.db_path}")

        self.tool_name = tool_name
        self.read_only = read_only
        self._tool_cache: dict[tuple[str, ...], list[Tool]] = {}
        self._finalizer: weakref.finalize | None = None

        if _delete_on_close:
            self._finalizer = weakref.finalize(
                self, _delete_file_if_exists, str(self.db_path)
            )

    def close(self) -> None:
        """Release manager-owned resources (deletes temp DB from from_dicts)."""
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()

    def _connect(self, *, writable: bool = False) -> sqlite3.Connection:
        if self.read_only:
            if writable:
                raise ValueError("Cannot open writable connection in read_only mode")
            readonly_uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(readonly_uri, uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))

        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _normalize_records(
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if not records:
            raise ValueError("from_dicts requires at least one row")

        columns: list[str] = []
        seen: set[str] = set()
        normalized_rows: list[dict[str, Any]] = []

        for row_idx, row in enumerate(records):
            normalized: dict[str, Any] = {}
            for raw_key, value in row.items():
                key = str(raw_key).strip()
                if not key:
                    raise ValueError(
                        f"Row {row_idx} contains an empty column name: {raw_key!r}"
                    )
                if key in normalized:
                    raise ValueError(
                        f"Row {row_idx} contains duplicate normalized key: {key!r}"
                    )
                normalized[key] = value
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
            normalized_rows.append(normalized)

        if not columns:
            raise ValueError("from_dicts could not infer columns (all rows were empty)")

        return columns, normalized_rows

    @staticmethod
    def _infer_columns(
        rows: Sequence[dict[str, Any]],
        columns: Sequence[str],
    ) -> list[InferredColumn]:
        inferred: list[InferredColumn] = []

        for column_name in columns:
            saw_nested = False
            saw_int = False
            saw_float = False
            saw_text = False
            saw_scalar = False

            for row in rows:
                if column_name not in row:
                    continue

                value = row[column_name]
                if value is None:
                    continue

                if isinstance(value, (dict, list)):
                    saw_nested = True
                    continue

                saw_scalar = True
                if isinstance(value, bool):
                    saw_int = True
                elif isinstance(value, int):
                    saw_int = True
                elif isinstance(value, float):
                    saw_float = True
                else:
                    saw_text = True

            if saw_nested and saw_scalar:
                raise ValueError(
                    f"Column {column_name!r} mixes nested values with scalar values. "
                    "Use consistent nested values (dict/list) or scalar values."
                )

            if saw_nested:
                sqlite_type: Literal["INTEGER", "REAL", "TEXT", "JSONB"] = "JSONB"
            elif saw_text:
                sqlite_type = "TEXT"
            elif saw_float:
                sqlite_type = "REAL"
            elif saw_int:
                sqlite_type = "INTEGER"
            else:
                sqlite_type = "TEXT"

            inferred.append(InferredColumn(name=column_name, sqlite_type=sqlite_type))

        return inferred

    @staticmethod
    def _coerce_insert_value(value: Any, sqlite_type: str) -> Any:
        if value is None:
            return None

        if sqlite_type == "JSONB":
            if not isinstance(value, (dict, list)):
                raise ValueError(
                    f"Expected dict/list for JSONB column, got {type(value).__name__}"
                )
            return json.dumps(value, separators=(",", ":"))

        if sqlite_type == "INTEGER":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            raise ValueError(
                f"Expected int-compatible value for INTEGER column, "
                f"got {type(value).__name__}"
            )

        if sqlite_type == "REAL":
            if isinstance(value, bool):
                return float(int(value))
            if isinstance(value, (int, float)):
                return float(value)
            raise ValueError(
                f"Expected number for REAL column, got {type(value).__name__}"
            )

        # TEXT fallback
        if isinstance(value, memoryview):
            return value.tobytes().hex()
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).hex()
        if isinstance(value, str):
            return value
        return str(value)

    @classmethod
    def from_dicts(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        table_name: str = "data",
        db_path: str | os.PathLike[str] | None = None,
        tool_name: str = "sqlite",
        read_only: bool = True,
    ) -> "SqliteManager":
        """
        Build a SQLite DB from list-of-dicts, inferring a practical schema.

        Inference behavior:
        - Missing keys become NULL
        - None + numbers => nullable numeric column
        - int + str => TEXT
        - nested dict/list => JSON column
        - nested dict/list mixed with scalar => ValueError
        """
        columns, normalized_rows = cls._normalize_records(rows)
        inferred = cls._infer_columns(normalized_rows, columns)

        delete_on_close = False
        if db_path is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="lm_deluge_sqlite_",
                suffix=".db",
                delete=False,
            )
            db_file = Path(handle.name)
            handle.close()
            delete_on_close = True
        else:
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        quoted_table = _quote_identifier(table_name)
        column_definitions = ", ".join(
            f"{_quote_identifier(col.name)} {col.sqlite_type}" for col in inferred
        )
        insert_columns = ", ".join(_quote_identifier(col.name) for col in inferred)
        placeholders = ", ".join("?" for _ in inferred)
        insert_sql = (
            f"INSERT INTO {quoted_table} ({insert_columns}) VALUES ({placeholders})"
        )

        conn = sqlite3.connect(str(db_file))
        try:
            conn.execute(f"CREATE TABLE {quoted_table} ({column_definitions})")

            prepared_rows: list[tuple[Any, ...]] = []
            for row in normalized_rows:
                values: list[Any] = []
                for col in inferred:
                    values.append(
                        cls._coerce_insert_value(
                            row.get(col.name),
                            col.sqlite_type,
                        )
                    )
                prepared_rows.append(tuple(values))

            conn.executemany(insert_sql, prepared_rows)
            conn.commit()
        finally:
            conn.close()

        return cls(
            db_file,
            tool_name=tool_name,
            read_only=read_only,
            _delete_on_close=delete_on_close,
        )

    def _format_query_output(
        self,
        columns: list[str],
        rows: list[dict[str, Any]],
        output_format: QueryOutputFormat,
    ) -> dict[str, Any]:
        if output_format == "json":
            return {"output_format": "json", "rows": rows}

        if output_format == "yaml":
            return {
                "output_format": "yaml",
                "output": yaml.safe_dump(rows, sort_keys=False),
            }

        delimiter = "," if output_format == "csv" else "\t"
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row.get(col)) for col in columns])
        return {"output_format": output_format, "output": buffer.getvalue()}

    def _execute_query(
        self,
        sql: str,
        query_params: list[Any] | dict[str, Any] | None,
        *,
        output_format: QueryOutputFormat,
        max_rows: int,
        allow_write: bool,
    ) -> dict[str, Any]:
        sql_text = sql.strip()
        if not sql_text:
            raise ValueError("sql is required for query")

        writable = allow_write and not self.read_only
        with self._connect(writable=writable) as conn:
            if not self.read_only and not allow_write:
                conn.execute("PRAGMA query_only = ON")

            params: Sequence[Any] | Mapping[str, Any] = ()
            if query_params is not None:
                if isinstance(query_params, dict):
                    params = query_params
                elif isinstance(query_params, list):
                    params = query_params
                else:
                    raise ValueError("query_params must be a list, dict, or null")

            cursor = conn.execute(sql_text, params)

            # Non-row queries (e.g., UPDATE) return no description.
            if cursor.description is None:
                if writable:
                    conn.commit()
                return {
                    "command": "query",
                    "sql": sql_text,
                    "output_format": "json",
                    "affected_rows": cursor.rowcount,
                }

            columns = [desc[0] for desc in cursor.description]
            fetched = cursor.fetchmany(max_rows + 1)
            truncated = len(fetched) > max_rows
            if truncated:
                fetched = fetched[:max_rows]

            rows = [
                {
                    column: _normalize_result_value(value)
                    for column, value in zip(columns, tuple(row), strict=False)
                }
                for row in fetched
            ]

            formatted = self._format_query_output(columns, rows, output_format)

            return {
                "command": "query",
                "sql": sql_text,
                "columns": columns,
                "row_count": len(rows),
                "max_rows": max_rows,
                "truncated": truncated,
                **formatted,
            }

    def _list_tables(
        self,
        *,
        include_views: bool,
        include_row_counts: bool,
    ) -> dict[str, Any]:
        types = ("table", "view") if include_views else ("table",)
        placeholders = ", ".join("?" for _ in types)
        sql = (
            "SELECT name, type, sql FROM sqlite_master "
            "WHERE type IN ({types}) AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).format(types=placeholders)

        with self._connect() as conn:
            objects = conn.execute(sql, types).fetchall()
            entries: list[dict[str, Any]] = []

            for obj in objects:
                name = str(obj["name"])
                obj_type = str(obj["type"])
                create_sql = obj["sql"]
                col_rows = conn.execute(
                    f"PRAGMA table_info({_quote_identifier(name)})"
                ).fetchall()
                column_names = [str(col["name"]) for col in col_rows]

                entry: dict[str, Any] = {
                    "name": name,
                    "type": obj_type,
                    "columns": column_names,
                    "column_count": len(column_names),
                    "create_sql": create_sql,
                }

                if include_row_counts:
                    try:
                        row_count = conn.execute(
                            f"SELECT COUNT(*) AS c FROM {_quote_identifier(name)}"
                        ).fetchone()
                        entry["row_count"] = int(row_count["c"]) if row_count else 0
                    except sqlite3.Error:
                        entry["row_count"] = None

                entries.append(entry)

            return {
                "command": "list_tables",
                "database_path": str(self.db_path),
                "objects": entries,
            }

    def _describe_table(
        self,
        table_name: str,
        *,
        sample_rows: int,
    ) -> dict[str, Any]:
        target = table_name.strip()
        if not target:
            raise ValueError("table_name is required for describe_table")

        with self._connect() as conn:
            object_row = conn.execute(
                "SELECT name, type, sql FROM sqlite_master "
                "WHERE name = ? AND type IN ('table', 'view')",
                (target,),
            ).fetchone()
            if object_row is None:
                raise ValueError(f"Table or view not found: {target}")

            pragma_columns = conn.execute(
                f"PRAGMA table_info({_quote_identifier(target)})"
            ).fetchall()

            columns = [
                {
                    "cid": int(col["cid"]),
                    "name": str(col["name"]),
                    "type": str(col["type"]) if col["type"] is not None else "",
                    "notnull": bool(col["notnull"]),
                    "default": col["dflt_value"],
                    "primary_key_position": int(col["pk"]),
                }
                for col in pragma_columns
            ]

            index_rows = conn.execute(
                f"PRAGMA index_list({_quote_identifier(target)})"
            ).fetchall()
            indexes: list[dict[str, Any]] = []
            for index_row in index_rows:
                index_name = str(index_row["name"])
                index_cols = conn.execute(
                    f"PRAGMA index_info({_quote_identifier(index_name)})"
                ).fetchall()
                indexes.append(
                    {
                        "name": index_name,
                        "unique": bool(index_row["unique"]),
                        "origin": index_row["origin"],
                        "partial": bool(index_row["partial"]),
                        "columns": [str(col["name"]) for col in index_cols],
                    }
                )

            fk_rows = conn.execute(
                f"PRAGMA foreign_key_list({_quote_identifier(target)})"
            ).fetchall()
            foreign_keys = [
                {
                    "id": int(fk["id"]),
                    "seq": int(fk["seq"]),
                    "table": str(fk["table"]),
                    "from": str(fk["from"]),
                    "to": str(fk["to"]),
                    "on_update": fk["on_update"],
                    "on_delete": fk["on_delete"],
                    "match": fk["match"],
                }
                for fk in fk_rows
            ]

            try:
                count_row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {_quote_identifier(target)}"
                ).fetchone()
                row_count = int(count_row["c"]) if count_row is not None else 0
            except sqlite3.Error:
                row_count = None

            samples: list[dict[str, Any]] = []
            if sample_rows > 0:
                sample_query = (
                    f"SELECT * FROM {_quote_identifier(target)} LIMIT {sample_rows}"
                )
                sample_result = conn.execute(sample_query).fetchall()
                sample_columns = [col["name"] for col in pragma_columns]
                samples = [
                    {
                        col: _normalize_result_value(value)
                        for col, value in zip(sample_columns, tuple(row), strict=False)
                    }
                    for row in sample_result
                ]

            return {
                "command": "describe_table",
                "database_path": str(self.db_path),
                "name": str(object_row["name"]),
                "type": str(object_row["type"]),
                "create_sql": object_row["sql"],
                "row_count": row_count,
                "columns": columns,
                "indexes": indexes,
                "foreign_keys": foreign_keys,
                "sample_rows": samples,
            }

    def _sqlite_tool(self, allowed_commands: set[SqliteCommand], **kwargs: Any) -> str:
        params = SqliteParams.model_validate(kwargs)

        try:
            if params.command not in allowed_commands:
                raise ValueError(
                    f"The '{params.command}' command is disabled for this tool "
                    "instance"
                )

            if params.command == "query":
                if params.sql is None:
                    raise ValueError("sql is required for command='query'")
                result = self._execute_query(
                    params.sql,
                    params.query_params,
                    output_format=params.output_format,
                    max_rows=params.max_rows,
                    allow_write=params.allow_write,
                )
            elif params.command == "list_tables":
                result = self._list_tables(
                    include_views=params.include_views,
                    include_row_counts=params.include_row_counts,
                )
            elif params.command == "describe_table":
                if params.table_name is None:
                    raise ValueError("table_name is required for describe_table")
                result = self._describe_table(
                    params.table_name,
                    sample_rows=params.sample_rows,
                )
            else:
                raise ValueError(f"Unknown command: {params.command}")

            return json.dumps({"ok": True, "result": result}, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                indent=2,
            )

    def get_tools(
        self,
        *,
        exclude: Sequence[SqliteCommand] | None = None,
    ) -> list[Tool]:
        exclude_set = set(exclude or [])
        unknown = exclude_set.difference(ALL_COMMANDS)
        if unknown:
            raise ValueError(f"Unknown commands in exclude list: {sorted(unknown)}")

        allowed = tuple(cmd for cmd in ALL_COMMANDS if cmd not in exclude_set)
        if not allowed:
            raise ValueError("Cannot exclude every sqlite command")

        cache_key = allowed
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        allowed_set = set(allowed)
        schema = SqliteParams.model_json_schema(ref_template="#/$defs/{model}")
        if (
            "properties" in schema
            and "command" in schema["properties"]
            and isinstance(schema["properties"]["command"], dict)
        ):
            schema["properties"]["command"]["enum"] = list(allowed)

        tool = Tool(
            name=self.tool_name,
            description=SQLITE_DESCRIPTION,
            parameters=schema.get("properties", {}),
            required=schema.get("required", []),
            definitions=schema.get("$defs"),
            run=partial(self._sqlite_tool, allowed_set),  # type: ignore[arg-type]
        )

        self._tool_cache[cache_key] = [tool]
        return [tool]
