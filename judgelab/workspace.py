from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    name: str
    description: str
    status: str
    row_count: int
    column_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DatasetStats:
    row_count: int
    column_count: int
    raw_data_ready: bool
    schema: List[str]


@dataclass(frozen=True)
class ImportBatch:
    batch_id: str
    file_name: str
    row_count: int
    status: str
    created_at: str
    error_message: str


class WorkspaceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.datasets_dir = self.root / "datasets"
        self.db_path = self.root / "workspace.db"
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self._init_workspace_db()

    def create_dataset(self, name: str, description: str = "") -> Dataset:
        dataset_id = new_dataset_id()
        now = utc_now()
        dataset_dir = self.dataset_dir(dataset_id)
        for subdir in [
            "files/imports",
            "files/exports",
            "artifacts/samples",
            "artifacts/labels",
            "artifacts/splits",
            "artifacts/predictions",
            "artifacts/reports",
            "models",
            "workflows",
            "jobs",
        ]:
            (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

        dataset_json = {
            "dataset_id": dataset_id,
            "name": name,
            "description": description,
            "created_at": now,
            "duckdb_path": str(self.duckdb_path(dataset_id)),
        }
        (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, ensure_ascii=False, indent=2), encoding="utf-8")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, description, status, row_count, column_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (dataset_id, name, description, "empty", 0, 0, now, now),
            )
        return self.get_dataset(dataset_id)

    def list_datasets(self) -> List[Dataset]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT dataset_id, name, description, status, row_count, column_count, created_at, updated_at
                FROM datasets
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dataset_from_row(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> Dataset:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT dataset_id, name, description, status, row_count, column_count, created_at, updated_at
                FROM datasets
                WHERE dataset_id = ?
                """,
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"数据集不存在: {dataset_id}")
        return dataset_from_row(row)

    def update_dataset_stats(self, dataset_id: str, row_count: int, column_count: int, status: str = "imported") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE datasets
                SET row_count = ?, column_count = ?, status = ?, updated_at = ?
                WHERE dataset_id = ?
                """,
                (row_count, column_count, status, utc_now(), dataset_id),
            )

    def get_dataset_stats(self, dataset_id: str) -> DatasetStats:
        schema = self.get_raw_schema(dataset_id)
        row_count = self.count_raw_records(dataset_id) if self.duckdb_path(dataset_id).exists() else 0
        return DatasetStats(
            row_count=row_count,
            column_count=len(schema),
            raw_data_ready=row_count > 0,
            schema=schema,
        )

    def get_raw_schema(self, dataset_id: str) -> List[str]:
        db_path = self.duckdb_path(dataset_id)
        if not db_path.exists():
            return []
        import duckdb

        with duckdb.connect(str(db_path)) as conn:
            table_exists = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'dataset_schema'"
            ).fetchone()[0]
            if not table_exists:
                return []
            rows = conn.execute("SELECT column_name FROM dataset_schema ORDER BY column_index").fetchall()
            return [row[0] for row in rows]

    def count_raw_records(self, dataset_id: str) -> int:
        db_path = self.duckdb_path(dataset_id)
        if not db_path.exists():
            return 0
        import duckdb

        with duckdb.connect(str(db_path)) as conn:
            table_exists = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'raw_records'"
            ).fetchone()[0]
            if not table_exists:
                return 0
            return int(conn.execute("SELECT count(*) FROM raw_records").fetchone()[0])

    def preview_raw_records(self, dataset_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        db_path = self.duckdb_path(dataset_id)
        if not db_path.exists():
            return []
        schema = self.get_raw_schema(dataset_id)
        if not schema:
            return []
        import duckdb

        data_columns = ", ".join(quote_ident(column) for column in schema)
        with duckdb.connect(str(db_path)) as conn:
            rows = conn.execute(
                f"SELECT {data_columns} FROM raw_records ORDER BY __record_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(zip(schema, row)) for row in rows]

    def list_import_batches(self, dataset_id: str, limit: int = 20) -> List[ImportBatch]:
        db_path = self.duckdb_path(dataset_id)
        if not db_path.exists():
            return []
        import duckdb

        with duckdb.connect(str(db_path)) as conn:
            table_exists = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'import_batches'"
            ).fetchone()[0]
            if not table_exists:
                return []
            rows = conn.execute(
                """
                SELECT batch_id, file_name, row_count, status, created_at, error_message
                FROM import_batches
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ImportBatch(
                batch_id=row[0],
                file_name=row[1],
                row_count=int(row[2]),
                status=row[3],
                created_at=row[4],
                error_message=row[5],
            )
            for row in rows
        ]

    def clear_dataset(self, dataset_id: str, scope: str = "derived") -> None:
        if scope not in {"derived", "raw", "models", "history", "all"}:
            raise ValueError(f"不支持的清空范围: {scope}")
        dataset_dir = self.dataset_dir(dataset_id)

        if scope in {"derived", "all"}:
            for relative in ["artifacts/samples", "artifacts/labels", "artifacts/splits", "artifacts/predictions", "artifacts/reports"]:
                reset_dir(dataset_dir / relative)
        if scope in {"models", "all"}:
            reset_dir(dataset_dir / "models")
        if scope in {"history", "all"}:
            reset_dir(dataset_dir / "workflows")
            reset_dir(dataset_dir / "jobs")
        if scope in {"raw", "all"}:
            db_path = self.duckdb_path(dataset_id)
            if db_path.exists():
                db_path.unlink()
            reset_dir(dataset_dir / "files/imports")
            self.update_dataset_stats(dataset_id, row_count=0, column_count=0, status="empty")

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def duckdb_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "data.duckdb"

    def imports_dir(self, dataset_id: str) -> Path:
        path = self.dataset_dir(dataset_id) / "files" / "imports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_workspace_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    column_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )


def dataset_from_row(row: sqlite3.Row) -> Dataset:
    return Dataset(
        dataset_id=row["dataset_id"],
        name=row["name"],
        description=row["description"],
        status=row["status"],
        row_count=int(row["row_count"]),
        column_count=int(row["column_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_dataset_id() -> str:
    return f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
