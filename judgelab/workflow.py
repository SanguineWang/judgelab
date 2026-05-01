from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from judgelab.workspace import WorkspaceStore, quote_ident


@dataclass(frozen=True)
class AssetResult:
    table_name: str
    row_count: int


@dataclass(frozen=True)
class SplitResult:
    train_count: int
    val_count: int
    test_count: int
    total_count: int


ASSET_TABLES = {
    "raw": "raw_records",
    "sampled": "sampled_records",
    "labeled": "labeled_records",
    "reviewed": "reviewed_records",
    "train": "train_records",
    "val": "val_records",
    "test": "test_records",
    "predictions": "predictions",
}

DOWNSTREAM_TABLES = {
    "raw": ["sampled_records", "labeled_records", "reviewed_records", "train_records", "val_records", "test_records", "predictions"],
    "sampled": ["labeled_records", "reviewed_records", "train_records", "val_records", "test_records", "predictions"],
    "labeled": ["reviewed_records", "train_records", "val_records", "test_records", "predictions"],
    "reviewed": ["train_records", "val_records", "test_records", "predictions"],
    "splits": ["predictions"],
}


def create_random_sample(store: WorkspaceStore, dataset_id: str, sample_size: int, seed: int = 42) -> AssetResult:
    if sample_size <= 0:
        raise ValueError("样本量必须大于 0")
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    with duckdb.connect(str(db_path)) as conn:
        ensure_table_exists(conn, "raw_records")
        total = table_count(conn, "raw_records")
        limit = min(sample_size, total)
        conn.execute("DROP TABLE IF EXISTS sampled_records")
        conn.execute(
            """
            CREATE TABLE sampled_records AS
            SELECT *, 'random' AS __sample_source
            FROM raw_records
            ORDER BY hash(CAST(__record_id AS VARCHAR) || ?)
            LIMIT ?
            """,
            (str(seed), limit),
        )
        count = table_count(conn, "sampled_records")
        drop_tables(conn, DOWNSTREAM_TABLES["sampled"])
    return AssetResult(table_name="sampled_records", row_count=count)


def create_stratified_sample(
    store: WorkspaceStore,
    dataset_id: str,
    strata_column: str,
    sample_size: int,
    seed: int = 42,
) -> AssetResult:
    if sample_size <= 0:
        raise ValueError("样本量必须大于 0")
    if not strata_column:
        raise ValueError("分层字段不能为空")
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    with duckdb.connect(str(db_path)) as conn:
        ensure_table_exists(conn, "raw_records")
        ensure_column_exists(conn, "raw_records", strata_column)
        total = table_count(conn, "raw_records")
        limit = min(sample_size, total)
        groups = conn.execute(
            f"""
            SELECT COALESCE(NULLIF({quote_ident(strata_column)}, ''), '未填写') AS group_value, COUNT(*) AS row_count
            FROM raw_records
            GROUP BY 1
            """
        ).fetchall()
        allocations = proportional_allocations({row[0]: int(row[1]) for row in groups}, limit)
        conn.execute("DROP TABLE IF EXISTS __sample_allocations")
        conn.execute("CREATE TEMP TABLE __sample_allocations (group_value VARCHAR, sample_count BIGINT)")
        conn.executemany("INSERT INTO __sample_allocations VALUES (?, ?)", list(allocations.items()))
        conn.execute("DROP TABLE IF EXISTS sampled_records")
        conn.execute(
            f"""
            CREATE TABLE sampled_records AS
            WITH ranked AS (
                SELECT
                    raw_records.*,
                    COALESCE(NULLIF({quote_ident(strata_column)}, ''), '未填写') AS __group_value,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(NULLIF({quote_ident(strata_column)}, ''), '未填写')
                        ORDER BY hash(CAST(__record_id AS VARCHAR) || ?)
                    ) AS __rn
                FROM raw_records
            )
            SELECT ranked.* EXCLUDE (__group_value, __rn), 'stratified' AS __sample_source
            FROM ranked
            JOIN __sample_allocations ON ranked.__group_value = __sample_allocations.group_value
            WHERE ranked.__rn <= __sample_allocations.sample_count
            """,
            (str(seed),),
        )
        count = table_count(conn, "sampled_records")
        drop_tables(conn, DOWNSTREAM_TABLES["sampled"])
    return AssetResult(table_name="sampled_records", row_count=count)


def count_filtered_records(
    store: WorkspaceStore,
    dataset_id: str,
    filter_columns: Sequence[str],
    keywords: Sequence[str],
    keyword_mode: str = "any",
    field_mode: str = "any",
) -> int:
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    with duckdb.connect(str(db_path)) as conn:
        ensure_table_exists(conn, "raw_records")
        where_sql, params = build_keyword_filter(conn, "raw_records", filter_columns, keywords, keyword_mode, field_mode)
        return int(conn.execute(f"SELECT COUNT(*) FROM raw_records WHERE {where_sql}", params).fetchone()[0])


def create_filtered_sample(
    store: WorkspaceStore,
    dataset_id: str,
    filter_columns: Sequence[str],
    keywords: Sequence[str],
    sample_size: int,
    seed: int = 42,
    keyword_mode: str = "any",
    field_mode: str = "any",
) -> AssetResult:
    if sample_size <= 0:
        raise ValueError("样本量必须大于 0")
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    with duckdb.connect(str(db_path)) as conn:
        ensure_table_exists(conn, "raw_records")
        where_sql, params = build_keyword_filter(conn, "raw_records", filter_columns, keywords, keyword_mode, field_mode)
        total = int(conn.execute(f"SELECT COUNT(*) FROM raw_records WHERE {where_sql}", params).fetchone()[0])
        if total <= 0:
            raise ValueError("筛选结果为 0 行，请调整字段或关键词。")
        limit = min(sample_size, total)
        conn.execute("DROP TABLE IF EXISTS sampled_records")
        conn.execute(
            f"""
            CREATE TABLE sampled_records AS
            SELECT *, 'keyword_filter' AS __sample_source
            FROM raw_records
            WHERE {where_sql}
            ORDER BY hash(CAST(__record_id AS VARCHAR) || ?)
            LIMIT ?
            """,
            (*params, str(seed), limit),
        )
        count = table_count(conn, "sampled_records")
        drop_tables(conn, DOWNSTREAM_TABLES["sampled"])
    return AssetResult(table_name="sampled_records", row_count=count)


def create_splits(
    store: WorkspaceStore,
    dataset_id: str,
    source_table: str,
    label_column: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    with duckdb.connect(str(db_path)) as conn:
        ensure_table_exists(conn, source_table)
        ensure_column_exists(conn, source_table, label_column)
        for table in ["train_records", "val_records", "test_records"]:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            f"""
            CREATE TEMP TABLE __split_ranked AS
            SELECT
                *,
                COUNT(*) OVER (PARTITION BY {quote_ident(label_column)}) AS __label_count,
                ROW_NUMBER() OVER (
                    PARTITION BY {quote_ident(label_column)}
                    ORDER BY hash(CAST(__record_id AS VARCHAR) || ?)
                ) AS __rn
            FROM {quote_ident(source_table)}
            WHERE COALESCE({quote_ident(label_column)}, '') <> ''
            """,
            (str(seed),),
        )
        conn.execute(
            """
            CREATE TABLE train_records AS
            SELECT * EXCLUDE (__label_count, __rn)
            FROM __split_ranked
            WHERE __rn <= FLOOR(__label_count * ?)
            """,
            (train_ratio,),
        )
        conn.execute(
            """
            CREATE TABLE val_records AS
            SELECT * EXCLUDE (__label_count, __rn)
            FROM __split_ranked
            WHERE __rn > FLOOR(__label_count * ?)
              AND __rn <= FLOOR(__label_count * (? + ?))
            """,
            (train_ratio, train_ratio, val_ratio),
        )
        conn.execute(
            """
            CREATE TABLE test_records AS
            SELECT * EXCLUDE (__label_count, __rn)
            FROM __split_ranked
            WHERE __rn > FLOOR(__label_count * (? + ?))
            """,
            (train_ratio, val_ratio),
        )
        train_count = table_count(conn, "train_records")
        val_count = table_count(conn, "val_records")
        test_count = table_count(conn, "test_records")
        drop_tables(conn, DOWNSTREAM_TABLES["splits"])
    return SplitResult(train_count=train_count, val_count=val_count, test_count=test_count, total_count=train_count + val_count + test_count)


def clear_downstream_assets(store: WorkspaceStore, dataset_id: str, step: str) -> None:
    import duckdb

    tables = DOWNSTREAM_TABLES.get(step, [])
    if not tables:
        return
    db_path = store.duckdb_path(dataset_id)
    if not db_path.exists():
        return
    with duckdb.connect(str(db_path)) as conn:
        drop_tables(conn, tables)


def get_asset_stats(store: WorkspaceStore, dataset_id: str) -> Dict[str, int]:
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    if not db_path.exists():
        return {key: 0 for key in ASSET_TABLES}
    with duckdb.connect(str(db_path)) as conn:
        return {key: table_count(conn, table) if table_exists(conn, table) else 0 for key, table in ASSET_TABLES.items()}


def preview_table(store: WorkspaceStore, dataset_id: str, table_name: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    if not db_path.exists():
        return []
    with duckdb.connect(str(db_path)) as conn:
        if not table_exists(conn, table_name):
            return []
        rows = conn.execute(f"SELECT * FROM {quote_ident(table_name)} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        columns = [description[0] for description in conn.description]
    return [dict(zip(columns, row)) for row in rows]


def fetch_table_records(store: WorkspaceStore, dataset_id: str, table_name: str, limit: int | None = None) -> List[Dict[str, Any]]:
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    if not db_path.exists():
        return []
    with duckdb.connect(str(db_path)) as conn:
        if not table_exists(conn, table_name):
            return []
        sql = f"SELECT * FROM {quote_ident(table_name)}"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
        columns = [description[0] for description in conn.description]
    return [dict(zip(columns, row)) for row in rows]


def replace_records_table(store: WorkspaceStore, dataset_id: str, table_name: str, records: List[Dict[str, Any]]) -> AssetResult:
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    if not records:
        raise ValueError("没有可写入的数据")
    columns: List[str] = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                columns.append(key)
                seen.add(key)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table_name)}")
        column_sql = ", ".join(f"{quote_ident(column)} VARCHAR" for column in columns)
        conn.execute(f"CREATE TABLE {quote_ident(table_name)} ({column_sql})")
        placeholders = ", ".join(["?"] * len(columns))
        conn.executemany(
            f"INSERT INTO {quote_ident(table_name)} VALUES ({placeholders})",
            [[none_to_empty(record.get(column)) for column in columns] for record in records],
        )
        count = table_count(conn, table_name)
    return AssetResult(table_name=table_name, row_count=count)


def label_distribution(store: WorkspaceStore, dataset_id: str, table_name: str, label_column: str) -> Dict[str, int]:
    import duckdb

    db_path = store.duckdb_path(dataset_id)
    if not db_path.exists():
        return {}
    with duckdb.connect(str(db_path)) as conn:
        if not table_exists(conn, table_name):
            return {}
        ensure_column_exists(conn, table_name, label_column)
        rows = conn.execute(
            f"""
            SELECT COALESCE(NULLIF({quote_ident(label_column)}, ''), '未标注') AS label_value, COUNT(*)
            FROM {quote_ident(table_name)}
            GROUP BY 1
            ORDER BY 2 DESC
            """
        ).fetchall()
    return {row[0]: int(row[1]) for row in rows}


def table_count(conn: Any, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table_name)}").fetchone()[0])


def drop_tables(conn: Any, table_names: List[str]) -> None:
    for table_name in table_names:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table_name)}")


def table_exists(conn: Any, table_name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            (table_name,),
        ).fetchone()[0]
    )


def ensure_table_exists(conn: Any, table_name: str) -> None:
    if not table_exists(conn, table_name):
        raise ValueError(f"数据表不存在: {table_name}")


def ensure_column_exists(conn: Any, table_name: str, column_name: str) -> None:
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()]
    if column_name not in columns:
        raise ValueError(f"字段不存在: {column_name}")


def build_keyword_filter(
    conn: Any,
    table_name: str,
    filter_columns: Sequence[str],
    keywords: Sequence[str],
    keyword_mode: str = "any",
    field_mode: str = "any",
) -> tuple[str, tuple[str, ...]]:
    columns = [str(column).strip() for column in filter_columns if str(column).strip()]
    terms = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if not columns:
        raise ValueError("请至少选择一个筛选字段")
    if not terms:
        raise ValueError("请至少输入一个关键词")
    for column in columns:
        ensure_column_exists(conn, table_name, column)
    if keyword_mode not in {"any", "all"}:
        raise ValueError("关键词匹配方式不支持")
    if field_mode not in {"any", "same_field"}:
        raise ValueError("字段匹配方式不支持")

    params: list[str] = []

    def condition(column: str, keyword: str) -> str:
        params.append(f"%{keyword}%")
        return f"COALESCE(CAST({quote_ident(column)} AS VARCHAR), '') ILIKE ?"

    if keyword_mode == "any":
        clauses = [condition(column, keyword) for column in columns for keyword in terms]
        return "(" + " OR ".join(clauses) + ")", tuple(params)

    if field_mode == "same_field":
        column_clauses = []
        for column in columns:
            column_clauses.append("(" + " AND ".join(condition(column, keyword) for keyword in terms) + ")")
        return "(" + " OR ".join(column_clauses) + ")", tuple(params)

    keyword_clauses = []
    for keyword in terms:
        keyword_clauses.append("(" + " OR ".join(condition(column, keyword) for column in columns) + ")")
    return "(" + " AND ".join(keyword_clauses) + ")", tuple(params)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if any(value < 0 for value in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("划分比例不能小于 0")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集、测试集比例之和必须等于 1")


def proportional_allocations(group_sizes: Dict[str, int], total: int) -> Dict[str, int]:
    if total <= 0:
        return {key: 0 for key in group_sizes}
    population = sum(group_sizes.values())
    raw = {key: size / population * total for key, size in group_sizes.items()}
    allocations = {key: int(value) for key, value in raw.items()}
    for key, size in group_sizes.items():
        if size > 0 and allocations[key] == 0 and sum(allocations.values()) < total:
            allocations[key] = 1
    remaining = total - sum(allocations.values())
    for key in sorted(raw, key=lambda item: raw[item] - int(raw[item]), reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < group_sizes[key]:
            allocations[key] += 1
            remaining -= 1
    return allocations


def none_to_empty(value: Any) -> str:
    return "" if value is None else str(value)
