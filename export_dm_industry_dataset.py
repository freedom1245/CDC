from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


try:
    import dmPython  # type: ignore

    HAS_DMPYTHON = True
except ImportError:  # pragma: no cover - depends on local driver installation
    dmPython = None
    HAS_DMPYTHON = False


DEFAULT_QUERY = """
SELECT
    ALARM_ID,
    RUN_TERM_ID,
    RUN_METER_ID,
    MEASURE_POINT_NO,
    PRE_RECE_TIME,
    ALARM_TIME,
    ALARM_CODE,
    ALARM_EXT_DATA,
    IS_FALS,
    IS_REST,
    REST_TIME,
    ALARM_NAME,
    ALARM_LEVEL_CODE,
    ALARM_TYPE_CODE,
    ALARM_SOUR_CODE,
    ALARM_CLASS,
    ALARM_CREAT_WAY,
    IS_DISP,
    PROC_PERS,
    TERMINAL_TYPE,
    WORK_ORDER_MODE,
    ORDER_ID,
    ORDER_STATE,
    LOAD_TIME,
    STATUTE_ALARM_CODE,
    DATA_PART,
    AREA_CODE,
    SYNC_TIME,
    HOLO_DATA_SOURCE
FROM HY.MK_MC_TERM_ALARM
"""

EXPORT_COLUMNS = [
    "ALARM_ID",
    "RUN_TERM_ID",
    "RUN_METER_ID",
    "MEASURE_POINT_NO",
    "PRE_RECE_TIME",
    "ALARM_TIME",
    "ALARM_CODE",
    "ALARM_EXT_DATA",
    "IS_FALS",
    "IS_REST",
    "REST_TIME",
    "ALARM_NAME",
    "ALARM_LEVEL_CODE",
    "ALARM_TYPE_CODE",
    "ALARM_SOUR_CODE",
    "ALARM_CLASS",
    "ALARM_CREAT_WAY",
    "IS_DISP",
    "PROC_PERS",
    "TERMINAL_TYPE",
    "WORK_ORDER_MODE",
    "ORDER_ID",
    "ORDER_STATE",
    "LOAD_TIME",
    "STATUTE_ALARM_CODE",
    "DATA_PART",
    "AREA_CODE",
    "SYNC_TIME",
    "HOLO_DATA_SOURCE",
]

DISQL_SEPARATOR = "||#||"
DEFAULT_BURST_WINDOW_SECONDS = 60


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_alarm_ext_items(raw: Any) -> int:
    text = _safe_str(raw)
    if not text:
        return 0
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return len(re.findall(r'"code"\s*:', text))
    if isinstance(payload, list):
        return len(payload)
    return 0


def _map_event_type(alarm_name: str, alarm_type_code: str) -> str:
    if "恢复" in alarm_name and "CT" in alarm_name:
        return "CT_OPEN_RECOVER"
    if "CT" in alarm_name and "开路" in alarm_name:
        return "CT_OPEN"
    if "逆相序" in alarm_name:
        return "PHASE_REVERSE"
    if "时钟异常" in alarm_name:
        return "CLOCK_ABNORMAL"
    if "上电" in alarm_name:
        return "TERMINAL_POWER_ON"
    if "掉电" in alarm_name:
        return "TERMINAL_POWER_OFF"
    if "通信" in alarm_name:
        return "COMMUNICATION_OVERFLOW"
    if "不平衡" in alarm_name:
        return "CURRENT_IMBALANCE"
    if "零序电流" in alarm_name:
        return "ZERO_CURRENT_HIGH"
    if alarm_type_code == "12":
        return "CT_OPEN"
    if alarm_type_code == "2":
        return "PHASE_REVERSE"
    if alarm_type_code == "5":
        return "CLOCK_ABNORMAL"
    if alarm_type_code == "8":
        return "TERMINAL_POWER"
    return "OTHER_ALARM"


def _map_business_domain(event_type: str) -> str:
    if event_type in {"CT_OPEN", "CT_OPEN_RECOVER", "PHASE_REVERSE", "CURRENT_IMBALANCE", "ZERO_CURRENT_HIGH"}:
        return "METERING"
    if event_type in {"COMMUNICATION_OVERFLOW"}:
        return "COMMUNICATION"
    if event_type in {"CLOCK_ABNORMAL"}:
        return "TIME_SYNC"
    if event_type in {"TERMINAL_POWER_ON", "TERMINAL_POWER_OFF", "TERMINAL_POWER"}:
        return "TERMINAL_STATUS"
    return "DEVICE_ALARM"


def _map_source_service(holo_data_source: str, alarm_source_code: str, terminal_type: str) -> str:
    left = holo_data_source or "UNKNOWN"
    middle = alarm_source_code or "UNKNOWN"
    right = terminal_type or "UNKNOWN"
    return f"{left}_{middle}_{right}"


def _map_user_level(alarm_level_code: str, event_type: str) -> str:
    if alarm_level_code == "1" and event_type in {
        "TERMINAL_POWER_OFF",
        "TERMINAL_POWER_ON",
        "PHASE_REVERSE",
        "COMMUNICATION_OVERFLOW",
    }:
        return "CRITICAL"
    if alarm_level_code == "1":
        return "IMPORTANT"
    return "NORMAL"


def _business_value(alarm_level_code: str, event_type: str) -> float:
    score = 60.0 if alarm_level_code == "1" else 30.0
    if event_type in {"TERMINAL_POWER_OFF", "TERMINAL_POWER_ON", "COMMUNICATION_OVERFLOW"}:
        score += 20.0
    elif event_type in {"PHASE_REVERSE", "CURRENT_IMBALANCE", "ZERO_CURRENT_HIGH"}:
        score += 15.0
    elif event_type == "CLOCK_ABNORMAL":
        score += 10.0
    elif event_type == "CT_OPEN_RECOVER":
        score -= 10.0
    return max(score, 0.0)


def _consistency_risk(event_type: str, alarm_class: str) -> float:
    if event_type in {"TERMINAL_POWER_OFF", "TERMINAL_POWER_ON"}:
        return 0.8
    if event_type in {"COMMUNICATION_OVERFLOW"}:
        return 0.6
    if event_type in {"PHASE_REVERSE", "CURRENT_IMBALANCE", "ZERO_CURRENT_HIGH"}:
        return 0.4
    if event_type in {"CT_OPEN_RECOVER"}:
        return 0.25
    if alarm_class == "2":
        return 0.35
    return 0.1


def _deadline_steps(alarm_level_code: str, event_type: str) -> float:
    if event_type in {
        "TERMINAL_POWER_OFF",
        "TERMINAL_POWER_ON",
        "COMMUNICATION_OVERFLOW",
        "PHASE_REVERSE",
    }:
        return 5.0 if alarm_level_code == "1" else 6.0
    if event_type in {"CLOCK_ABNORMAL", "CT_OPEN", "CURRENT_IMBALANCE", "ZERO_CURRENT_HIGH"}:
        return 6.0 if alarm_level_code == "1" else 8.0
    if event_type in {"CT_OPEN_RECOVER"}:
        return 8.0
    return 7.0


def _scaled_queue_wait_time(processing_delay_seconds: float) -> float:
    capped_delay = min(max(processing_delay_seconds, 0.0), 41.0)
    return max(1.0, min(5.1, 1.0 + capped_delay / 10.0))


def _estimated_sync_cost(
    record_size: float,
    changed_columns_count: float,
    processing_delay_seconds: float,
    event_type: str,
) -> float:
    type_complexity = 1.0
    if event_type in {"PHASE_REVERSE", "CURRENT_IMBALANCE", "ZERO_CURRENT_HIGH"}:
        type_complexity = 1.6
    elif event_type in {"COMMUNICATION_OVERFLOW", "CLOCK_ABNORMAL"}:
        type_complexity = 1.3
    elif event_type in {"TERMINAL_POWER_OFF", "TERMINAL_POWER_ON", "TERMINAL_POWER"}:
        type_complexity = 1.4
    elif event_type == "CT_OPEN_RECOVER":
        type_complexity = 0.8

    size_factor = min(max(record_size - 200.0, 0.0), 1400.0) / 250.0
    latency_factor = min(max(processing_delay_seconds, 0.0), 30.0) * 0.04
    estimated = (
        1.5
        + 0.9 * type_complexity
        + 0.08 * min(changed_columns_count, 20.0)
        + 0.35 * size_factor
        + latency_factor
    )
    return max(1.0, min(12.5, estimated))


def _rolling_count_per_key(
    frame: pd.DataFrame,
    key_column: str,
    timestamp_column: str,
    window_seconds: int,
) -> pd.Series:
    counts = pd.Series(0, index=frame.index, dtype="int64")
    for _, group in frame.groupby(key_column, sort=False):
        timestamps = group[timestamp_column].astype("int64") // 10**9
        values = timestamps.to_numpy()
        left = 0
        group_counts: list[int] = []
        for right, current in enumerate(values):
            while current - values[left] > window_seconds:
                left += 1
            group_counts.append(right - left + 1)
        counts.loc[group.index] = group_counts
    return counts


def _aggregate_alarm_bursts(
    frame: pd.DataFrame,
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    aggregated = frame.copy()
    aggregated["prev_timestamp"] = aggregated.groupby(["_object_key", "event_type"], sort=False)[
        "timestamp"
    ].shift()
    gap_seconds = (aggregated["timestamp"] - aggregated["prev_timestamp"]).dt.total_seconds()
    starts_new_burst = gap_seconds.isna() | (gap_seconds > float(burst_window_seconds))
    aggregated["burst_index"] = starts_new_burst.groupby(
        [aggregated["_object_key"], aggregated["event_type"]]
    ).cumsum()
    aggregated["burst_id"] = (
        aggregated["_object_key"]
        + "||"
        + aggregated["event_type"]
        + "||"
        + aggregated["burst_index"].astype(str)
    )

    grouped = aggregated.groupby("burst_id", sort=False)
    burst_size = grouped.size().rename("burst_size")
    result = grouped.agg(
        event_id=("event_id", "first"),
        timestamp=("timestamp", "min"),
        table_name=("table_name", "first"),
        event_type=("event_type", "first"),
        business_domain=("business_domain", "first"),
        source_service=("source_service", "first"),
        user_level=("user_level", "first"),
        record_size=("record_size", "max"),
        changed_columns_count=("changed_columns_count", "max"),
        business_value=("business_value", "max"),
        consistency_risk=("consistency_risk", "max"),
        dependency_count=("dependency_count", "max"),
        queue_wait_time=("queue_wait_time", "max"),
        deadline=("deadline", "min"),
        estimated_sync_cost=("estimated_sync_cost", "max"),
        ALARM_CODE=("ALARM_CODE", "first"),
        _object_key=("_object_key", "first"),
    ).join(burst_size)
    result = result.reset_index(drop=True)
    result["event_id"] = result.apply(
        lambda row: row["event_id"]
        if int(row["burst_size"]) <= 1
        else f"{row['event_id']}::b{int(row['burst_size'])}",
        axis=1,
    )
    burst_bonus = result["burst_size"].map(lambda value: min(math.log2(max(float(value), 1.0)), 4.0))
    result["estimated_sync_cost"] = (
        result["estimated_sync_cost"] + 0.35 * burst_bonus
    ).clip(lower=1.0, upper=12.5)
    result["dependency_count"] = (
        result["dependency_count"] + 0.25 * (result["burst_size"] - 1.0).clip(lower=0.0, upper=8.0)
    )
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)
def transform_alarm_frame(
    frame: pd.DataFrame,
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
) -> pd.DataFrame:
    transformed = frame.copy()
    transformed["ALARM_TIME"] = pd.to_datetime(transformed["ALARM_TIME"], errors="coerce")
    transformed["SYNC_TIME"] = pd.to_datetime(transformed["SYNC_TIME"], errors="coerce")
    transformed["PRE_RECE_TIME"] = pd.to_datetime(transformed["PRE_RECE_TIME"], errors="coerce")

    transformed["event_id"] = transformed["ALARM_ID"].astype(str)
    transformed["timestamp"] = (
        transformed["PRE_RECE_TIME"]
        .fillna(transformed["ALARM_TIME"])
        .fillna(transformed["SYNC_TIME"])
    )
    transformed = transformed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    transformed["table_name"] = "MK_MC_TERM_ALARM"

    transformed["event_type"] = transformed.apply(
        lambda row: _map_event_type(
            _safe_str(row.get("ALARM_NAME")),
            _safe_str(row.get("ALARM_TYPE_CODE")),
        ),
        axis=1,
    )
    transformed["business_domain"] = transformed["event_type"].map(_map_business_domain)
    transformed["source_service"] = transformed.apply(
        lambda row: _map_source_service(
            _safe_str(row.get("HOLO_DATA_SOURCE")),
            _safe_str(row.get("ALARM_SOUR_CODE")),
            _safe_str(row.get("TERMINAL_TYPE")),
        ),
        axis=1,
    )
    transformed["user_level"] = transformed.apply(
        lambda row: _map_user_level(
            _safe_str(row.get("ALARM_LEVEL_CODE")),
            _safe_str(row.get("event_type")),
        ),
        axis=1,
    )

    transformed["record_size"] = transformed["ALARM_EXT_DATA"].map(
        lambda value: float(len(_safe_str(value)))
    )
    transformed["changed_columns_count"] = transformed["ALARM_EXT_DATA"].map(
        lambda value: float(_count_alarm_ext_items(value))
    )
    transformed["business_value"] = transformed.apply(
        lambda row: _business_value(
            _safe_str(row.get("ALARM_LEVEL_CODE")),
            _safe_str(row.get("event_type")),
        ),
        axis=1,
    )
    transformed["consistency_risk"] = transformed.apply(
        lambda row: _consistency_risk(
            _safe_str(row.get("event_type")),
            _safe_str(row.get("ALARM_CLASS")),
        ),
        axis=1,
    )
    transformed["dependency_count"] = (
        1.0
        + transformed["RUN_METER_ID"].map(lambda value: 1.0 if _safe_str(value) else 0.0)
        + transformed["RUN_TERM_ID"].map(lambda value: 1.0 if _safe_str(value) else 0.0)
        + transformed["changed_columns_count"].clip(upper=25.0) / 5.0
    )

    processing_delay_seconds = (
        (transformed["SYNC_TIME"] - transformed["timestamp"]).dt.total_seconds().fillna(0.0)
    ).clip(lower=0.0)
    transformed["queue_wait_time"] = processing_delay_seconds.map(_scaled_queue_wait_time)
    transformed["deadline"] = transformed.apply(
        lambda row: _deadline_steps(
            _safe_str(row.get("ALARM_LEVEL_CODE")),
            _safe_str(row.get("event_type")),
        ),
        axis=1,
    )
    transformed["estimated_sync_cost"] = transformed.apply(
        lambda row: _estimated_sync_cost(
            _safe_numeric(row.get("record_size")),
            _safe_numeric(row.get("changed_columns_count")),
            _safe_numeric(processing_delay_seconds.loc[row.name]),
            _safe_str(row.get("event_type")),
        ),
        axis=1,
    )

    object_key = transformed["RUN_TERM_ID"].fillna("").astype(str)
    missing_object_mask = object_key.eq("")
    object_key = object_key.mask(
        missing_object_mask,
        transformed["RUN_METER_ID"].fillna("").astype(str),
    )
    transformed["_object_key"] = object_key
    transformed = _aggregate_alarm_bursts(
        transformed,
        burst_window_seconds=burst_window_seconds,
    )

    retry_key = (
        transformed["_object_key"].fillna("").astype(str)
        + "||"
        + transformed["ALARM_CODE"].fillna("").astype(str)
    )
    transformed["retry_count"] = retry_key.groupby(retry_key).cumcount().astype(float)

    transformed["event_hour"] = transformed["timestamp"].dt.hour.fillna(0).astype(int)
    transformed["is_peak_hour"] = transformed["event_hour"].between(9, 18).astype(int)

    transformed["source_load"] = _rolling_count_per_key(
        transformed.assign(_source_key=transformed["source_service"]),
        key_column="_source_key",
        timestamp_column="timestamp",
        window_seconds=15 * 60,
    ).astype(float)
    transformed["db_load"] = _rolling_count_per_key(
        transformed.assign(_global_key="GLOBAL"),
        key_column="_global_key",
        timestamp_column="timestamp",
        window_seconds=15 * 60,
    ).astype(float)

    transformed["object_event_count"] = transformed.groupby("_object_key").cumcount() + 1
    transformed["is_hot_data"] = (transformed["object_event_count"] >= 3).astype(int)

    selected_columns = [
        "event_id",
        "timestamp",
        "event_type",
        "table_name",
        "business_domain",
        "source_service",
        "user_level",
        "record_size",
        "changed_columns_count",
        "estimated_sync_cost",
        "business_value",
        "consistency_risk",
        "dependency_count",
        "queue_wait_time",
        "deadline",
        "retry_count",
        "source_load",
        "db_load",
        "is_peak_hour",
        "is_hot_data",
    ]
    return transformed[selected_columns].copy()


def load_alarm_frame_from_dameng(
    host: str,
    port: int,
    user: str,
    password: str,
    query: str = DEFAULT_QUERY,
    disql_path: str = r"D:\dmdbms\bin\DIsql.exe",
) -> pd.DataFrame:
    if HAS_DMPYTHON and dmPython is not None:
        connection = dmPython.connect(user=user, password=password, server=host, port=port)
        try:
            return pd.read_sql(query, connection)
        finally:
            connection.close()
    return load_alarm_frame_with_disql(
        host=host,
        port=port,
        user=user,
        password=password,
        query=query,
        disql_path=disql_path,
    )


def load_alarm_frame_with_disql(
    host: str,
    port: int,
    user: str,
    password: str,
    query: str,
    disql_path: str,
) -> pd.DataFrame:
    disql = Path(disql_path)
    if not disql.exists():
        raise RuntimeError(f"DIsql executable not found: {disql}")

    project_root = Path(__file__).resolve().parent
    temp_dir = project_root / "outputs" / "dm_export_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    spool_path = temp_dir / f"dm_export_{uuid.uuid4().hex}.txt"

    sql_script = f"""
set heading off
set feedback off
set timing off
set time off
set pagesize 0
set linesize 131072
set long 131072
set wrap off
set lobcomplete on
set lineshow off
set trimspool on
set colsep '{DISQL_SEPARATOR}'
spool {spool_path} replace
{query.strip().rstrip(';')};
spool off
exit
"""

    try:
        completed = subprocess.run(
            [str(disql), f"{user}/{password}@{host}:{port}"],
            input=sql_script,
            text=True,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "DIsql export failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if not spool_path.exists():
            raise RuntimeError(
                "DIsql export did not create spool file.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        rows = _parse_disql_spool(spool_path, expected_columns=EXPORT_COLUMNS)
        return pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    finally:
        if spool_path.exists():
            for _ in range(5):
                try:
                    spool_path.unlink()
                    break
                except PermissionError:
                    time.sleep(0.2)


def _parse_disql_spool(
    spool_path: Path,
    expected_columns: list[str],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in spool_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("SQL>"):
            continue
        if line.startswith("已用时间"):
            continue
        if DISQL_SEPARATOR not in line:
            continue
        parts = [part.strip() for part in line.split(DISQL_SEPARATOR)]
        if len(parts) != len(expected_columns):
            continue
        rows.append(parts)
    if not rows:
        raise RuntimeError(
            f"No rows parsed from DIsql spool file: {spool_path}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Dameng HY.MK_MC_TERM_ALARM into the project's industry dataset schema."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5236)
    parser.add_argument("--user", default="SYSDBA")
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--output",
        default="data/industry/dm_alarm_events.csv",
        help="Output CSV path in project workspace.",
    )
    parser.add_argument(
        "--query-file",
        default="",
        help="Optional SQL file path. Defaults to querying HY.MK_MC_TERM_ALARM.",
    )
    parser.add_argument(
        "--disql-path",
        default=r"D:\dmdbms\bin\DIsql.exe",
        help="Path to DIsql.exe used when dmPython is unavailable.",
    )
    parser.add_argument(
        "--burst-window-seconds",
        type=int,
        default=DEFAULT_BURST_WINDOW_SECONDS,
        help="Aggregate repeated alarms for the same object and event type within this many seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8")
    else:
        query = DEFAULT_QUERY

    raw_frame = load_alarm_frame_from_dameng(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        query=query,
        disql_path=args.disql_path,
    )
    transformed = transform_alarm_frame(
        raw_frame,
        burst_window_seconds=int(args.burst_window_seconds),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[dm-export] rows: {len(transformed)}")
    print(f"[dm-export] burst_window_seconds: {int(args.burst_window_seconds)}")
    print(f"[dm-export] output: {output_path}")


if __name__ == "__main__":
    main()
