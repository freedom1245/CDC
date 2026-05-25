import pandas as pd

from export_industry_dataset import transform_alarm_frame


def test_transform_alarm_frame_builds_project_schema() -> None:
    source = pd.DataFrame(
        [
            {
                "ALARM_ID": "a1",
                "RUN_TERM_ID": "term_1",
                "RUN_METER_ID": "meter_1",
                "MEASURE_POINT_NO": 1,
                "PRE_RECE_TIME": "2025-01-03 05:49:38",
                "ALARM_TIME": "2025-01-03 05:49:33",
                "ALARM_CODE": "E200003A01",
                "ALARM_EXT_DATA": '[{"code":"x","name":"n1","value":"1"},{"code":"y","name":"n2","value":"2"}]',
                "IS_FALS": None,
                "IS_REST": None,
                "REST_TIME": None,
                "ALARM_NAME": "月通信流量越限",
                "ALARM_LEVEL_CODE": "2",
                "ALARM_TYPE_CODE": "9",
                "ALARM_SOUR_CODE": "1",
                "ALARM_CLASS": "1",
                "ALARM_CREAT_WAY": "1",
                "IS_DISP": "0",
                "PROC_PERS": None,
                "TERMINAL_TYPE": "26",
                "WORK_ORDER_MODE": None,
                "ORDER_ID": None,
                "ORDER_STATE": None,
                "LOAD_TIME": "2025-01-03 05:49:39",
                "STATUTE_ALARM_CODE": None,
                "DATA_PART": 20250103,
                "AREA_CODE": "080000",
                "SYNC_TIME": "2025-01-03 05:49:46",
                "HOLO_DATA_SOURCE": "SS",
            },
            {
                "ALARM_ID": "a2",
                "RUN_TERM_ID": "term_1",
                "RUN_METER_ID": "meter_1",
                "MEASURE_POINT_NO": 1,
                "PRE_RECE_TIME": "2025-01-03 05:50:38",
                "ALARM_TIME": "2025-01-03 05:50:33",
                "ALARM_CODE": "E200003301",
                "ALARM_EXT_DATA": '[{"code":"x","name":"n1","value":"1"}]',
                "IS_FALS": None,
                "IS_REST": None,
                "REST_TIME": None,
                "ALARM_NAME": "终�??掉电",
                "ALARM_LEVEL_CODE": "1",
                "ALARM_TYPE_CODE": "8",
                "ALARM_SOUR_CODE": "2",
                "ALARM_CLASS": "2",
                "ALARM_CREAT_WAY": "1",
                "IS_DISP": "0",
                "PROC_PERS": None,
                "TERMINAL_TYPE": "26",
                "WORK_ORDER_MODE": None,
                "ORDER_ID": None,
                "ORDER_STATE": None,
                "LOAD_TIME": "2025-01-03 05:50:39",
                "STATUTE_ALARM_CODE": None,
                "DATA_PART": 20250103,
                "AREA_CODE": "031300",
                "SYNC_TIME": "2025-01-03 05:50:36",
                "HOLO_DATA_SOURCE": "SS",
            },
        ]
    )

    transformed = transform_alarm_frame(source)

    assert transformed.columns.tolist() == [
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
    assert transformed.loc[0, "event_id"] == "a1"
    assert transformed.loc[0, "event_type"] == "COMMUNICATION_OVERFLOW"
    assert transformed.loc[1, "event_type"] == "TERMINAL_POWER_OFF"
    assert transformed.loc[1, "business_domain"] == "TERMINAL_STATUS"
    assert transformed.loc[1, "source_service"] == "SS_2_26"
    assert transformed.loc[1, "user_level"] == "CRITICAL"
    assert str(transformed.loc[0, "timestamp"]) == "2025-01-03 05:49:38"
    assert transformed.loc[0, "queue_wait_time"] == 1.8
    assert transformed.loc[1, "deadline"] == 5.0
    assert transformed.loc[1, "retry_count"] == 0.0
    assert transformed.loc[1, "is_hot_data"] == 0


def test_transform_alarm_frame_respects_burst_window_seconds() -> None:
    source = pd.DataFrame(
        [
            {
                "ALARM_ID": "b1",
                "RUN_TERM_ID": "term_b",
                "RUN_METER_ID": "meter_b",
                "MEASURE_POINT_NO": 1,
                "PRE_RECE_TIME": "2025-01-03 05:49:38",
                "ALARM_TIME": "2025-01-03 05:49:33",
                "ALARM_CODE": "E1",
                "ALARM_EXT_DATA": '[{"code":"x","name":"n1","value":"1"}]',
                "IS_FALS": None,
                "IS_REST": None,
                "REST_TIME": None,
                "ALARM_NAME": "A相CT二�?�侧开�?",
                "ALARM_LEVEL_CODE": "1",
                "ALARM_TYPE_CODE": "12",
                "ALARM_SOUR_CODE": "1",
                "ALARM_CLASS": "1",
                "ALARM_CREAT_WAY": "1",
                "IS_DISP": "0",
                "PROC_PERS": None,
                "TERMINAL_TYPE": "26",
                "WORK_ORDER_MODE": None,
                "ORDER_ID": None,
                "ORDER_STATE": None,
                "LOAD_TIME": "2025-01-03 05:49:39",
                "STATUTE_ALARM_CODE": None,
                "DATA_PART": 20250103,
                "AREA_CODE": "031300",
                "SYNC_TIME": "2025-01-03 05:49:46",
                "HOLO_DATA_SOURCE": "SS",
            },
            {
                "ALARM_ID": "b2",
                "RUN_TERM_ID": "term_b",
                "RUN_METER_ID": "meter_b",
                "MEASURE_POINT_NO": 1,
                "PRE_RECE_TIME": "2025-01-03 05:50:08",
                "ALARM_TIME": "2025-01-03 05:50:03",
                "ALARM_CODE": "E1",
                "ALARM_EXT_DATA": '[{"code":"x","name":"n1","value":"1"}]',
                "IS_FALS": None,
                "IS_REST": None,
                "REST_TIME": None,
                "ALARM_NAME": "A相CT二�?�侧开�?",
                "ALARM_LEVEL_CODE": "1",
                "ALARM_TYPE_CODE": "12",
                "ALARM_SOUR_CODE": "1",
                "ALARM_CLASS": "1",
                "ALARM_CREAT_WAY": "1",
                "IS_DISP": "0",
                "PROC_PERS": None,
                "TERMINAL_TYPE": "26",
                "WORK_ORDER_MODE": None,
                "ORDER_ID": None,
                "ORDER_STATE": None,
                "LOAD_TIME": "2025-01-03 05:50:09",
                "STATUTE_ALARM_CODE": None,
                "DATA_PART": 20250103,
                "AREA_CODE": "031300",
                "SYNC_TIME": "2025-01-03 05:50:16",
                "HOLO_DATA_SOURCE": "SS",
            },
        ]
    )

    aggregated = transform_alarm_frame(source, burst_window_seconds=60)
    separated = transform_alarm_frame(source, burst_window_seconds=10)

    assert len(aggregated) == 1
    assert "::b2" in aggregated.loc[0, "event_id"]
    assert len(separated) == 2
