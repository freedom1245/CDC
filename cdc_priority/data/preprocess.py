import pandas as pd


def preprocess_events(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].fillna("UNKNOWN").astype(str).str.strip()

    numeric_like_columns = [
        "record_size",
        "estimated_sync_cost",
        "dependency_count",
        "queue_wait_time",
        "deadline",
        "retry_count",
        "business_value",
        "source_load",
    ]
    for column in numeric_like_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0)

    if "timestamp" in cleaned.columns:
        timestamp = pd.to_datetime(cleaned["timestamp"], errors="coerce")
        cleaned["timestamp"] = timestamp
        cleaned["event_hour"] = timestamp.dt.hour.fillna(0).astype("int64")
        cleaned["is_peak_hour"] = cleaned["event_hour"].between(9, 18).astype("int64")

    if "queue_wait_time" in cleaned.columns:
        cleaned["queue_wait_time"] = cleaned["queue_wait_time"].clip(lower=0)

    if "deadline" in cleaned.columns and "queue_wait_time" in cleaned.columns:
        cleaned["deadline_gap"] = (cleaned["deadline"] - cleaned["queue_wait_time"]).clip(
            lower=0
        )

    return cleaned
