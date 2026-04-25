import pandas as pd


DEFAULT_LABELING_CONFIG = {
    "priority_score": {
        "numeric_weights": {
            "business_value": 0.30,
            "queue_wait_time": 0.20,
            "dependency_count": 0.15,
            "estimated_sync_cost": 0.10,
            "retry_count": 0.10,
        },
        "invert_numeric": ["estimated_sync_cost"],
        "categorical_weights": {
            "event_type": 0.10,
            "business_domain": 0.05,
        },
        "hot_values": {
            "event_type": ["DELETE", "PAYMENT", "ALERT"],
            "business_domain": ["PAYMENT", "ORDER", "RISK", "INVENTORY"],
        },
        "thresholds": {
            "medium": 0.40,
            "high": 0.70,
        },
    }
}


def _normalized_flag(frame: pd.DataFrame, column: str, hot_values: set[str]) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = frame[column].astype(str).str.upper()
    return values.isin(hot_values).astype("float32")


def _normalized_numeric(frame: pd.DataFrame, column: str, invert: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype("float32")
    max_value = float(values.max())
    min_value = float(values.min())
    if max_value - min_value <= 1e-9:
        normalized = pd.Series(0.0, index=frame.index, dtype="float32")
    else:
        normalized = (values - min_value) / (max_value - min_value)
    return 1.0 - normalized if invert else normalized


def _merged_labeling_config(labeling_config: dict | None) -> dict:
    merged = {
        "priority_score": {
            "numeric_weights": dict(
                DEFAULT_LABELING_CONFIG["priority_score"]["numeric_weights"]
            ),
            "invert_numeric": list(
                DEFAULT_LABELING_CONFIG["priority_score"]["invert_numeric"]
            ),
            "categorical_weights": dict(
                DEFAULT_LABELING_CONFIG["priority_score"]["categorical_weights"]
            ),
            "hot_values": {
                key: list(values)
                for key, values in DEFAULT_LABELING_CONFIG["priority_score"][
                    "hot_values"
                ].items()
            },
            "thresholds": dict(DEFAULT_LABELING_CONFIG["priority_score"]["thresholds"]),
        }
    }
    if not labeling_config:
        return merged

    priority_score = labeling_config.get("priority_score", {})
    for key in ("numeric_weights", "categorical_weights", "thresholds"):
        merged["priority_score"][key].update(priority_score.get(key, {}))
    if "invert_numeric" in priority_score:
        merged["priority_score"]["invert_numeric"] = list(priority_score["invert_numeric"])
    if "hot_values" in priority_score:
        for column, values in priority_score["hot_values"].items():
            merged["priority_score"]["hot_values"][column] = list(values)
    return merged


def build_priority_score(frame: pd.DataFrame, labeling_config: dict | None = None) -> pd.Series:
    config = _merged_labeling_config(labeling_config)["priority_score"]
    score = pd.Series(0.0, index=frame.index, dtype="float32")

    for column, weight in config["numeric_weights"].items():
        score += float(weight) * _normalized_numeric(
            frame,
            column,
            invert=column in set(config["invert_numeric"]),
        )

    for column, weight in config["categorical_weights"].items():
        hot_values = set(str(value).upper() for value in config["hot_values"].get(column, []))
        score += float(weight) * _normalized_flag(frame, column, hot_values)

    return score


def attach_priority_label(frame: pd.DataFrame, labeling_config: dict | None = None) -> pd.DataFrame:
    labeled = frame.copy()
    if "priority_label" in labeled.columns:
        labeled["priority_label"] = (
            labeled["priority_label"].fillna("medium").astype(str).str.lower()
        )
        return labeled

    config = _merged_labeling_config(labeling_config)["priority_score"]
    thresholds = config["thresholds"]
    priority_score = build_priority_score(labeled, labeling_config=labeling_config)
    labeled["priority_score"] = priority_score
    labeled["priority_label"] = "low"
    labeled.loc[priority_score >= float(thresholds["medium"]), "priority_label"] = "medium"
    labeled.loc[priority_score >= float(thresholds["high"]), "priority_label"] = "high"
    return labeled
