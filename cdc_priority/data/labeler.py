import math

import pandas as pd


DEFAULT_LABELING_CONFIG = {
    "priority_score": {
        "components": {
            "business_importance": {
                "weight": 0.40,
                "mode": "linear",
                "numeric_weights": {
                    "business_value": 0.75,
                },
                "categorical_weights": {
                    "business_domain": 0.15,
                    "user_level": 0.10,
                },
                "invert_numeric": [],
                "hot_values": {
                    "business_domain": ["PAYMENT", "ORDER", "RISK", "INVENTORY"],
                    "user_level": ["VIP", "INTERNAL"],
                },
            },
            "timeliness": {
                "weight": 0.30,
                "mode": "linear",
                "numeric_weights": {
                    "queue_wait_time": 0.45,
                    "deadline_gap": 0.35,
                    "retry_count": 0.10,
                    "is_peak_hour": 0.10,
                },
                "categorical_weights": {},
                "invert_numeric": ["deadline_gap"],
                "hot_values": {},
            },
            "dependency_impact": {
                "weight": 0.20,
                "mode": "linear",
                "numeric_weights": {
                    "dependency_count": 0.60,
                    "consistency_risk": 0.40,
                },
                "categorical_weights": {
                    "event_type": 1.0,
                },
                "invert_numeric": [],
                "hot_values": {
                    "event_type": ["DELETE", "PAYMENT", "ALERT"],
                },
            },
            "execution_feasibility": {
                "weight": 0.10,
                "mode": "linear",
                "numeric_weights": {
                    "estimated_sync_cost": 0.70,
                    "source_load": 0.15,
                    "db_load": 0.15,
                },
                "categorical_weights": {},
                "invert_numeric": ["estimated_sync_cost", "source_load", "db_load"],
                "hot_values": {},
            },
        },
        "thresholds": {
            "strategy": "fixed",
            "medium": 0.40,
            "high": 0.70,
        },
    },
    "impact_score": {
        "components": {
            "deadline_miss_impact": {
                "weight": 0.35,
                "mode": "deadline_risk",
                "params": {
                    "pressure_threshold": 0.85,
                    "sigmoid_scale": 6.0,
                    "retry_boost": 0.25,
                    "peak_hour_boost": 0.10,
                    "load_boost": 0.20,
                },
            },
            "delay_cost_impact": {
                "weight": 0.25,
                "mode": "delay_cost",
                "params": {
                    "delay_threshold": 0.45,
                    "delay_scale": 5.0,
                    "business_value_boost": 0.30,
                },
            },
            "consistency_impact": {
                "weight": 0.20,
                "mode": "consistency_risk",
                "numeric_weights": {
                    "consistency_risk": 0.45,
                    "dependency_count": 0.35,
                },
                "categorical_weights": {
                    "event_type": 0.10,
                    "table_name": 0.05,
                    "source_service": 0.05,
                },
                "hot_values": {
                    "event_type": ["PURCHASE", "CART"],
                    "table_name": ["ORDERS", "PAYMENTS", "INVENTORY"],
                },
                "interactions": [
                    {
                        "columns": ["event_type", "table_name"],
                        "hot_values": {
                            "event_type": ["PURCHASE", "CART"],
                            "table_name": ["ORDERS", "PAYMENTS"],
                        },
                        "weight": 0.15,
                    }
                ],
            },
            "starvation_externality": {
                "weight": 0.10,
                "mode": "fairness_cost",
                "hot_values": {
                    "business_domain": ["ARCHIVE", "ANALYTICS"],
                    "user_level": ["NORMAL", "GUEST"],
                },
                "params": {
                    "low_priority_boost": 0.50,
                    "peak_hour_boost": 0.20,
                },
            },
            "throughput_value": {
                "weight": 0.10,
                "mode": "business_value",
                "numeric_weights": {
                    "business_value": 0.70,
                },
                "categorical_weights": {
                    "user_level": 0.20,
                    "business_domain": 0.10,
                },
                "hot_values": {
                    "user_level": ["VIP", "INTERNAL"],
                    "business_domain": ["ELECTRONICS", "COMPUTERS", "APPLIANCES"],
                },
            },
        },
        "thresholds": {
            "strategy": "quantile",
            "low_high_quantiles": [0.33, 0.66],
        },
    },
}


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0).astype("float32")


def _component_template() -> dict[str, object]:
    return {
        "weight": 0.0,
        "mode": "linear",
        "numeric_weights": {},
        "categorical_weights": {},
        "invert_numeric": [],
        "hot_values": {},
        "params": {},
        "interactions": [],
    }


def _normalized_flag(frame: pd.DataFrame, column: str, hot_values: set[str]) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    values = frame[column].astype(str).str.upper()
    return values.isin(hot_values).astype("float32")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype("float32")


def _normalized_numeric(frame: pd.DataFrame, column: str, invert: bool = False) -> pd.Series:
    values = _numeric_series(frame, column)
    max_value = float(values.max())
    min_value = float(values.min())
    if max_value - min_value <= 1e-9:
        normalized = pd.Series(0.0, index=frame.index, dtype="float32")
    else:
        normalized = (values - min_value) / (max_value - min_value)
    normalized = normalized.astype("float32")
    return 1.0 - normalized if invert else normalized


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series | float,
    eps: float = 1e-6,
) -> pd.Series:
    if isinstance(denominator, pd.Series):
        denominator_series = denominator.astype("float32").clip(lower=eps)
    else:
        denominator_series = pd.Series(float(denominator), index=numerator.index, dtype="float32")
        denominator_series = denominator_series.clip(lower=eps)
    return (numerator.astype("float32") / denominator_series).astype("float32")


def _sigmoid(series: pd.Series, scale: float = 1.0, shift: float = 0.0) -> pd.Series:
    logits = scale * (series.astype("float32") - float(shift))
    return logits.map(lambda value: 1.0 / (1.0 + math.exp(-float(value)))).astype("float32")


def _merge_component_config(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = {
        "weight": float(base.get("weight", 0.0)),
        "mode": str(base.get("mode", "linear")),
        "numeric_weights": dict(base.get("numeric_weights", {})),
        "categorical_weights": dict(base.get("categorical_weights", {})),
        "invert_numeric": list(base.get("invert_numeric", [])),
        "hot_values": {
            key: list(values)
            for key, values in base.get("hot_values", {}).items()
        },
        "params": dict(base.get("params", {})),
        "interactions": [dict(item) for item in base.get("interactions", [])],
    }
    if "weight" in override:
        merged["weight"] = float(override["weight"])
    if "mode" in override:
        merged["mode"] = str(override["mode"])
    for key in ("numeric_weights", "categorical_weights", "params"):
        merged[key].update(override.get(key, {}))
    if "invert_numeric" in override:
        merged["invert_numeric"] = list(override["invert_numeric"])
    if "hot_values" in override:
        for column, values in override["hot_values"].items():
            merged["hot_values"][column] = list(values)
    if "interactions" in override:
        merged["interactions"] = [dict(item) for item in override["interactions"]]
    return merged


def _merge_score_config(score_name: str, override: dict | None) -> dict[str, object]:
    default_score = DEFAULT_LABELING_CONFIG[score_name]
    merged = {
        "components": {
            component_name: _merge_component_config(_component_template(), component)
            for component_name, component in default_score["components"].items()
        },
        "thresholds": dict(default_score.get("thresholds", {})),
    }
    if not override:
        return merged

    if "thresholds" in override:
        merged["thresholds"].update(override["thresholds"])

    components = override.get("components")
    if components:
        for component_name, component in components.items():
            target = merged["components"].get(component_name, _component_template())
            merged["components"][component_name] = _merge_component_config(target, component)
        return merged

    legacy_component = merged["components"].get("legacy_priority", _component_template())
    legacy_component["weight"] = float(override.get("weight", legacy_component.get("weight", 1.0) or 1.0))
    for key in ("numeric_weights", "categorical_weights"):
        legacy_component[key].update(override.get(key, {}))
    if "invert_numeric" in override:
        legacy_component["invert_numeric"] = list(override["invert_numeric"])
    if "hot_values" in override:
        for column, values in override["hot_values"].items():
            legacy_component["hot_values"][column] = list(values)
    merged["components"]["legacy_priority"] = legacy_component
    return merged


def _merged_labeling_config(labeling_config: dict | None) -> dict[str, object]:
    config = labeling_config or {}
    return {
        "priority_score": _merge_score_config("priority_score", config.get("priority_score")),
        "impact_score": _merge_score_config("impact_score", config.get("impact_score")),
    }


def _resolve_labeling_spec(labeling_config: dict | None) -> tuple[str, dict[str, object]]:
    merged = _merged_labeling_config(labeling_config)
    if labeling_config and "impact_score" in labeling_config:
        return "impact_score", merged["impact_score"]
    if labeling_config and "priority_score" in labeling_config:
        return "priority_score", merged["priority_score"]
    return "priority_score", merged["priority_score"]


def _build_linear_component(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype="float32")
    total_weight = 0.0
    invert_numeric = set(component.get("invert_numeric", []))

    for column, weight in component.get("numeric_weights", {}).items():
        weight_value = float(weight)
        score += weight_value * _normalized_numeric(
            frame,
            column,
            invert=column in invert_numeric,
        )
        total_weight += weight_value

    for column, weight in component.get("categorical_weights", {}).items():
        weight_value = float(weight)
        hot_values = set(str(value).upper() for value in component.get("hot_values", {}).get(column, []))
        score += weight_value * _normalized_flag(frame, column, hot_values)
        total_weight += weight_value

    if total_weight <= 1e-9:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    return _clip01(score / total_weight)


def _build_deadline_risk_component(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    params = component.get("params", {})
    pressure_threshold = float(params.get("pressure_threshold", 0.85))
    sigmoid_scale = float(params.get("sigmoid_scale", 6.0))
    retry_boost = float(params.get("retry_boost", 0.25))
    peak_hour_boost = float(params.get("peak_hour_boost", 0.10))
    load_boost = float(params.get("load_boost", 0.20))

    deadline = _numeric_series(frame, "deadline")
    queue_wait_time = _numeric_series(frame, "queue_wait_time")
    estimated_sync_cost = _numeric_series(frame, "estimated_sync_cost")
    retry_count = _normalized_numeric(frame, "retry_count")
    is_peak_hour = _numeric_series(frame, "is_peak_hour").clip(lower=0.0, upper=1.0)
    source_load = _normalized_numeric(frame, "source_load")
    db_load = _normalized_numeric(frame, "db_load")

    load_factor = 1.0 + load_boost * (source_load + db_load)
    deadline_pressure = _safe_divide(
        queue_wait_time + estimated_sync_cost * load_factor,
        deadline.clip(lower=1e-6),
    )
    risk = _sigmoid(deadline_pressure, scale=sigmoid_scale, shift=pressure_threshold)
    risk *= 1.0 + retry_boost * retry_count
    risk *= 1.0 + peak_hour_boost * is_peak_hour
    return _clip01(risk)


def _build_delay_cost_component(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    params = component.get("params", {})
    delay_threshold = float(params.get("delay_threshold", 0.45))
    delay_scale = float(params.get("delay_scale", 5.0))
    business_value_boost = float(params.get("business_value_boost", 0.30))

    queue_wait_time = _normalized_numeric(frame, "queue_wait_time")
    source_load = _normalized_numeric(frame, "source_load")
    db_load = _normalized_numeric(frame, "db_load")
    business_value = _normalized_numeric(frame, "business_value")

    delay_load = queue_wait_time * (1.0 + source_load + db_load)
    impact = _sigmoid(delay_load, scale=delay_scale, shift=delay_threshold)
    impact *= 1.0 + business_value_boost * business_value
    return _clip01(impact)


def _build_interaction_bonus(frame: pd.DataFrame, interactions: list[dict[str, object]]) -> pd.Series:
    bonus = pd.Series(0.0, index=frame.index, dtype="float32")
    total_weight = 0.0
    for interaction in interactions:
        columns = list(interaction.get("columns", []))
        if not columns:
            continue
        active = pd.Series(1.0, index=frame.index, dtype="float32")
        hot_values = interaction.get("hot_values", {})
        for column in columns:
            values = set(str(value).upper() for value in hot_values.get(column, []))
            active *= _normalized_flag(frame, column, values)
        weight = float(interaction.get("weight", 0.0))
        bonus += weight * active
        total_weight += weight
    if total_weight <= 1e-9:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    return bonus / total_weight


def _build_consistency_risk_component(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    base = _build_linear_component(frame, component)
    interaction_bonus = _build_interaction_bonus(frame, list(component.get("interactions", [])))
    dependency_pressure = _normalized_numeric(frame, "dependency_count")
    return _clip01(base + 0.20 * interaction_bonus * dependency_pressure)


def _build_fairness_cost_component(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    params = component.get("params", {})
    low_priority_boost = float(params.get("low_priority_boost", 0.50))
    peak_hour_boost = float(params.get("peak_hour_boost", 0.20))

    queue_wait_time = _normalized_numeric(frame, "queue_wait_time")
    is_peak_hour = _numeric_series(frame, "is_peak_hour").clip(lower=0.0, upper=1.0)

    group_risk = pd.Series(0.0, index=frame.index, dtype="float32")
    group_columns = list(component.get("hot_values", {}).keys())
    if group_columns:
        for column in group_columns:
            hot_values = set(str(value).upper() for value in component.get("hot_values", {}).get(column, []))
            group_risk += _normalized_flag(frame, column, hot_values)
        group_risk = group_risk / len(group_columns)

    impact = queue_wait_time * (
        1.0
        + low_priority_boost * group_risk
        + peak_hour_boost * is_peak_hour
    )
    return _clip01(impact)


def _build_business_value_component(frame: pd.DataFrame, component: dict[str, object]) -> pd.Series:
    return _build_linear_component(frame, component)


def _build_component(frame: pd.DataFrame, component_name: str, component: dict[str, object]) -> pd.Series:
    mode = str(component.get("mode", "linear"))
    if mode == "linear":
        return _build_linear_component(frame, component)
    if mode == "deadline_risk":
        return _build_deadline_risk_component(frame, component)
    if mode == "delay_cost":
        return _build_delay_cost_component(frame, component)
    if mode == "consistency_risk":
        return _build_consistency_risk_component(frame, component)
    if mode == "fairness_cost":
        return _build_fairness_cost_component(frame, component)
    if mode == "business_value":
        return _build_business_value_component(frame, component)
    raise ValueError(f"Unsupported component mode for {component_name}: {mode}")


def build_priority_score(frame: pd.DataFrame, labeling_config: dict | None = None) -> pd.Series:
    _, config = _resolve_labeling_spec(labeling_config)
    score = pd.Series(0.0, index=frame.index, dtype="float32")
    total_component_weight = 0.0

    for component_name, component in config["components"].items():
        component_weight = float(component.get("weight", 0.0))
        if component_weight <= 0:
            continue
        score += component_weight * _build_component(frame, component_name, component)
        total_component_weight += component_weight

    if total_component_weight <= 1e-9:
        return pd.Series(0.0, index=frame.index, dtype="float32")
    return _clip01(score / total_component_weight)


def _resolve_label_thresholds(
    scores: pd.Series,
    thresholds: dict[str, object],
) -> tuple[float, float]:
    strategy = str(thresholds.get("strategy", "fixed")).lower()
    if strategy == "quantile":
        quantiles = thresholds.get("low_high_quantiles", [0.33, 0.66])
        if len(quantiles) != 2:
            raise ValueError("quantile thresholds must provide exactly two quantiles")
        medium_threshold = float(scores.quantile(float(quantiles[0])))
        high_threshold = float(scores.quantile(float(quantiles[1])))
        return medium_threshold, high_threshold
    return (
        float(thresholds["medium"]),
        float(thresholds["high"]),
    )


def attach_priority_label(frame: pd.DataFrame, labeling_config: dict | None = None) -> pd.DataFrame:
    labeled = frame.copy()
    if "priority_label" in labeled.columns:
        labeled["priority_label"] = (
            labeled["priority_label"].fillna("medium").astype(str).str.lower()
        )
        return labeled

    _, config = _resolve_labeling_spec(labeling_config)
    priority_score = build_priority_score(labeled, labeling_config=labeling_config)
    medium_threshold, high_threshold = _resolve_label_thresholds(
        priority_score,
        config["thresholds"],
    )
    labeled["priority_score"] = priority_score
    labeled["priority_label"] = "low"
    labeled.loc[priority_score >= medium_threshold, "priority_label"] = "medium"
    labeled.loc[priority_score >= high_threshold, "priority_label"] = "high"
    return labeled
