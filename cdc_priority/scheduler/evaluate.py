from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..settings import default_settings
from ..utils import to_project_relative_path
from .env import SchedulerEnv
from .event import CDCEvent

ARRIVAL_STEP_TIME_UNIT_SECONDS = 1.0
COMPARISON_CACHE_VERSION = 6


@dataclass
class SchedulerMetrics:
    throughput: float
    average_delay_steps: float
    max_low_priority_wait_steps: float
    completed_events: int
    high_priority_average_delay_steps: float
    fairness_index: float
    high_priority_match_rate: float
    matched_high_priority_count: int
    expected_high_priority_count: int
    evaluated_priority_window_size: int
    timely_high_priority_match_rate: float
    timely_matched_high_priority_count: int
    timely_match_window_steps: int


def _build_arrival_steps(
    frame: pd.DataFrame,
    arrival_step_time_unit_seconds: float = ARRIVAL_STEP_TIME_UNIT_SECONDS,
) -> list[int]:
    if "timestamp" not in frame.columns:
        return list(range(len(frame)))

    parsed_timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if parsed_timestamps.isna().any():
        return list(range(len(frame)))

    base_timestamp = parsed_timestamps.iloc[0]
    delta_seconds = (parsed_timestamps - base_timestamp).dt.total_seconds()
    unit_seconds = max(float(arrival_step_time_unit_seconds), 1e-6)
    return (
        (delta_seconds / unit_seconds)
        .round()
        .clip(lower=0)
        .astype(int)
        .tolist()
    )


def load_scheduler_events(
    data_path: Path,
    arrival_step_time_unit_seconds: float = ARRIVAL_STEP_TIME_UNIT_SECONDS,
) -> list[CDCEvent]:
    frame = pd.read_csv(data_path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    arrival_steps = _build_arrival_steps(
        frame,
        arrival_step_time_unit_seconds=arrival_step_time_unit_seconds,
    )

    events: list[CDCEvent] = []
    for index, row in frame.iterrows():
        deadline_value = row["deadline"] if "deadline" in frame.columns else None
        deadline_step = None if pd.isna(deadline_value) else int(math.ceil(float(deadline_value)))
        service_steps = max(1, int(math.ceil(float(row.get("estimated_sync_cost", 1.0)))))
        arrival_hour = None
        if "timestamp" in frame.columns and pd.notna(row.get("timestamp")):
            arrival_hour = int(row["timestamp"].hour)
        events.append(
            CDCEvent(
                event_id=str(row.get("event_id", f"event_{index}")),
                priority=str(row.get("priority_label", "low")),
                arrival_step=arrival_steps[index],
                sync_cost=float(row.get("estimated_sync_cost", 1.0)),
                arrival_hour=arrival_hour,
                deadline_step=deadline_step,
                service_steps=service_steps,
            )
        )
    return events


def _select_action(policy_name: str, state) -> int:
    if policy_name == "fifo":
        return 0
    if policy_name == "strict_priority":
        return 1
    if policy_name == "aging":
        if state.priority_counts.get("high", 0) > 0:
            return 4
        return 2
    raise ValueError(f"Unsupported policy: {policy_name}")


def _jain_fairness(values: list[float]) -> float:
    filtered = [value for value in values if value >= 0]
    if not filtered:
        return 0.0
    numerator = sum(filtered) ** 2
    denominator = len(filtered) * sum(value * value for value in filtered)
    return 0.0 if denominator <= 1e-12 else numerator / denominator


def _high_priority_match_summary(
    events: list[CDCEvent],
    processed_event_ids: list[str],
) -> dict[str, float | int]:
    expected_high_ids = {event.event_id for event in events if event.priority == "high"}
    expected_high_count = len(expected_high_ids)
    if expected_high_count <= 0:
        return {
            "high_priority_match_rate": 1.0,
            "matched_high_priority_count": 0,
            "expected_high_priority_count": 0,
            "evaluated_priority_window_size": 0,
        }

    priority_window = processed_event_ids[:expected_high_count]
    matched_high_priority_count = sum(
        1 for event_id in priority_window if event_id in expected_high_ids
    )
    return {
        "high_priority_match_rate": matched_high_priority_count / expected_high_count,
        "matched_high_priority_count": matched_high_priority_count,
        "expected_high_priority_count": expected_high_count,
        "evaluated_priority_window_size": len(priority_window),
    }


def _timely_high_priority_match_summary(
    events: list[CDCEvent],
    processed_high_delays: dict[str, int],
    timely_match_window_steps: int,
) -> dict[str, float | int]:
    expected_high_ids = {event.event_id for event in events if event.priority == "high"}
    expected_high_count = len(expected_high_ids)
    if expected_high_count <= 0:
        return {
            "timely_high_priority_match_rate": 1.0,
            "timely_matched_high_priority_count": 0,
            "timely_match_window_steps": timely_match_window_steps,
        }

    timely_matched_high_priority_count = sum(
        1
        for event_id in expected_high_ids
        if processed_high_delays.get(event_id, timely_match_window_steps + 1)
        <= timely_match_window_steps
    )
    return {
        "timely_high_priority_match_rate": timely_matched_high_priority_count / expected_high_count,
        "timely_matched_high_priority_count": timely_matched_high_priority_count,
        "timely_match_window_steps": timely_match_window_steps,
    }


def simulate_policy(
    events: list[CDCEvent],
    policy_name: str,
    starvation_threshold: int = 5,
    env_kwargs: dict[str, object] | None = None,
    timely_match_window_steps: int = 10,
) -> SchedulerMetrics:
    merged_env_kwargs = {"starvation_threshold": starvation_threshold}
    if env_kwargs:
        merged_env_kwargs.update(env_kwargs)
    env = SchedulerEnv(
        events=events,
        **merged_env_kwargs,
    )
    state = env.reset()
    completed = 0
    delay_totals: list[int] = []
    high_delay_totals: list[int] = []
    per_priority_delay: dict[str, list[int]] = {"high": [], "medium": [], "low": []}
    processed_event_ids: list[str] = []
    processed_high_delays: dict[str, int] = {}

    evaluation_budget = max(len(events) * 50, 10000)
    for _ in range(evaluation_budget):
        action = _select_action(policy_name, state)
        state, _, done, info = env.step(action)
        priority = info.get("processed_priority")
        if priority is not None:
            delay = int(info.get("processed_delay_steps", 0))
            delay_totals.append(delay)
            per_priority_delay[priority].append(delay)
            processed_event_id = info.get("processed_event_id")
            if processed_event_id is not None:
                processed_event_id = str(processed_event_id)
                processed_event_ids.append(processed_event_id)
                if priority == "high":
                    processed_high_delays[processed_event_id] = delay
            if priority == "high":
                high_delay_totals.append(delay)
            completed += 1
        if done:
            break

    average_delay_steps = sum(delay_totals) / max(len(delay_totals), 1)
    high_average_delay_steps = sum(high_delay_totals) / max(len(high_delay_totals), 1)
    max_low_priority_wait_steps = max(per_priority_delay["low"], default=0)
    fairness_index = _jain_fairness(
        [
            sum(per_priority_delay["high"]) / max(len(per_priority_delay["high"]), 1),
            sum(per_priority_delay["medium"]) / max(len(per_priority_delay["medium"]), 1),
            sum(per_priority_delay["low"]) / max(len(per_priority_delay["low"]), 1),
        ]
    )
    throughput = completed / max(env.current_step, 1)
    match_summary = _high_priority_match_summary(events, processed_event_ids)
    timely_match_summary = _timely_high_priority_match_summary(
        events,
        processed_high_delays,
        timely_match_window_steps=timely_match_window_steps,
    )
    return SchedulerMetrics(
        throughput=throughput,
        average_delay_steps=average_delay_steps,
        max_low_priority_wait_steps=max_low_priority_wait_steps,
        completed_events=completed,
        high_priority_average_delay_steps=high_average_delay_steps,
        fairness_index=fairness_index,
        high_priority_match_rate=float(match_summary["high_priority_match_rate"]),
        matched_high_priority_count=int(match_summary["matched_high_priority_count"]),
        expected_high_priority_count=int(match_summary["expected_high_priority_count"]),
        evaluated_priority_window_size=int(match_summary["evaluated_priority_window_size"]),
        timely_high_priority_match_rate=float(
            timely_match_summary["timely_high_priority_match_rate"]
        ),
        timely_matched_high_priority_count=int(
            timely_match_summary["timely_matched_high_priority_count"]
        ),
        timely_match_window_steps=int(timely_match_summary["timely_match_window_steps"]),
    )


def compare_policies(
    events: list[CDCEvent],
    starvation_threshold: int = 5,
    env_kwargs: dict[str, object] | None = None,
    timely_match_window_steps: int = 10,
) -> pd.DataFrame:
    rows = []
    for policy_name in ("fifo", "strict_priority", "aging"):
        metrics = simulate_policy(
            events,
            policy_name=policy_name,
            starvation_threshold=starvation_threshold,
            env_kwargs=env_kwargs,
            timely_match_window_steps=timely_match_window_steps,
        )
        rows.append(
            {
                "policy": policy_name,
                "throughput": metrics.throughput,
                "average_delay_steps": metrics.average_delay_steps,
                "high_priority_average_delay_steps": metrics.high_priority_average_delay_steps,
                "max_low_priority_wait_steps": metrics.max_low_priority_wait_steps,
                "fairness_index": metrics.fairness_index,
                "high_priority_match_rate": metrics.high_priority_match_rate,
                "matched_high_priority_count": metrics.matched_high_priority_count,
                "expected_high_priority_count": metrics.expected_high_priority_count,
                "evaluated_priority_window_size": metrics.evaluated_priority_window_size,
                "timely_high_priority_match_rate": metrics.timely_high_priority_match_rate,
                "timely_matched_high_priority_count": metrics.timely_matched_high_priority_count,
                "timely_match_window_steps": metrics.timely_match_window_steps,
                "completed_events": metrics.completed_events,
            }
        )
    return pd.DataFrame(rows)


def _cache_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def _is_comparison_cache_valid(
    data_path: Path,
    output_path: Path,
    starvation_threshold: int,
    env_kwargs: dict[str, object] | None = None,
    timely_match_window_steps: int = 10,
    arrival_step_time_unit_seconds: float = ARRIVAL_STEP_TIME_UNIT_SECONDS,
) -> bool:
    metadata_path = _cache_metadata_path(output_path)
    if not output_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    project_root = default_settings().project_root
    return (
        metadata.get("cache_version") == COMPARISON_CACHE_VERSION
        and metadata.get("data_path") == to_project_relative_path(data_path.resolve(), project_root)
        and metadata.get("starvation_threshold") == starvation_threshold
        and metadata.get("env_kwargs") == (env_kwargs or {})
        and metadata.get("timely_match_window_steps") == timely_match_window_steps
        and metadata.get("arrival_step_time_unit_seconds")
        == float(arrival_step_time_unit_seconds)
        and metadata.get("data_mtime_ns") == data_path.stat().st_mtime_ns
    )


def _write_comparison_cache_metadata(
    data_path: Path,
    output_path: Path,
    starvation_threshold: int,
    env_kwargs: dict[str, object] | None = None,
    timely_match_window_steps: int = 10,
    arrival_step_time_unit_seconds: float = ARRIVAL_STEP_TIME_UNIT_SECONDS,
) -> None:
    metadata_path = _cache_metadata_path(output_path)
    project_root = default_settings().project_root
    metadata = {
        "cache_version": COMPARISON_CACHE_VERSION,
        "data_path": to_project_relative_path(data_path.resolve(), project_root),
        "starvation_threshold": starvation_threshold,
        "env_kwargs": env_kwargs or {},
        "timely_match_window_steps": timely_match_window_steps,
        "arrival_step_time_unit_seconds": float(arrival_step_time_unit_seconds),
        "data_mtime_ns": data_path.stat().st_mtime_ns,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def export_policy_comparison(
    data_path: Path,
    output_path: Path,
    starvation_threshold: int = 5,
    env_kwargs: dict[str, object] | None = None,
    timely_match_window_steps: int = 10,
    arrival_step_time_unit_seconds: float = ARRIVAL_STEP_TIME_UNIT_SECONDS,
) -> pd.DataFrame:
    if _is_comparison_cache_valid(
        data_path,
        output_path,
        starvation_threshold,
        env_kwargs=env_kwargs,
        timely_match_window_steps=timely_match_window_steps,
        arrival_step_time_unit_seconds=arrival_step_time_unit_seconds,
    ):
        return pd.read_csv(output_path)

    events = load_scheduler_events(
        data_path,
        arrival_step_time_unit_seconds=arrival_step_time_unit_seconds,
    )
    comparison = compare_policies(
        events,
        starvation_threshold=starvation_threshold,
        env_kwargs=env_kwargs,
        timely_match_window_steps=timely_match_window_steps,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    _write_comparison_cache_metadata(
        data_path,
        output_path,
        starvation_threshold,
        env_kwargs=env_kwargs,
        timely_match_window_steps=timely_match_window_steps,
        arrival_step_time_unit_seconds=arrival_step_time_unit_seconds,
    )
    return comparison


def export_policy_comparison_figure(
    comparison_csv_path: Path,
    output_path: Path,
) -> Path:
    comparison = pd.read_csv(comparison_csv_path)
    metrics = [
        ("throughput", "Throughput"),
        ("average_delay_steps", "Average Delay (steps)"),
        ("high_priority_average_delay_steps", "High-Priority Delay (steps)"),
        ("fairness_index", "Fairness Index"),
        ("high_priority_match_rate", "High-Priority Match Rate"),
        ("timely_high_priority_match_rate", "Timed High-Priority Match Rate"),
    ]
    colors = {
        "fifo": "#3B82F6",
        "strict_priority": "#EF4444",
        "aging": "#10B981",
        "dqn": "#F59E0B",
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    policies = comparison["policy"].tolist()

    for axis, (metric_key, title) in zip(axes, metrics):
        values = comparison[metric_key].tolist()
        axis.bar(
            policies,
            values,
            color=[colors.get(policy, "#6B7280") for policy in policies],
        )
        axis.set_title(title)
        axis.set_xlabel("Policy")
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    for axis in axes[len(metrics) :]:
        axis.axis("off")

    fig.suptitle("Scheduler Policy Comparison", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def append_policy_result(
    comparison_csv_path: Path,
    row: dict[str, object],
) -> pd.DataFrame:
    comparison = pd.read_csv(comparison_csv_path)
    comparison = comparison[comparison["policy"] != row["policy"]].copy()
    comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
    comparison.to_csv(comparison_csv_path, index=False)
    return comparison
