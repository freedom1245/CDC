from cdc_priority.scheduler.env import SchedulerEnv
from pathlib import Path
import shutil
import uuid

import pandas as pd

from cdc_priority.scheduler.evaluate import (
    export_policy_comparison,
    export_policy_comparison_figure,
    simulate_policy,
)
from cdc_priority.scheduler.event import CDCEvent
from cdc_priority.scheduler.reward import compute_reward
from cdc_priority.scheduler.policies import aging_policy, strict_priority_policy
from cdc_priority.scheduler.queue_manager import QueueManager


def _make_scheduler_temp_dir() -> Path:
    path = Path("outputs/scheduler") / f"pytest-temp-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_scheduler_env_reset() -> None:
    env = SchedulerEnv()
    state = env.reset()
    assert state.queue_length == 0
    assert state.priority_counts == {"high": 0, "medium": 0, "low": 0}


def test_strict_priority_policy_prefers_high_priority() -> None:
    queue_manager = QueueManager()
    queue_manager.push(CDCEvent("e1", "low", arrival_step=0, sync_cost=1.0))
    queue_manager.push(CDCEvent("e2", "high", arrival_step=1, sync_cost=1.0))

    event = strict_priority_policy(queue_manager)

    assert event is not None
    assert event.event_id == "e2"


def test_aging_policy_boosts_waiting_low_priority_event() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("low_old", "low", arrival_step=0, sync_cost=1.0, wait_steps=10)
    )
    queue_manager.push(
        CDCEvent("medium_new", "medium", arrival_step=1, sync_cost=1.0, wait_steps=1)
    )

    event = aging_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "low_old"


def test_aging_policy_still_prefers_high_priority_when_wait_is_similar() -> None:
    queue_manager = QueueManager()
    queue_manager.push(
        CDCEvent("low_waiting", "low", arrival_step=0, sync_cost=1.0, wait_steps=6)
    )
    queue_manager.push(
        CDCEvent("high_waiting", "high", arrival_step=1, sync_cost=1.0, wait_steps=6)
    )

    event = aging_policy(queue_manager, starvation_threshold=5)

    assert event is not None
    assert event.event_id == "high_waiting"


def test_simulate_policy_returns_metrics() -> None:
    events = [
        CDCEvent("e1", "low", arrival_step=0, sync_cost=1.0, service_steps=1),
        CDCEvent("e2", "high", arrival_step=1, sync_cost=1.0, service_steps=1),
        CDCEvent("e3", "medium", arrival_step=2, sync_cost=1.0, service_steps=1),
    ]

    metrics = simulate_policy(events, policy_name="strict_priority", starvation_threshold=5)

    assert metrics.completed_events == 3
    assert metrics.throughput > 0
    assert metrics.average_delay_steps >= 0


def test_export_policy_comparison_writes_csv() -> None:
    temp_dir = _make_scheduler_temp_dir()
    try:
        data_path = temp_dir / "events.csv"
        output_path = temp_dir / "policy_comparison.csv"
        frame = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3"],
                "timestamp": [
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:03",
                ],
                "priority_label": ["low", "high", "medium"],
                "estimated_sync_cost": [1.0, 1.0, 1.0],
                "deadline": [3.0, 2.0, 4.0],
            }
        )
        frame.to_csv(data_path, index=False)

        comparison = export_policy_comparison(
            data_path,
            output_path,
            starvation_threshold=5,
        )

        assert output_path.exists()
        assert set(comparison["policy"]) == {"fifo", "strict_priority", "aging"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_policy_comparison_figure_writes_image() -> None:
    temp_dir = _make_scheduler_temp_dir()
    try:
        csv_path = temp_dir / "policy_comparison.csv"
        figure_path = temp_dir / "policy_comparison.png"
        pd.DataFrame(
            {
                "policy": ["fifo", "strict_priority", "aging"],
                "throughput": [0.1, 0.1, 0.1],
                "average_delay_steps": [100, 80, 90],
                "high_priority_average_delay_steps": [100, 10, 20],
                "max_low_priority_wait_steps": [200, 300, 250],
                "fairness_index": [1.0, 0.5, 0.6],
                "completed_events": [3, 3, 3],
            }
        ).to_csv(csv_path, index=False)

        output = export_policy_comparison_figure(csv_path, figure_path)

        assert output.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_compute_reward_rewards_reducing_low_wait() -> None:
    before = SchedulerEnv().reset()
    before.max_low_priority_wait_steps = 20
    after = SchedulerEnv().reset()
    after.max_low_priority_wait_steps = 5

    reward = compute_reward(
        previous_state=before,
        state=after,
        processed_priority="low",
        processed_delay_steps=2,
        deadline_missed=False,
        reward_weights={
            "high_priority_throughput": 1.0,
            "average_delay_penalty": 0.05,
            "starvation_penalty": 3.0,
            "deadline_miss_penalty": 2.0,
        },
    )

    assert reward > 0
