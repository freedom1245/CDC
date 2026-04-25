from dataclasses import dataclass, field

from .event import CDCEvent
from .policies import aging_policy, fifo_policy, strict_priority_policy
from .queue_manager import QueueManager
from .reward import compute_reward


@dataclass
class SchedulerState:
    queue_length: int
    average_wait_steps: float
    priority_counts: dict[str, int]
    max_low_priority_wait_steps: int

    def to_vector(self) -> list[float]:
        return [
            float(self.queue_length),
            float(self.average_wait_steps),
            float(self.priority_counts.get("high", 0)),
            float(self.priority_counts.get("medium", 0)),
            float(self.priority_counts.get("low", 0)),
            float(self.max_low_priority_wait_steps),
        ]


@dataclass
class SchedulerEnv:
    events: list[CDCEvent] = field(default_factory=list)
    queue_manager: QueueManager = field(default_factory=QueueManager)
    reward_weights: dict[str, float] = field(default_factory=dict)
    starvation_threshold: int = 5
    current_step: int = 0
    next_event_index: int = 0

    def reset(self) -> SchedulerState:
        self.queue_manager = QueueManager()
        self.current_step = 0
        self.next_event_index = 0
        self._enqueue_arrivals()
        return self.get_state()

    def _enqueue_arrivals(self) -> None:
        while (
            self.next_event_index < len(self.events)
            and self.events[self.next_event_index].arrival_step <= self.current_step
        ):
            source = self.events[self.next_event_index]
            self.queue_manager.push(
                CDCEvent(
                    event_id=source.event_id,
                    priority=source.priority,
                    arrival_step=source.arrival_step,
                    sync_cost=source.sync_cost,
                    deadline_step=source.deadline_step,
                    wait_steps=0,
                    service_steps=source.service_steps,
                )
            )
            self.next_event_index += 1

    def _select_event(self, action: int) -> CDCEvent | None:
        if action == 0:
            return fifo_policy(self.queue_manager)
        if action == 1:
            return strict_priority_policy(self.queue_manager)
        if action == 2:
            return aging_policy(
                self.queue_manager,
                starvation_threshold=self.starvation_threshold,
            )
        raise ValueError(f"Unsupported action: {action}")

    def get_state(self) -> SchedulerState:
        low_waits = [
            event.wait_steps for event in self.queue_manager.events if event.priority == "low"
        ]
        return SchedulerState(
            queue_length=len(self.queue_manager),
            average_wait_steps=self.queue_manager.average_wait_steps(),
            priority_counts=self.queue_manager.priority_counts(),
            max_low_priority_wait_steps=max(low_waits, default=0),
        )

    def step(self, action: int) -> tuple[SchedulerState, float, bool, dict]:
        if len(self.queue_manager) == 0 and self.next_event_index < len(self.events):
            self.current_step = max(
                self.current_step,
                self.events[self.next_event_index].arrival_step,
            )
            self._enqueue_arrivals()

        previous_state = self.get_state()
        processed_event = self._select_event(action) if len(self.queue_manager) > 0 else None
        processed_delay_steps = 0
        deadline_missed = False
        info = {"action": action, "processed_event_id": None, "processed_priority": None}

        if processed_event is not None:
            processed_delay_steps = max(self.current_step - processed_event.arrival_step, 0)
            if processed_event.deadline_step is not None:
                deadline_missed = processed_delay_steps > processed_event.deadline_step
            info["processed_event_id"] = processed_event.event_id
            info["processed_priority"] = processed_event.priority
            self.queue_manager.increment_wait_steps(processed_event.service_steps)
            self.current_step += processed_event.service_steps
        else:
            self.current_step += 1

        self._enqueue_arrivals()
        state = self.get_state()
        reward = compute_reward(
            previous_state=previous_state,
            state=state,
            processed_priority=info["processed_priority"],
            processed_delay_steps=processed_delay_steps,
            deadline_missed=deadline_missed,
            reward_weights=self.reward_weights,
        )
        done = self.next_event_index >= len(self.events) and len(self.queue_manager) == 0
        info["processed_delay_steps"] = processed_delay_steps
        info["deadline_missed"] = deadline_missed
        return state, reward, done, info
