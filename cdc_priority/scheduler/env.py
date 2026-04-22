from dataclasses import dataclass, field

from .queue_manager import QueueManager


@dataclass
class SchedulerState:
    queue_length: int
    average_wait_steps: float


@dataclass
class SchedulerEnv:
    queue_manager: QueueManager = field(default_factory=QueueManager)
    current_step: int = 0

    def reset(self) -> SchedulerState:
        self.queue_manager = QueueManager()
        self.current_step = 0
        return self.get_state()

    def get_state(self) -> SchedulerState:
        return SchedulerState(
            queue_length=len(self.queue_manager),
            average_wait_steps=0.0,
        )

    def step(self, action: int) -> tuple[SchedulerState, float, bool, dict]:
        self.current_step += 1
        reward = 0.0
        done = False
        info = {"action": action}
        return self.get_state(), reward, done, info
