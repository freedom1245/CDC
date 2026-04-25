from collections import deque
from dataclasses import dataclass, field

from .event import CDCEvent


@dataclass
class QueueManager:
    events: deque[CDCEvent] = field(default_factory=deque)

    def push(self, event: CDCEvent) -> None:
        self.events.append(event)

    def pop(self) -> CDCEvent | None:
        return self.events.popleft() if self.events else None

    def pop_at(self, index: int) -> CDCEvent | None:
        if index < 0 or index >= len(self.events):
            return None
        event = self.events[index]
        del self.events[index]
        return event

    def increment_wait_steps(self, delta: int = 1) -> None:
        for event in self.events:
            event.wait_steps += delta

    def priority_counts(self) -> dict[str, int]:
        counts = {"high": 0, "medium": 0, "low": 0}
        for event in self.events:
            counts[event.priority] = counts.get(event.priority, 0) + 1
        return counts

    def average_wait_steps(self) -> float:
        if not self.events:
            return 0.0
        return sum(event.wait_steps for event in self.events) / len(self.events)

    def __len__(self) -> int:
        return len(self.events)
