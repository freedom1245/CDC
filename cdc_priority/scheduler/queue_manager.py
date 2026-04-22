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

    def __len__(self) -> int:
        return len(self.events)
