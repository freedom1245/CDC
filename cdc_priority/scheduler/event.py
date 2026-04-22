from dataclasses import dataclass


@dataclass
class CDCEvent:
    event_id: str
    priority: str
    arrival_step: int
    sync_cost: float
    deadline_step: int | None = None
    wait_steps: int = 0
