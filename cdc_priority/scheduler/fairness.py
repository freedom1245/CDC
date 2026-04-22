from .event import CDCEvent


def apply_aging(event: CDCEvent, starvation_threshold: int) -> CDCEvent:
    aged_event = event
    if aged_event.wait_steps >= starvation_threshold and aged_event.priority == "low":
        aged_event.priority = "medium"
    return aged_event
