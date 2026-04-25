from .event import CDCEvent


def apply_aging(event: CDCEvent, starvation_threshold: int) -> CDCEvent:
    aged_event = event
    if aged_event.wait_steps >= starvation_threshold and aged_event.priority == "low":
        aged_event.priority = "medium"
    if aged_event.wait_steps >= starvation_threshold * 2 and aged_event.priority == "medium":
        aged_event.priority = "high"
    return aged_event


def effective_priority_rank(event: CDCEvent, starvation_threshold: int) -> int:
    boosted_rank = event.priority_rank
    if starvation_threshold > 0:
        boosted_rank += event.wait_steps // starvation_threshold
    return min(boosted_rank, 2)


def aging_priority_key(event: CDCEvent, starvation_threshold: int) -> tuple[int, int, int, int]:
    boosted_rank = effective_priority_rank(event, starvation_threshold)
    starvation_level = 0
    if starvation_threshold > 0:
        starvation_level = min(event.wait_steps // starvation_threshold, 2)
    return (
        boosted_rank,
        starvation_level,
        event.priority_rank,
        event.wait_steps,
    )
