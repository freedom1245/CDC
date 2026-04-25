from .fairness import aging_priority_key
from .queue_manager import QueueManager


def fifo_policy(queue_manager: QueueManager):
    return queue_manager.pop()


def strict_priority_policy(queue_manager: QueueManager):
    if not queue_manager.events:
        return None
    best_index = max(
        range(len(queue_manager.events)),
        key=lambda index: (
            queue_manager.events[index].priority_rank,
            -queue_manager.events[index].arrival_step,
        ),
    )
    return queue_manager.pop_at(best_index)


def weighted_round_robin_policy(queue_manager: QueueManager):
    return queue_manager.pop()


def aging_policy(queue_manager: QueueManager, starvation_threshold: int = 5):
    if not queue_manager.events:
        return None
    best_index = max(
        range(len(queue_manager.events)),
        key=lambda index: (
            *aging_priority_key(queue_manager.events[index], starvation_threshold),
            -queue_manager.events[index].arrival_step,
        ),
    )
    return queue_manager.pop_at(best_index)
