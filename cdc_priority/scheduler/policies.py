from .queue_manager import QueueManager


def fifo_policy(queue_manager: QueueManager):
    return queue_manager.pop()


def strict_priority_policy(queue_manager: QueueManager):
    return queue_manager.pop()


def weighted_round_robin_policy(queue_manager: QueueManager):
    return queue_manager.pop()
