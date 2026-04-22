from dataclasses import dataclass


@dataclass
class SchedulerMetrics:
    throughput: float
    average_delay: float
    max_low_priority_wait: float
