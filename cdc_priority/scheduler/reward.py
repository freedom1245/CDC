from .env import SchedulerState


def compute_reward(state: SchedulerState) -> float:
    return -state.average_wait_steps
