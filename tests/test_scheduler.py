from cdc_priority.scheduler.env import SchedulerEnv


def test_scheduler_env_reset() -> None:
    env = SchedulerEnv()
    state = env.reset()
    assert state.queue_length == 0
