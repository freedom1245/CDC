def compute_reward(
    previous_state,
    state,
    processed_priority: str | None,
    processed_delay_steps: int,
    deadline_missed: bool,
    reward_weights: dict[str, float],
) -> float:
    reward = 0.0
    if processed_priority == "high":
        reward += reward_weights.get("high_priority_throughput", 1.0)
    elif processed_priority == "medium":
        reward += reward_weights.get("high_priority_throughput", 1.0) * 0.25

    reward -= reward_weights.get("average_delay_penalty", 0.1) * processed_delay_steps

    starvation_penalty = reward_weights.get("starvation_penalty", 1.0)
    wait_increase = max(
        state.max_low_priority_wait_steps - previous_state.max_low_priority_wait_steps,
        0,
    )
    wait_reduction = max(
        previous_state.max_low_priority_wait_steps - state.max_low_priority_wait_steps,
        0,
    )
    reward -= starvation_penalty * wait_increase
    reward += starvation_penalty * 0.5 * wait_reduction

    if processed_priority == "low" and previous_state.max_low_priority_wait_steps > 0:
        reward += starvation_penalty * 0.25

    if deadline_missed:
        reward -= reward_weights.get("deadline_miss_penalty", 1.5)
    return reward
