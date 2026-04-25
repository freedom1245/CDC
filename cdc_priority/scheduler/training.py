import json
from pathlib import Path

import torch

from ..settings import default_settings, load_yaml_config
from ..utils import ensure_directory
from .agent import DQNAgent
from .env import SchedulerEnv
from .evaluate import export_policy_comparison, load_scheduler_events


def _run_single_episode(
    env: SchedulerEnv,
    agent: DQNAgent,
    max_steps: int,
    target_update_interval: int,
) -> tuple[float, int, float]:
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []

    for step in range(max_steps):
        state_vector = state.to_vector()
        action = agent.select_action(state_vector)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(
            state_vector,
            action,
            reward,
            next_state.to_vector(),
            done,
        )
        loss = agent.optimize()
        if loss is not None:
            losses.append(loss)
        total_reward += reward
        state = next_state

        if (step + 1) % target_update_interval == 0:
            agent.update_target_network()
        if done:
            return total_reward, step + 1, sum(losses) / max(len(losses), 1)

    return total_reward, max_steps, sum(losses) / max(len(losses), 1)


def _evaluate_greedy_policy(
    env: SchedulerEnv,
    agent: DQNAgent,
    max_steps: int,
) -> float:
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state.to_vector())
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    agent.epsilon = original_epsilon
    return total_reward


def run_scheduler_training(config_path: Path) -> None:
    settings = default_settings()
    config = load_yaml_config(config_path)
    output_dir = ensure_directory(settings.project_root / config.values["output_dir"])
    dataset_dir = settings.project_root / config.values["scheduler_dataset_dir"]
    train_events = load_scheduler_events(dataset_dir / "train.csv")
    valid_events = load_scheduler_events(dataset_dir / "valid.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_weights = dict(config.values.get("reward_weights", {}))
    starvation_threshold = int(config.values.get("starvation_threshold", 50))
    max_steps = int(config.values.get("max_steps_per_episode", 500))
    episode_count = int(config.values.get("episode_count", 200))
    target_update_interval = int(config.values.get("target_update_interval", 20))

    train_env = SchedulerEnv(
        events=train_events,
        reward_weights=reward_weights,
        starvation_threshold=starvation_threshold,
    )
    valid_env = SchedulerEnv(
        events=valid_events,
        reward_weights=reward_weights,
        starvation_threshold=starvation_threshold,
    )
    state_dim = len(train_env.reset().to_vector())
    agent = DQNAgent(
        action_count=3,
        state_dim=state_dim,
        gamma=float(config.values.get("gamma", 0.99)),
        epsilon=float(config.values.get("epsilon_start", 1.0)),
        epsilon_end=float(config.values.get("epsilon_end", 0.05)),
        epsilon_decay=float(config.values.get("epsilon_decay", 0.995)),
        replay_capacity=int(config.values.get("replay_capacity", 5000)),
        batch_size=int(config.values.get("batch_size", 64)),
        learning_rate=float(config.values.get("learning_rate", 1e-3)),
        device=device,
    )

    print(f"[scheduler] config: {config.path}")
    print(f"[scheduler] output_dir: {output_dir}")
    print(f"[scheduler] device: {device}")
    print(f"[scheduler] train events: {len(train_events)}")
    print(f"[scheduler] valid events: {len(valid_events)}")

    history: list[dict[str, float | int]] = []
    best_reward = float("-inf")
    best_state = None
    for episode in range(episode_count):
        train_reward, steps_used, average_loss = _run_single_episode(
            train_env,
            agent,
            max_steps=max_steps,
            target_update_interval=target_update_interval,
        )
        validation_reward = _evaluate_greedy_policy(valid_env, agent, max_steps=max_steps)
        history.append(
            {
                "episode": episode + 1,
                "train_reward": train_reward,
                "validation_reward": validation_reward,
                "steps_used": steps_used,
                "average_loss": average_loss,
                "epsilon": agent.epsilon,
            }
        )
        print(
            f"episode {episode + 1}/{episode_count} "
            f"train_reward={train_reward:.4f} "
            f"validation_reward={validation_reward:.4f} "
            f"epsilon={agent.epsilon:.4f}"
        )
        if validation_reward > best_reward:
            best_reward = validation_reward
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in agent.policy_network.state_dict().items()
            }
        agent.decay_epsilon()

    if best_state is not None:
        agent.policy_network.load_state_dict(best_state)
        agent.update_target_network()

    model_path = output_dir / "dqn_agent.pt"
    torch.save(
        {
            "model_state_dict": agent.policy_network.state_dict(),
            "history": history,
            "best_validation_reward": best_reward,
        },
        model_path,
    )

    policy_comparison = export_policy_comparison(
        data_path=dataset_dir / "test.csv",
        output_path=output_dir / "policy_comparison.csv",
        starvation_threshold=starvation_threshold,
    )
    report_path = output_dir / "scheduler_report.json"
    report_path.write_text(
        json.dumps(
            {
                "algorithm": config.values.get("algorithm", "dqn"),
                "best_validation_reward": best_reward,
                "history": history,
                "policy_comparison": policy_comparison.to_dict(orient="records"),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[scheduler] saved agent to: {model_path}")
    print(f"[scheduler] saved report to: {report_path}")
