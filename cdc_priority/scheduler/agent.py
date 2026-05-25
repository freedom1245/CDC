from collections import deque
from dataclasses import dataclass, field
import random

import torch
import torch.nn as nn


class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass
class DQNAgent:
    action_count: int
    state_dim: int
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_capacity: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "cpu"
    policy_network: DQNetwork = field(init=False)
    target_network: DQNetwork = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    replay_buffer: deque = field(init=False)

    def __post_init__(self) -> None:
        self.policy_network = DQNetwork(self.state_dim, self.action_count).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_count).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )
        self.replay_buffer = deque(maxlen=self.replay_capacity)

    def select_action(self, state_vector: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_count)
        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vector,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.policy_network(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.replay_buffer.append((state, action, reward, next_state, done))

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_network(states_tensor).gather(1, actions_tensor).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states_tensor).max(dim=1).values
            target_q = rewards_tensor + self.gamma * next_q * (1.0 - dones_tensor)

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
