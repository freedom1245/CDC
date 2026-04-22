from dataclasses import dataclass
import random


@dataclass
class DQNAgent:
    action_count: int
    epsilon: float = 0.1

    def select_action(self) -> int:
        return random.randrange(self.action_count)
