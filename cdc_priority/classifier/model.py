import torch
import torch.nn as nn

from thesios_classifier.model import ThesiosClassifier, ThesiosClassifierV2


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if categorical_inputs.numel() == 0:
            inputs = numeric_inputs
        elif numeric_inputs.numel() == 0:
            inputs = categorical_inputs.float()
        else:
            inputs = torch.cat([categorical_inputs.float(), numeric_inputs], dim=1)
        return self.network(inputs)


__all__ = [
    "MLPClassifier",
    "ThesiosClassifier",
    "ThesiosClassifierV2",
]
