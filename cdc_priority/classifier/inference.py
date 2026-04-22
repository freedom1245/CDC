from dataclasses import dataclass


@dataclass
class PredictionResult:
    label: str
    score: float
