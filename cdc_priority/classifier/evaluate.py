from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
