from dataclasses import dataclass

from sklearn.metrics import classification_report, f1_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float


def build_classification_metrics(
    labels: list[int],
    predictions: list[int],
) -> ClassificationMetrics:
    total = max(len(labels), 1)
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / total
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    )


def build_classification_report(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, object]:
    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
