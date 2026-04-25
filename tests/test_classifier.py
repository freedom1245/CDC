from cdc_priority.classifier.evaluate import build_classification_metrics
from cdc_priority.classifier.model import MLPClassifier


def test_classifier_model_builds() -> None:
    model = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3, dropout=0.1)
    assert model is not None


def test_build_classification_metrics() -> None:
    metrics = build_classification_metrics(
        labels=[0, 1, 1, 2],
        predictions=[0, 1, 0, 2],
    )

    assert metrics.accuracy == 0.75
    assert metrics.macro_f1 > 0
