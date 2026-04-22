from cdc_priority.classifier.model import MLPClassifier


def test_classifier_model_builds() -> None:
    model = MLPClassifier(input_dim=4, hidden_dim=8, num_classes=3, dropout=0.1)
    assert model is not None
