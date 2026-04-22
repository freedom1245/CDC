from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_baseline_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }
