from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class FeatureArtifacts:
    categorical_columns: list[str]
    numeric_columns: list[str]
    label_encoder: LabelEncoder


def fit_feature_artifacts(
    train_frame: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    target: str,
) -> FeatureArtifacts:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_frame[target].astype(str))
    return FeatureArtifacts(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        label_encoder=label_encoder,
    )
