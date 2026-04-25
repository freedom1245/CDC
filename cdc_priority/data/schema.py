from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSchema:
    target: str
    categorical_columns: list[str]
    numeric_columns: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.categorical_columns + self.numeric_columns

    def validate_frame(self, frame: pd.DataFrame) -> None:
        required_columns = set(self.feature_columns + [self.target])
        missing_columns = sorted(column for column in required_columns if column not in frame)
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Dataset is missing required columns: {missing_text}")
