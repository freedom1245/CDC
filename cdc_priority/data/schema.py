from dataclasses import dataclass


@dataclass
class DatasetSchema:
    target: str
    categorical_columns: list[str]
    numeric_columns: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.categorical_columns + self.numeric_columns
