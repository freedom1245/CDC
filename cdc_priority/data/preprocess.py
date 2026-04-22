import pandas as pd


def preprocess_events(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].fillna("UNKNOWN")
    return cleaned
