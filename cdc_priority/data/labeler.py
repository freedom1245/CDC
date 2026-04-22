import pandas as pd


def attach_priority_label(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    if "priority_label" not in labeled.columns:
        labeled["priority_label"] = "medium"
    return labeled
