from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplit(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    frame: pd.DataFrame,
    target: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    random_state: int,
) -> DatasetSplit:
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    train_frame, temp_frame = train_test_split(
        frame,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        stratify=frame[target] if target in frame.columns else None,
    )
    valid_portion = valid_ratio / (valid_ratio + test_ratio)
    valid_frame, test_frame = train_test_split(
        temp_frame,
        test_size=(1.0 - valid_portion),
        random_state=random_state,
        stratify=temp_frame[target] if target in temp_frame.columns else None,
    )
    return DatasetSplit(train=train_frame, valid=valid_frame, test=test_frame)
