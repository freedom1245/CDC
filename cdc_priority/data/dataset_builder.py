from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .labeler import attach_priority_label
from .loader import load_events
from .preprocess import preprocess_events
from .schema import DatasetSchema
from .splitter import DatasetSplit, split_dataset


@dataclass
class PreparedDataset:
    schema: DatasetSchema
    full_frame: pd.DataFrame
    split: DatasetSplit


def build_dataset(
    data_path: Path,
    schema: DatasetSchema,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    random_state: int,
) -> PreparedDataset:
    frame = load_events(data_path)
    frame = preprocess_events(frame)
    frame = attach_priority_label(frame)
    split = split_dataset(
        frame=frame,
        target=schema.target,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    return PreparedDataset(schema=schema, full_frame=frame, split=split)
