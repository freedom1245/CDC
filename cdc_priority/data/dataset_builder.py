from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from ..settings import load_yaml_config
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
    report: dict[str, object]
    labeling_config: dict[str, object] | None = None


def build_dataset_report(frame: pd.DataFrame, split: DatasetSplit, target: str) -> dict[str, object]:
    label_distribution = (
        frame[target].value_counts(dropna=False).sort_index().to_dict() if target in frame else {}
    )
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "target": target,
        "label_distribution": label_distribution,
        "split_sizes": {
            "train": int(len(split.train)),
            "valid": int(len(split.valid)),
            "test": int(len(split.test)),
        },
    }


def build_report_with_metadata(
    frame: pd.DataFrame,
    split: DatasetSplit,
    target: str,
    labeling_config: dict | None = None,
) -> dict[str, object]:
    report = build_dataset_report(frame, split, target)
    if labeling_config:
        report["labeling"] = labeling_config
    return report


def split_dataset_by_time(
    frame: pd.DataFrame,
    timestamp_column: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> DatasetSplit:
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")
    if timestamp_column not in frame.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_column}")

    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], errors="coerce")
    ordered = ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)

    total = len(ordered)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    return DatasetSplit(
        train=ordered.iloc[:train_end].copy(),
        valid=ordered.iloc[train_end:valid_end].copy(),
        test=ordered.iloc[valid_end:].copy(),
    )


def resolve_project_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (config_path.parent.parent / path)


def load_dataset_config(config_path: Path) -> tuple[Path, DatasetSchema, dict[str, float], dict]:
    config = load_yaml_config(config_path)
    values = config.values
    schema = DatasetSchema(
        target=values["target"],
        categorical_columns=list(values.get("categorical_columns", [])),
        numeric_columns=list(values.get("numeric_columns", [])),
    )
    split = values.get("split", {})
    ratios = {
        "train": float(split.get("train", 0.7)),
        "valid": float(split.get("valid", 0.15)),
        "test": float(split.get("test", 0.15)),
    }
    data_path = resolve_project_path(config.path, values["data_path"])
    labeling_config = dict(values.get("labeling", {}))
    return data_path, schema, ratios, labeling_config


def build_dataset(
    data_path: Path,
    schema: DatasetSchema,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    random_state: int,
    labeling_config: dict | None = None,
) -> PreparedDataset:
    frame = load_events(data_path)
    frame = preprocess_events(frame)
    frame = attach_priority_label(frame, labeling_config=labeling_config)
    schema.validate_frame(frame)
    split = split_dataset(
        frame=frame,
        target=schema.target,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    report = build_report_with_metadata(
        frame,
        split,
        schema.target,
        labeling_config=labeling_config,
    )
    return PreparedDataset(
        schema=schema,
        full_frame=frame,
        split=split,
        report=report,
        labeling_config=labeling_config,
    )


def build_scheduler_dataset(
    data_path: Path,
    schema: DatasetSchema,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    timestamp_column: str = "timestamp",
    labeling_config: dict | None = None,
) -> PreparedDataset:
    frame = load_events(data_path)
    frame = preprocess_events(frame)
    frame = attach_priority_label(frame, labeling_config=labeling_config)
    schema.validate_frame(frame)
    split = split_dataset_by_time(
        frame=frame,
        timestamp_column=timestamp_column,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    report = build_report_with_metadata(
        frame,
        split,
        schema.target,
        labeling_config=labeling_config,
    )
    report["split_strategy"] = "time_ordered"
    report["timestamp_column"] = timestamp_column
    return PreparedDataset(
        schema=schema,
        full_frame=frame,
        split=split,
        report=report,
        labeling_config=labeling_config,
    )


def export_prepared_dataset(prepared: PreparedDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.split.train.to_csv(output_dir / "train.csv", index=False)
    prepared.split.valid.to_csv(output_dir / "valid.csv", index=False)
    prepared.split.test.to_csv(output_dir / "test.csv", index=False)
    with (output_dir / "dataset_report.json").open("w", encoding="utf-8") as file:
        json.dump(prepared.report, file, ensure_ascii=True, indent=2)


def build_dataset_from_config(
    config_path: Path,
    random_state: int,
) -> PreparedDataset:
    data_path, schema, ratios, labeling_config = load_dataset_config(config_path)
    return build_dataset(
        data_path=data_path,
        schema=schema,
        train_ratio=ratios["train"],
        valid_ratio=ratios["valid"],
        test_ratio=ratios["test"],
        random_state=random_state,
        labeling_config=labeling_config,
    )


def build_and_export_dataset_from_config(
    config_path: Path,
    output_dir: Path,
    random_state: int,
) -> PreparedDataset:
    prepared = build_dataset_from_config(config_path, random_state=random_state)
    export_prepared_dataset(prepared, output_dir)
    return prepared


def build_scheduler_dataset_from_config(
    config_path: Path,
    timestamp_column: str = "timestamp",
) -> PreparedDataset:
    data_path, schema, ratios, labeling_config = load_dataset_config(config_path)
    return build_scheduler_dataset(
        data_path=data_path,
        schema=schema,
        train_ratio=ratios["train"],
        valid_ratio=ratios["valid"],
        test_ratio=ratios["test"],
        timestamp_column=timestamp_column,
        labeling_config=labeling_config,
    )


def build_and_export_scheduler_dataset_from_config(
    config_path: Path,
    output_dir: Path,
    timestamp_column: str = "timestamp",
) -> PreparedDataset:
    prepared = build_scheduler_dataset_from_config(
        config_path,
        timestamp_column=timestamp_column,
    )
    export_prepared_dataset(prepared, output_dir)
    return prepared
