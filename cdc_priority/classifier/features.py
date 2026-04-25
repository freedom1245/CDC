from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from ..data.dataset_builder import PreparedDataset


@dataclass
class EncodedSplit:
    categorical: torch.Tensor
    numeric: torch.Tensor
    labels: torch.Tensor


@dataclass
class NumericStats:
    means: dict[str, float]
    stds: dict[str, float]


@dataclass
class FeatureArtifacts:
    categorical_columns: list[str]
    numeric_columns: list[str]
    label_encoder: LabelEncoder
    category_maps: dict[str, dict[str, int]]
    categorical_vocab_sizes: list[int]
    numeric_stats: NumericStats


@dataclass
class EncodedDataset:
    train: EncodedSplit
    valid: EncodedSplit
    test: EncodedSplit
    artifacts: FeatureArtifacts


def _encode_categorical_series(
    train_series: pd.Series,
    other_series: pd.Series,
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    vocab = {"__UNK__": 0}
    for index, value in enumerate(sorted(train_series.astype(str).unique().tolist()), start=1):
        vocab[value] = index

    train_encoded = train_series.astype(str).map(lambda value: vocab.get(value, 0))
    other_encoded = other_series.astype(str).map(lambda value: vocab.get(value, 0))
    return (
        vocab,
        torch.tensor(train_encoded.to_numpy(), dtype=torch.long),
        torch.tensor(other_encoded.to_numpy(), dtype=torch.long),
    )


def _encode_categorical_block(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[dict[str, dict[str, int]], list[int], torch.Tensor, torch.Tensor]:
    category_maps: dict[str, dict[str, int]] = {}
    vocab_sizes: list[int] = []
    train_tensors: list[torch.Tensor] = []
    other_tensors: list[torch.Tensor] = []

    for column in categorical_columns:
        vocab, train_encoded, other_encoded = _encode_categorical_series(
            train_frame[column],
            other_frame[column],
        )
        category_maps[column] = vocab
        vocab_sizes.append(len(vocab))
        train_tensors.append(train_encoded.unsqueeze(1))
        other_tensors.append(other_encoded.unsqueeze(1))

    if not train_tensors:
        train_categorical = torch.zeros((len(train_frame), 0), dtype=torch.long)
        other_categorical = torch.zeros((len(other_frame), 0), dtype=torch.long)
    else:
        train_categorical = torch.cat(train_tensors, dim=1)
        other_categorical = torch.cat(other_tensors, dim=1)

    return category_maps, vocab_sizes, train_categorical, other_categorical


def _encode_numeric_block(
    train_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[NumericStats, torch.Tensor, torch.Tensor]:
    if not numeric_columns:
        empty_train = torch.zeros((len(train_frame), 0), dtype=torch.float32)
        empty_other = torch.zeros((len(other_frame), 0), dtype=torch.float32)
        return NumericStats(means={}, stds={}), empty_train, empty_other

    train_numeric = train_frame[numeric_columns].copy().fillna(0)
    other_numeric = other_frame[numeric_columns].copy().fillna(0)
    means = train_numeric.mean()
    stds = train_numeric.std().replace(0, 1.0).fillna(1.0)
    train_scaled = ((train_numeric - means) / stds).astype("float32")
    other_scaled = ((other_numeric - means) / stds).astype("float32")
    stats = NumericStats(
        means={column: float(means[column]) for column in numeric_columns},
        stds={column: float(stds[column]) for column in numeric_columns},
    )
    return (
        stats,
        torch.tensor(train_scaled.to_numpy(), dtype=torch.float32),
        torch.tensor(other_scaled.to_numpy(), dtype=torch.float32),
    )


def _encode_labels(
    label_encoder: LabelEncoder,
    frame: pd.DataFrame,
    target: str,
) -> torch.Tensor:
    return torch.tensor(
        label_encoder.transform(frame[target].astype(str)),
        dtype=torch.long,
    )


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
        category_maps={},
        categorical_vocab_sizes=[],
        numeric_stats=NumericStats(means={}, stds={}),
    )


def encode_dataset(prepared: PreparedDataset) -> EncodedDataset:
    schema = prepared.schema
    train_frame = prepared.split.train.reset_index(drop=True)
    valid_frame = prepared.split.valid.reset_index(drop=True)
    test_frame = prepared.split.test.reset_index(drop=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_frame[schema.target].astype(str))

    category_maps, categorical_vocab_sizes, train_categorical, valid_categorical = (
        _encode_categorical_block(
            train_frame,
            valid_frame,
            schema.categorical_columns,
        )
    )
    _, _, _, test_categorical = _encode_categorical_block(
        train_frame,
        test_frame,
        schema.categorical_columns,
    )

    numeric_stats, train_numeric, valid_numeric = _encode_numeric_block(
        train_frame,
        valid_frame,
        schema.numeric_columns,
    )
    _, _, test_numeric = _encode_numeric_block(
        train_frame,
        test_frame,
        schema.numeric_columns,
    )

    artifacts = FeatureArtifacts(
        categorical_columns=schema.categorical_columns,
        numeric_columns=schema.numeric_columns,
        label_encoder=label_encoder,
        category_maps=category_maps,
        categorical_vocab_sizes=categorical_vocab_sizes,
        numeric_stats=numeric_stats,
    )
    return EncodedDataset(
        train=EncodedSplit(
            categorical=train_categorical,
            numeric=train_numeric,
            labels=_encode_labels(label_encoder, train_frame, schema.target),
        ),
        valid=EncodedSplit(
            categorical=valid_categorical,
            numeric=valid_numeric,
            labels=_encode_labels(label_encoder, valid_frame, schema.target),
        ),
        test=EncodedSplit(
            categorical=test_categorical,
            numeric=test_numeric,
            labels=_encode_labels(label_encoder, test_frame, schema.target),
        ),
        artifacts=artifacts,
    )
