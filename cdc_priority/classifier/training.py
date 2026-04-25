from pathlib import Path
from types import SimpleNamespace
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from thesios_classifier.data import (
    collapse_rare_applications,
    load_data,
    prepare_encoded_data,
)
from thesios_classifier.training import set_seed, train_and_save

from ..data.dataset_builder import build_dataset_from_config
from ..settings import default_settings, load_yaml_config
from ..utils import ensure_directory
from .evaluate import build_classification_metrics, build_classification_report
from .features import EncodedDataset, encode_dataset
from .model import MLPClassifier, ThesiosClassifier, ThesiosClassifierV2


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


class EncodedTensorDataset(Dataset):
    def __init__(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.categorical_inputs = categorical_inputs
        self.numeric_inputs = numeric_inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.categorical_inputs[index],
            self.numeric_inputs[index],
            self.labels[index],
        )


def _build_legacy_thesios_args(config_values: dict, project_root: Path) -> SimpleNamespace:
    output_dir = ensure_directory(_resolve_path(project_root, config_values["output_dir"]))
    model_path = output_dir / "classifier_model.pt"
    report_path = output_dir / "classifier_report.json"
    return SimpleNamespace(
        model_variant=config_values.get("model_variant", "v2"),
        data_path=_resolve_path(project_root, config_values["data_path"]),
        target=config_values.get("target", "priority_label"),
        max_rows=int(config_values.get("max_rows", 0)),
        top_apps=int(config_values.get("top_apps", 64)),
        test_size=float(config_values.get("test_size", 0.2)),
        random_state=int(config_values.get("random_state", 42)),
        batch_size=int(config_values.get("batch_size", 2048)),
        epochs=int(config_values.get("epochs", 20)),
        lr=float(config_values.get("lr", 1e-3)),
        weight_decay=float(config_values.get("weight_decay", 1e-4)),
        hidden_dim=int(config_values.get("hidden_dim", 256)),
        dropout=float(config_values.get("dropout", 0.3)),
        attention_dim=int(config_values.get("attention_dim", 128)),
        attention_heads=int(config_values.get("attention_heads", 8)),
        attention_layers=int(config_values.get("attention_layers", 3)),
        patience=int(config_values.get("patience", 4)),
        num_workers=int(config_values.get("num_workers", 0)),
        model_path=model_path,
        report_path=report_path,
    )


def _run_thesios_legacy_training(config_values: dict, project_root: Path) -> None:
    args = _build_legacy_thesios_args(config_values, project_root)
    set_seed(args.random_state)

    print(f"[classifier] source: thesios_legacy")
    print(f"[classifier] loading data from: {args.data_path}")
    frame = load_data(args.data_path, args.max_rows)
    frame = collapse_rare_applications(frame, args.top_apps)
    encoded = prepare_encoded_data(
        frame=frame,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_and_save(args, encoded)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _build_model(config_values: dict, encoded: EncodedDataset) -> nn.Module:
    model_variant = config_values.get("model_variant", "mlp")
    num_classes = len(encoded.artifacts.label_encoder.classes_)
    hidden_dim = int(config_values.get("hidden_dim", 256))
    dropout = float(config_values.get("dropout", 0.3))
    if model_variant == "mlp":
        input_dim = (
            encoded.train.categorical.size(1)
            + encoded.train.numeric.size(1)
        )
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_variant == "v1":
        return ThesiosClassifier(
            categorical_vocab_sizes=encoded.artifacts.categorical_vocab_sizes,
            numeric_dim=encoded.train.numeric.size(1),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            attention_dim=int(config_values.get("attention_dim", 128)),
            attention_heads=int(config_values.get("attention_heads", 8)),
            attention_layers=int(config_values.get("attention_layers", 3)),
        )
    if model_variant == "v2":
        return ThesiosClassifierV2(
            categorical_vocab_sizes=encoded.artifacts.categorical_vocab_sizes,
            numeric_dim=encoded.train.numeric.size(1),
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            attention_dim=int(config_values.get("attention_dim", 128)),
            attention_heads=int(config_values.get("attention_heads", 8)),
            attention_layers=int(config_values.get("attention_layers", 3)),
        )
    raise ValueError(f"Unsupported model variant: {model_variant}")


def _evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    with torch.no_grad():
        for categorical_inputs, numeric_inputs, labels in data_loader:
            categorical_inputs = categorical_inputs.to(device)
            numeric_inputs = numeric_inputs.to(device)
            labels = labels.to(device)

            logits = model(categorical_inputs, numeric_inputs)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(total_examples, 1)
    metrics = build_classification_metrics(labels_all, predictions_all)
    return average_loss, metrics.accuracy, labels_all, predictions_all


def _save_artifacts(
    output_dir: Path,
    model: nn.Module,
    encoded: EncodedDataset,
    artifact: dict[str, object],
) -> None:
    ensure_directory(output_dir)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": artifact,
        "label_classes": encoded.artifacts.label_encoder.classes_.tolist(),
        "category_maps": encoded.artifacts.category_maps,
        "numeric_stats": {
            "means": encoded.artifacts.numeric_stats.means,
            "stds": encoded.artifacts.numeric_stats.stds,
        },
    }
    torch.save(checkpoint, output_dir / "classifier_model.pt")
    with (output_dir / "classifier_report.json").open("w", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=True, indent=2)


def _run_configured_dataset_training(config_values: dict, project_root: Path) -> None:
    dataset_config_path = _resolve_path(project_root, config_values["dataset_config"])
    output_dir = ensure_directory(_resolve_path(project_root, config_values["output_dir"]))
    random_state = int(config_values.get("random_state", 42))
    _set_seed(random_state)

    print("[classifier] source: configured_dataset")
    print(f"[classifier] dataset config: {dataset_config_path}")
    prepared = build_dataset_from_config(dataset_config_path, random_state=random_state)
    encoded = encode_dataset(prepared)

    train_dataset = EncodedTensorDataset(
        encoded.train.categorical,
        encoded.train.numeric,
        encoded.train.labels,
    )
    valid_dataset = EncodedTensorDataset(
        encoded.valid.categorical,
        encoded.valid.numeric,
        encoded.valid.labels,
    )
    test_dataset = EncodedTensorDataset(
        encoded.test.categorical,
        encoded.test.numeric,
        encoded.test.labels,
    )

    batch_size = int(config_values.get("batch_size", 512))
    num_workers = int(config_values.get("num_workers", 0))
    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = _make_loader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = _make_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config_values, encoded).to(device)
    label_counts = torch.bincount(encoded.train.labels)
    class_weights = label_counts.sum() / label_counts.clamp(min=1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config_values.get("lr", config_values.get("learning_rate", 1e-3))),
        weight_decay=float(config_values.get("weight_decay", 1e-4)),
    )

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = int(config_values.get("patience", 4))
    epochs = int(config_values.get("epochs", 20))
    history: list[dict[str, float | int]] = []

    print(f"[classifier] device: {device}")
    print(f"[classifier] train size: {len(train_dataset)}")
    print(f"[classifier] valid size: {len(valid_dataset)}")
    print(f"[classifier] test size: {len(test_dataset)}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for categorical_inputs, numeric_inputs, labels in train_loader:
            categorical_inputs = categorical_inputs.to(device)
            numeric_inputs = numeric_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(categorical_inputs, numeric_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_examples += labels.size(0)

        train_loss = running_loss / max(running_examples, 1)
        train_acc = running_correct / max(running_examples, 1)
        val_loss, val_acc, _, _ = _evaluate_model(model, valid_loader, device, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc,
            }
        )
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"valid_loss={val_loss:.4f} "
            f"valid_acc={val_acc:.4f}"
        )

        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) <= 1e-9 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_loss, valid_acc, valid_labels, valid_predictions = _evaluate_model(
        model, valid_loader, device, criterion
    )
    test_loss, test_acc, test_labels, test_predictions = _evaluate_model(
        model, test_loader, device, criterion
    )
    valid_metrics = build_classification_metrics(valid_labels, valid_predictions)
    test_metrics = build_classification_metrics(test_labels, test_predictions)
    class_names = encoded.artifacts.label_encoder.classes_.tolist()
    artifact = {
        "target": prepared.schema.target,
        "model_variant": config_values.get("model_variant", "mlp"),
        "classes": class_names,
        "categorical_columns": prepared.schema.categorical_columns,
        "numeric_columns": prepared.schema.numeric_columns,
        "categorical_vocab_sizes": encoded.artifacts.categorical_vocab_sizes,
        "batch_size": batch_size,
        "epochs": epochs,
        "hidden_dim": int(config_values.get("hidden_dim", 256)),
        "dropout": float(config_values.get("dropout", 0.3)),
        "learning_rate": float(config_values.get("lr", config_values.get("learning_rate", 1e-3))),
        "weight_decay": float(config_values.get("weight_decay", 1e-4)),
        "validation": {
            "loss": valid_loss,
            "accuracy": valid_acc,
            "macro_f1": valid_metrics.macro_f1,
            "weighted_f1": valid_metrics.weighted_f1,
        },
        "test": {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": test_metrics.macro_f1,
            "weighted_f1": test_metrics.weighted_f1,
        },
        "history": history,
        "dataset_report": prepared.report,
        "test_report": build_classification_report(
            test_labels,
            test_predictions,
            class_names,
        ),
        "numeric_stats": {
            "means": encoded.artifacts.numeric_stats.means,
            "stds": encoded.artifacts.numeric_stats.stds,
        },
    }
    _save_artifacts(output_dir, model, encoded, artifact)
    print(f"[classifier] saved model to: {output_dir / 'classifier_model.pt'}")
    print(f"[classifier] saved report to: {output_dir / 'classifier_report.json'}")


def run_classifier_training(config_path: Path) -> None:
    settings = default_settings()
    config = load_yaml_config(config_path)
    source = config.values.get("source", "thesios_legacy")

    print(f"[classifier] config: {config.path}")
    if source == "thesios_legacy":
        _run_thesios_legacy_training(config.values, settings.project_root)
        return
    if source == "configured_dataset":
        _run_configured_dataset_training(config.values, settings.project_root)
        return

    raise ValueError(f"Unsupported classifier source: {source}")
