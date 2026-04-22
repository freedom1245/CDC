from pathlib import Path
from types import SimpleNamespace

from thesios_classifier.data import (
    collapse_rare_applications,
    load_data,
    prepare_encoded_data,
)
from thesios_classifier.training import set_seed, train_and_save

from ..settings import default_settings, load_yaml_config
from ..utils import ensure_directory


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path)


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


def run_classifier_training(config_path: Path) -> None:
    settings = default_settings()
    config = load_yaml_config(config_path)
    source = config.values.get("source", "thesios_legacy")

    print(f"[classifier] config: {config.path}")
    if source == "thesios_legacy":
        _run_thesios_legacy_training(config.values, settings.project_root)
        return

    raise ValueError(f"Unsupported classifier source: {source}")
