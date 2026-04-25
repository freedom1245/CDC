import argparse
from pathlib import Path
from typing import Sequence

from .classifier.training import run_classifier_training
from .data.dataset_builder import (
    build_and_export_dataset_from_config,
    build_and_export_scheduler_dataset_from_config,
)
from .pipeline.online_simulation import run_pipeline
from .scheduler.training import run_scheduler_training
from .settings import default_settings


def build_parser() -> argparse.ArgumentParser:
    settings = default_settings()
    parser = argparse.ArgumentParser(
        description="CDC priority classification and RL scheduling toolkit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classifier_parser = subparsers.add_parser("classifier")
    classifier_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "classifier.yaml",
        help="Classifier YAML config path.",
    )

    scheduler_parser = subparsers.add_parser("scheduler")
    scheduler_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "scheduler.yaml",
        help="Scheduler YAML config path.",
    )

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "dataset.yaml",
        help="Dataset YAML config path.",
    )
    dataset_parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.project_root / "data" / "processed",
        help="Where to export train/valid/test splits and dataset report.",
    )
    dataset_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for dataset splitting.",
    )

    scheduler_dataset_parser = subparsers.add_parser("scheduler-dataset")
    scheduler_dataset_parser.add_argument(
        "--config",
        type=Path,
        default=settings.configs_dir / "dataset.yaml",
        help="Dataset YAML config path.",
    )
    scheduler_dataset_parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.project_root / "data" / "scheduler_processed",
        help="Where to export time-ordered scheduler train/valid/test splits and report.",
    )
    scheduler_dataset_parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp",
        help="Timestamp column used for time-ordered scheduler splitting.",
    )

    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument(
        "--classifier-config",
        type=Path,
        default=settings.configs_dir / "classifier.yaml",
        help="Classifier YAML config path.",
    )
    pipeline_parser.add_argument(
        "--scheduler-config",
        type=Path,
        default=settings.configs_dir / "scheduler.yaml",
        help="Scheduler YAML config path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "classifier":
        run_classifier_training(args.config)
        return

    if args.command == "scheduler":
        run_scheduler_training(args.config)
        return

    if args.command == "dataset":
        prepared = build_and_export_dataset_from_config(
            args.config,
            args.output_dir,
            random_state=args.random_state,
        )
        print(f"[dataset] exported train split to: {args.output_dir / 'train.csv'}")
        print(f"[dataset] exported valid split to: {args.output_dir / 'valid.csv'}")
        print(f"[dataset] exported test split to: {args.output_dir / 'test.csv'}")
        print(f"[dataset] exported report to: {args.output_dir / 'dataset_report.json'}")
        print(f"[dataset] row count: {prepared.report['row_count']}")
        return

    if args.command == "scheduler-dataset":
        prepared = build_and_export_scheduler_dataset_from_config(
            args.config,
            args.output_dir,
            timestamp_column=args.timestamp_column,
        )
        print(
            f"[scheduler-dataset] exported train split to: {args.output_dir / 'train.csv'}"
        )
        print(
            f"[scheduler-dataset] exported valid split to: {args.output_dir / 'valid.csv'}"
        )
        print(
            f"[scheduler-dataset] exported test split to: {args.output_dir / 'test.csv'}"
        )
        print(
            f"[scheduler-dataset] exported report to: {args.output_dir / 'dataset_report.json'}"
        )
        print(f"[scheduler-dataset] split strategy: {prepared.report['split_strategy']}")
        return

    run_pipeline(args.classifier_config, args.scheduler_config)
