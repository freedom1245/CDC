import argparse
from pathlib import Path
from typing import Sequence

from .classifier.training import run_classifier_training
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

    run_pipeline(args.classifier_config, args.scheduler_config)
