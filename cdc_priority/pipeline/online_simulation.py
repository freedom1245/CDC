from pathlib import Path


def run_pipeline(classifier_config: Path, scheduler_config: Path) -> None:
    print(f"[pipeline] classifier config: {classifier_config}")
    print(f"[pipeline] scheduler config: {scheduler_config}")
    print("[pipeline] skeleton is ready for end-to-end integration.")
