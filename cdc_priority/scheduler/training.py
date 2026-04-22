from pathlib import Path

from ..settings import load_yaml_config
from ..utils import ensure_directory


def run_scheduler_training(config_path: Path) -> None:
    config = load_yaml_config(config_path)
    output_dir = ensure_directory(Path(config.values["output_dir"]))
    print(f"[scheduler] config: {config.path}")
    print(f"[scheduler] output_dir: {output_dir}")
    print("[scheduler] skeleton is ready for environment simulation and RL training.")
