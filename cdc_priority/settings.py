from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppSettings:
    project_root: Path
    configs_dir: Path
    outputs_dir: Path


@dataclass
class ConfigBundle:
    path: Path
    values: dict[str, Any] = field(default_factory=dict)


def default_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parent.parent
    return AppSettings(
        project_root=project_root,
        configs_dir=project_root / "configs",
        outputs_dir=project_root / "outputs",
    )


def load_yaml_config(path: Path) -> ConfigBundle:
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    return ConfigBundle(path=path, values=values)
