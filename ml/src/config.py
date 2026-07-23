"""Loads config/config.yaml and resolves relative paths against the ml/ root.

Phase 2 and Phase 3 should import `load_config` from here rather than
re-reading YAML themselves, so every phase agrees on where data lives.
"""
from pathlib import Path
from typing import Optional, Union

import yaml

ML_ROOT = Path(__file__).resolve().parent.parent  # .../ml
DEFAULT_CONFIG_PATH = ML_ROOT / "config" / "config.yaml"


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load the YAML config and resolve every entry under `paths:` to an
    absolute path rooted at ml/, so scripts work regardless of CWD.
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.setdefault("paths", {})
    for key, value in paths.items():
        paths[key] = str((ML_ROOT / value).resolve())

    return cfg
