"""Config loader.

Merges ``configs/base.yaml`` with a per-model config and enforces that the model
config only overrides the ``model:`` block. Any attempt to change data/train/eval
fields raises ConfigError — that's the "fair comparison" guard.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    pass


BASE_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
ALLOWED_MODEL_OVERRIDE_KEYS = {"model"}


def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def load_config(model_config_path: str | Path) -> Dict[str, Any]:
    """Load and validate a model config, merging on top of base.yaml.

    Returns the merged config dict.
    """
    model_cfg_path = Path(model_config_path)
    base = _read_yaml(BASE_PATH)
    model = _read_yaml(model_cfg_path)

    extraneous = set(model) - ALLOWED_MODEL_OVERRIDE_KEYS
    if extraneous:
        raise ConfigError(
            f"{model_cfg_path}: model configs may only set 'model:'. "
            f"Found also: {sorted(extraneous)}. Move shared settings to base.yaml."
        )

    if "model" not in model:
        raise ConfigError(f"{model_cfg_path}: missing required 'model:' block")

    merged = dict(base)
    merged["model"] = model["model"]

    _validate(merged)
    return merged


def _validate(cfg: Dict[str, Any]) -> None:
    required_top = {"data", "train", "eval", "head", "model"}
    missing = required_top - cfg.keys()
    if missing:
        raise ConfigError(f"missing top-level keys: {sorted(missing)}")
    if cfg["model"].get("name") in (None, "REQUIRED"):
        raise ConfigError("model.name must be set in the model config")
    if cfg["head"]["num_classes"] != cfg["data"]["num_classes"]:
        raise ConfigError(
            "head.num_classes must equal data.num_classes — both are shared."
        )
