"""Config loader.

Merges ``configs/base.yaml`` with a per-model config. A model config may set
the ``model:`` block and, optionally, deep-merge overrides onto the ``head:``,
``data:`` and ``eval:`` blocks. ``train:`` stays globally locked (raises
ConfigError if a model config touches it) so the optimizer / schedule / loss
recipe remains identical across every run.

Fairness note: overriding ``head:``/``data:``/``eval:`` per model BREAKS
cross-backbone fairness — that is why the fair-comparison sweep leaves them at
the base.yaml defaults. They are unlocked only for the deliberate "best-effort"
track, where a single model is tuned for its best achievable number (e.g. a
larger ``data.patch_size`` + matching ``eval.sliding_window.roi_size`` to widen
the field of view). Keep these blocks at base for any run you intend to compare
across backbones/heads; vary them only when the point of the run is that lever.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    pass


BASE_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
ALLOWED_MODEL_OVERRIDE_KEYS = {"model", "head", "data", "eval"}
# Deep-merged over the base block (nested dicts merged recursively); ``train``
# is intentionally NOT here — the recipe stays locked.
DEEP_MERGE_KEYS = ("head", "data", "eval")


def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` onto ``base``.

    Nested mappings are merged key-by-key; any non-dict value (scalars, lists
    such as ``patch_size``/``roi_size``) is replaced wholesale by the override.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = _deep_merge(base.get(k), v) if k in base else v
        return merged
    return override


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
            f"{model_cfg_path}: model configs may only set 'model:' (and optionally "
            f"'head:'). Found also: {sorted(extraneous)}. Move shared settings to base.yaml."
        )

    if "model" not in model:
        raise ConfigError(f"{model_cfg_path}: missing required 'model:' block")

    merged = dict(base)
    merged["model"] = model["model"]
    # Optional deep-merge overrides: head (swap the decoder), data (e.g. a larger
    # patch_size / spacing for a best-effort run), eval (matching sliding-window
    # roi_size). Nested keys are merged recursively so a config need only restate
    # the leaf it changes — e.g. eval.sliding_window.roi_size without repeating
    # sw_batch_size/overlap/mode. train stays locked (rejected as extraneous).
    for key in DEEP_MERGE_KEYS:
        if key in model:
            merged[key] = _deep_merge(base.get(key, {}), model[key])

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
