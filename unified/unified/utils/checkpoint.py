from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(path, *, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, *, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu", strict: bool = True):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0), state.get("extra", {})
