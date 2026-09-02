from __future__ import annotations
from pathlib import Path
import hashlib
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
    if state.get("model_format") == "partial":
        # Frozen-pretrained run: finalize_run dropped the tensors that were
        # bit-identical to the weights file (see drop_frozen_backbone). The
        # caller must have built the model FROM THAT SAME FILE, in which case
        # the missing tensors are already correct. Verified exactly, not
        # assumed: anything missing beyond the recorded list is a real error.
        expected = set(state.get("dropped_keys", ()))
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        stray = set(missing) - expected
        if stray or unexpected or not expected:
            raise RuntimeError(
                f"{path}: partial checkpoint does not fit this model "
                f"(unaccounted missing: {sorted(stray)[:5]}, "
                f"unexpected: {list(unexpected)[:5]}). Rebuild the model from "
                f"the run's own config with its pretrained weights.")
        # Key names are identical whether the caller loaded the pretrained
        # weights or left the backbone random, so verify CONTENT too — otherwise
        # a model built with weights=None loads silently and is quietly wrong.
        want = state.get("dropped_sha256")
        if want is not None:
            got = _fingerprint(model.state_dict(), expected)
            if got != want:
                ref = state.get("backbone_ref", {})
                raise RuntimeError(
                    f"{path}: the tensors this checkpoint omitted do not match "
                    f"what the model was built with (sha256 {got[:12]} != "
                    f"{want[:12]}). Build the backbone from "
                    f"{ref.get('name')} weights={ref.get('weights')} before loading.")
    else:
        model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("epoch", 0), state.get("extra", {})


def strip_to_weights(path) -> tuple[int, int]:
    """Rewrite a checkpoint in place keeping only what inference needs.

    Drops the ``optimizer`` / ``scheduler`` / ``scaler`` entries, keeping
    ``model`` plus the ``epoch`` / ``extra`` metadata. AdamW carries two fp32
    moment tensors per trainable parameter, so for a fine-tune run this is ~3x
    the model weights — dead weight once the run is finished, since both
    ``scripts.evaluate`` and ``scripts.infer`` load ``state["model"]`` only.

    Written to a temp file and atomically replaced, so an interruption mid-strip
    leaves the original checkpoint intact.

    Returns ``(bytes_before, bytes_after)``.
    """
    path = Path(path)
    before = path.stat().st_size
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not any(k in state for k in ("optimizer", "scheduler", "scaler")):
        return before, before          # already stripped — no-op
    slim = {"model": state["model"], "epoch": state.get("epoch", 0),
            "extra": state.get("extra", {})}
    tmp = path.with_suffix(".pt.tmp")
    torch.save(slim, tmp)
    tmp.replace(path)
    return before, path.stat().st_size


def _fingerprint(state_dict, keys) -> str:
    """Content hash over a set of tensors, order-independent of dict insertion.

    Key names alone cannot tell a pretrained backbone from a randomly
    initialised one — the key sets are identical. So a partial checkpoint
    records what the dropped tensors CONTAINED, and the loader recomputes it
    against whatever the caller actually built.
    """
    h = hashlib.sha256()
    for k in sorted(keys):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def drop_frozen_backbone(path, cfg) -> tuple[int, int]:
    """Drop the tensors a frozen-PRETRAINED run never changed.

    In a ``freeze_backbone: true`` + ``pretrained: true`` run most of the encoder
    never receives a gradient, so what is stored is byte-for-byte the weights
    file already on disk — duplicated once per fraction, up to 1.2 GB apiece for
    dino3d. Only the head and the bridging adapter actually learned anything.

    ``freeze_backbone`` does NOT mean the whole ``backbone.`` subtree is frozen:
    the adapter/stem that maps the encoder onto the unified 5-level contract
    trains in every condition (supSwin: 291,680 of its 8,965,752 trainable
    parameters). So rather than assume which subtree is frozen, every tensor is
    compared against a freshly built reference and only the bit-identical ones
    are dropped. That is self-configuring, and a run whose "frozen" encoder did
    move keeps every tensor that moved.

    NOT applied to scratch runs: there is no ``torch.manual_seed`` in the
    training path, so a randomly-initialised backbone is unreproducible and the
    checkpoint is the only copy that will ever exist.

    Returns ``(bytes_before, bytes_after)``.
    """
    path = Path(path)
    before = path.stat().st_size
    mcfg = cfg.get("model", {})
    if not (mcfg.get("freeze_backbone") and mcfg.get("pretrained", True)):
        return before, before
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("model_format") == "partial":
        return before, before                      # already dropped — no-op
    sd = state["model"]
    if not any(k.startswith("backbone.") for k in sd):
        return before, before

    from unified.models import build_backbone      # local: avoids an import cycle
    ref = build_backbone(mcfg["name"], weights=mcfg.get("weights"),
                         **mcfg.get("kwargs", {})).state_dict()
    dropped = []
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        r = ref.get(k[len("backbone."):])
        if r is not None and r.shape == v.shape and torch.equal(r, v.to(r.dtype)):
            dropped.append(k)
    if not dropped:
        return before, before

    state["model"] = {k: v for k, v in sd.items() if k not in set(dropped)}
    state["model_format"] = "partial"
    state["dropped_keys"] = dropped
    state["backbone_ref"] = {"name": mcfg["name"], "weights": mcfg.get("weights"),
                             "kwargs": mcfg.get("kwargs", {})}
    state["dropped_sha256"] = _fingerprint(sd, dropped)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return before, path.stat().st_size


def finalize_run(output_dir, *, strip_best: bool = True, cfg=None) -> dict:
    """End-of-run disk cleanup: drop resume state, keep the deliverable.

    Deletes every ``epoch_*.pt`` (they exist only so an OOM-killed run can
    resume) and strips ``best.pt`` down to weights. Call this ONLY after a run
    has finished cleanly — while a run is still live or retryable those
    checkpoints are what makes it resumable.

    Returns a summary dict: ``{"removed": n, "freed_bytes": n}``.
    """
    output_dir = Path(output_dir)
    freed = 0
    removed = 0
    for ckpt in sorted(output_dir.glob("epoch_*.pt")):
        try:
            freed += ckpt.stat().st_size
            ckpt.unlink()
            removed += 1
        except OSError:
            freed -= 0
    best = output_dir / "best.pt"
    if strip_best and best.exists():
        before, after = strip_to_weights(best)
        freed += before - after
        if cfg is not None:
            before, after = drop_frozen_backbone(best, cfg)
            freed += before - after
    return {"removed": removed, "freed_bytes": freed}
