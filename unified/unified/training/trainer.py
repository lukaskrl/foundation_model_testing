"""Shared training loop.

The loop is identical for every model. Add per-model differences via configs,
not here.
"""
from __future__ import annotations
import math
import os
import time
from pathlib import Path

import torch
import wandb
from tqdm.auto import tqdm

from ..utils import get_logger, save_checkpoint, load_checkpoint
from .loss import build_loss
from .optim import build_optimizer, build_scheduler


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, evaluator, output_dir,
                 *, resume=None):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = build_loss(cfg)
        # Filter so AdamW doesn't track moments for frozen params.
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(cfg, trainable)
        # AMP defaults to train.amp; a model config may opt out via model.amp=false.
        model_amp = cfg["model"].get("amp")
        self.amp = bool(model_amp if model_amp is not None else cfg["train"].get("amp", True))
        # Prefer bf16 autocast when the GPU supports it. bf16 shares fp32's 8-bit
        # exponent, so the loss-scaling overflow that produced late-training NaNs
        # under fp16 (vista3d ~ep78, ctfm ~ep476) cannot occur and no GradScaler is
        # needed. Fall back to fp16 + GradScaler only on hardware without bf16.
        self.amp_dtype = None
        self.scaler = None
        if self.amp:
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            self.amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            if self.amp_dtype is torch.float16:
                self.scaler = torch.cuda.amp.GradScaler()
        self.grad_clip = cfg["train"].get("grad_clip", 0.0)
        # Gradient accumulation: effective batch = batch_size * grad_accum_steps.
        # Model configs may override via model.grad_accum_steps.
        model_accum = cfg["model"].get("grad_accum_steps")
        self.grad_accum_steps = max(1, int(
            model_accum if model_accum is not None
            else cfg["train"].get("grad_accum_steps", 1)
        ))
        # Scheduler ticks once per optimizer step, not per microbatch, so its
        # horizon is in optimizer steps too.
        microbatches_per_epoch = max(1, len(train_loader))
        opt_steps_per_epoch = max(
            1, math.ceil(microbatches_per_epoch / self.grad_accum_steps)
        )
        self.scheduler = build_scheduler(cfg, self.optimizer, opt_steps_per_epoch)
        self.log = get_logger("trainer")
        self.best_dice = -1.0
        self.early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
        self.val_rounds_since_improve = 0
        self._init_wandb()
        self.global_step = 0
        self.start_epoch = 0

        if resume is not None:
            ckpt = self._find_resume_checkpoint(resume)
            if ckpt is not None:
                self.log.info("resuming from %s", ckpt)
                epoch, extra = load_checkpoint(
                    ckpt,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    map_location=self.device,
                )
                self.start_epoch = epoch
                if "mean_dice" in extra:
                    self.best_dice = extra["mean_dice"]
                self.global_step = epoch * max(1, len(train_loader))
                self.log.info(
                    "resumed: start_epoch=%d best_dice=%.4f global_step=%d",
                    self.start_epoch, self.best_dice, self.global_step,
                )
            else:
                self.log.warning(
                    "--resume set but no checkpoint found in %s; starting fresh",
                    self.output_dir,
                )

    def _init_wandb(self) -> None:
        """Start (or resume) the W&B run for this output directory.

        Logging config lives in a top-level ``wandb:`` block in base.yaml — it is
        infrastructure, not a fair-comparison hyperparameter, so it sits outside
        the locked train/data/eval blocks. The run id is derived from the output
        directory name so ``--resume`` continues the same W&B run rather than
        spawning a duplicate. WANDB_MODE in the environment overrides the config
        mode (set it to ``offline`` on nodes without network, ``disabled`` to mute
        logging entirely — wandb.log then becomes a no-op).
        """
        wcfg = self.cfg.get("wandb", {}) or {}
        run_id = "".join(c if c.isalnum() or c in "-_" else "-"
                         for c in self.output_dir.name)
        wandb.init(
            project=wcfg.get("project", "unified-foundation-models"),
            entity=wcfg.get("entity"),
            name=self.output_dir.name,
            id=run_id,
            resume="allow",
            dir=str(self.output_dir),
            mode=os.environ.get("WANDB_MODE", wcfg.get("mode", "online")),
            config=self.cfg,
        )
        # Plot per-step series against global_step and per-epoch series against
        # epoch, instead of W&B's internal monotonic counter.
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss_step", step_metric="global_step")
        wandb.define_metric("train/lr", step_metric="global_step")
        wandb.define_metric("train/loss_epoch", step_metric="epoch")
        wandb.define_metric("train/epoch_time_s", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("val_individual_organ/*", step_metric="epoch")

    def _find_resume_checkpoint(self, resume) -> Path | None:
        if isinstance(resume, (str, Path)) and Path(resume).is_file():
            return Path(resume)
        # Auto-detect: prefer latest epoch checkpoint (has optimizer/scheduler
        # state), fall back to best.pt.
        epoch_ckpts = sorted(self.output_dir.glob("epoch_*.pt"))
        if epoch_ckpts:
            return epoch_ckpts[-1]
        best = self.output_dir / "best.pt"
        if best.exists():
            return best
        return None

    def _train_epoch(self, epoch: int):
        self.model.train()
        losses = []
        total_steps = len(self.train_loader)
        accum = self.grad_accum_steps
        pbar = tqdm(
            self.train_loader,
            total=total_steps,
            desc=f"epoch {epoch:>3d}/{self.cfg['train']['epochs']} [train]",
            dynamic_ncols=True,
            leave=False,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            # Step the optimizer at every Nth microbatch and on the final
            # microbatch of the epoch (drains any partial accumulation).
            step_now = ((step + 1) % accum == 0) or (step + 1 == total_steps)

            if self.amp:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    logits = self.model(image)
                    loss = self.loss_fn(logits, label)
            else:
                logits = self.model(image)
                loss = self.loss_fn(logits, label)

            # Scale by 1/accum so the accumulated gradient is the mean of the
            # microbatch gradients, matching the no-accum case. The scaler path
            # is only taken under fp16 AMP; bf16 and fp32 need no loss scaling.
            if self.scaler is not None:
                self.scaler.scale(loss / accum).backward()
                if step_now:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                (loss / accum).backward()
                if step_now:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()  # raw microbatch loss for logging
            losses.append(loss_val)
            self.global_step += 1

            if not math.isfinite(loss_val):
                # Abort fast: prior sweeps wasted 100+ epochs after a NaN with
                # AMP scaler bouncing. Caller marks the run FAIL; best.pt is
                # preserved.
                self.log.error(
                    "non-finite train loss at epoch %d step %d (loss=%s); aborting",
                    epoch, step + 1, loss_val,
                )
                raise RuntimeError(
                    f"non-finite train loss at epoch {epoch} step {step + 1}"
                )

            lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({
                "train/loss_step": loss_val,
                "train/lr": lr,
                "global_step": self.global_step,
            })

            running = sum(losses[-50:]) / min(50, len(losses))
            postfix = {"loss": f"{running:.4f}", "lr": f"{lr:.2e}"}
            if torch.cuda.is_available():
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}"
            pbar.set_postfix(postfix, refresh=False)

            if (step + 1) % self.cfg["train"]["log_interval_steps"] == 0:
                self.log.info(
                    "epoch %d step %d/%d loss=%.4f lr=%.2e",
                    epoch, step + 1, total_steps, running, lr,
                )

        pbar.close()
        return sum(losses) / max(1, len(losses))

    def _validate(self, epoch: int) -> float:
        if self.val_loader is None or self.evaluator is None:
            return float("nan")
        metrics = self.evaluator.evaluate(self.model, self.val_loader, self.device)
        mean_dice = metrics["mean_dice"]
        self.log.info("epoch %d val_mean_dice=%.4f", epoch, mean_dice)
        log_row = {"epoch": epoch}
        for key, value in metrics.items():
            if value is None or value != value:  # skip NaN
                continue
            if key.startswith("mean_"):
                tag = f"val/{key}"
            else:
                tag = f"val_individual_organ/{key}"
            log_row[tag] = value
        wandb.log(log_row)
        # Persist a per-class metrics row.
        (self.output_dir / "val_metrics.jsonl").open("a").write(
            __import__("json").dumps({"epoch": epoch, **metrics}) + "\n"
        )
        return mean_dice

    def run(self):
        t_total = time.time()
        n_train = len(self.train_loader)
        n_val = len(self.val_loader) if self.val_loader is not None else 0
        total_epochs = self.cfg["train"]["epochs"]

        amp_desc = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(
            self.amp_dtype, "off"
        )
        self.log.info(
            "start: epochs %d-%d/%d device=%s amp=%s train_batches=%d val_batches=%d",
            self.start_epoch + 1, total_epochs, total_epochs,
            self.device, amp_desc, n_train, n_val,
        )
        if self.start_epoch >= total_epochs:
            self.log.warning(
                "start_epoch=%d >= total epochs=%d; nothing to train",
                self.start_epoch, total_epochs,
            )
            return
        try:
            for epoch in range(self.start_epoch + 1, total_epochs + 1):
                t = time.time()
                train_loss = self._train_epoch(epoch)
                epoch_time = time.time() - t
                self.log.info("epoch %d done train_loss=%.4f dt=%.1fs",
                              epoch, train_loss, epoch_time)
                wandb.log({
                    "train/loss_epoch": train_loss,
                    "train/epoch_time_s": epoch_time,
                    "epoch": epoch,
                })

                stop_early = False
                if epoch % self.cfg["train"]["val_interval_epochs"] == 0:
                    dice = self._validate(epoch)
                    if dice > self.best_dice and not (dice != dice):  # not NaN
                        self.best_dice = dice
                        self.val_rounds_since_improve = 0
                        wandb.log({"val/best_mean_dice": self.best_dice, "epoch": epoch})
                        save_checkpoint(
                            self.output_dir / "best.pt",
                            model=self.model, optimizer=self.optimizer,
                            scheduler=self.scheduler, scaler=self.scaler,
                            epoch=epoch, extra={"mean_dice": dice},
                        )
                    else:
                        self.val_rounds_since_improve += 1
                        if (self.early_stop_patience > 0
                                and self.val_rounds_since_improve >= self.early_stop_patience):
                            self.log.info(
                                "early stopping at epoch %d: %d val rounds without dice improvement (best=%.4f)",
                                epoch, self.val_rounds_since_improve, self.best_dice,
                            )
                            stop_early = True

                if epoch % self.cfg["train"]["ckpt_interval_epochs"] == 0:
                    save_checkpoint(
                        self.output_dir / f"epoch_{epoch:04d}.pt",
                        model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, scaler=self.scaler,
                        epoch=epoch,
                    )
                    # Rotate: keep only the 2 most recent epoch checkpoints.
                    # /store filled at 22 TB during the earlier sweep with
                    # 20 x 3.8 GB checkpoints per model; best.pt is preserved
                    # separately and is what downstream eval uses.
                    epoch_ckpts = sorted(self.output_dir.glob("epoch_*.pt"))
                    for old in epoch_ckpts[:-2]:
                        try:
                            old.unlink()
                        except OSError as e:
                            self.log.warning("failed to remove %s: %s", old, e)

                if stop_early:
                    break

            self.log.info(
                "total time %.1f s, best dice %.4f (epochs %d-%d / %d)",
                time.time() - t_total, self.best_dice,
                self.start_epoch + 1, total_epochs, total_epochs,
            )
        finally:
            wandb.finish()
