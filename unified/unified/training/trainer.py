"""Shared training loop.

The loop is identical for every model. Add per-model differences via configs,
not here.
"""
from __future__ import annotations
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..utils import get_logger, save_checkpoint
from .loss import build_loss
from .optim import build_optimizer, build_scheduler


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, evaluator, output_dir):
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
        self.optimizer = build_optimizer(cfg, model.parameters())
        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = build_scheduler(cfg, self.optimizer, steps_per_epoch)
        self.amp = bool(cfg["train"].get("amp", True))
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        self.grad_clip = cfg["train"].get("grad_clip", 0.0)
        self.log = get_logger("trainer")
        self.best_dice = -1.0
        self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        self.global_step = 0

    def _train_epoch(self, epoch: int):
        self.model.train()
        losses = []
        total_steps = len(self.train_loader)
        pbar = tqdm(
            self.train_loader,
            total=total_steps,
            desc=f"epoch {epoch:>3d}/{self.cfg['train']['epochs']} [train]",
            dynamic_ncols=True,
            leave=False,
        )
        for step, batch in enumerate(pbar):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(image)
                    loss = self.loss_fn(logits, label)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(image)
                loss = self.loss_fn(logits, label)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            losses.append(loss.item())
            self.global_step += 1

            lr = self.optimizer.param_groups[0]["lr"]
            self.tb.add_scalar("train/loss_step", loss.item(), self.global_step)
            self.tb.add_scalar("train/lr", lr, self.global_step)

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
        self.tb.add_scalar("val/mean_dice", mean_dice, epoch)
        for key, value in metrics.items():
            if key == "mean_dice" or value is None or value != value:  # skip NaN
                continue
            self.tb.add_scalar(f"val/{key}", value, epoch)
        # Persist a per-class metrics row.
        (self.output_dir / "val_metrics.jsonl").open("a").write(
            __import__("json").dumps({"epoch": epoch, **metrics}) + "\n"
        )
        return mean_dice

    def run(self):
        t_total = time.time()
        n_train = len(self.train_loader)
        n_val = len(self.val_loader) if self.val_loader is not None else 0
        self.log.info(
            "start: epochs=%d device=%s amp=%s train_batches=%d val_batches=%d",
            self.cfg["train"]["epochs"], self.device, self.amp, n_train, n_val,
        )
        try:
            for epoch in range(1, self.cfg["train"]["epochs"] + 1):
                t = time.time()
                train_loss = self._train_epoch(epoch)
                epoch_time = time.time() - t
                self.log.info("epoch %d done train_loss=%.4f dt=%.1fs",
                              epoch, train_loss, epoch_time)
                self.tb.add_scalar("train/loss_epoch", train_loss, epoch)
                self.tb.add_scalar("train/epoch_time_s", epoch_time, epoch)

                if epoch % self.cfg["train"]["val_interval_epochs"] == 0:
                    dice = self._validate(epoch)
                    if dice > self.best_dice and not (dice != dice):  # not NaN
                        self.best_dice = dice
                        self.tb.add_scalar("val/best_mean_dice", self.best_dice, epoch)
                        save_checkpoint(
                            self.output_dir / "best.pt",
                            model=self.model, optimizer=self.optimizer,
                            scheduler=self.scheduler, scaler=self.scaler,
                            epoch=epoch, extra={"mean_dice": dice},
                        )

                if epoch % self.cfg["train"]["ckpt_interval_epochs"] == 0:
                    save_checkpoint(
                        self.output_dir / f"epoch_{epoch:04d}.pt",
                        model=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, scaler=self.scaler,
                        epoch=epoch,
                    )

            self.log.info("total time %.1f s, best dice %.4f",
                          time.time() - t_total, self.best_dice)
        finally:
            self.tb.flush()
            self.tb.close()
