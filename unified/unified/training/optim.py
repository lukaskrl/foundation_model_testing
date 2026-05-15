import math


def build_optimizer(cfg, params):
    o = cfg["train"]["optimizer"]
    if o["name"] != "adamw":
        raise ValueError("base.yaml fixes optimizer=adamw for fair comparison")
    import torch
    return torch.optim.AdamW(
        params,
        lr=o["lr"],
        weight_decay=o["weight_decay"],
    )


def build_scheduler(cfg, optimizer, steps_per_epoch: int):
    s = cfg["train"]["scheduler"]
    if s["name"] != "warmup_cosine":
        raise ValueError("base.yaml fixes scheduler=warmup_cosine for fair comparison")
    import torch
    epochs = cfg["train"]["epochs"]
    warmup_steps = s["warmup_epochs"] * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    base_lr = cfg["train"]["optimizer"]["lr"]
    min_lr = s["min_lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # interpolate between base_lr*cosine and min_lr
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
