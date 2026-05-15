def build_loss(cfg):
    name = cfg["train"]["loss"]["name"]
    if name != "dice_ce":
        raise ValueError(
            f"loss {name!r} not supported — base.yaml fixes loss=dice_ce for fair comparison"
        )
    from monai.losses import DiceCELoss
    p = cfg["train"]["loss"]
    return DiceCELoss(
        include_background=p.get("include_background", False),
        softmax=p.get("softmax", True),
        to_onehot_y=p.get("to_onehot_y", True),
    )
