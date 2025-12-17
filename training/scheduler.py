import math


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps,
    total_steps,
    min_lr_ratio=0.1,
):
    """
    Cosine LR with warmup.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return lambda step: lr_lambda(step)
