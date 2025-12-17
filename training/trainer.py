import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utility import get_device, save_checkpoint
from .scheduler import get_cosine_schedule_with_warmup


def train(
    model,
    dataset,
    batch_size,
    epochs,
    lr,
    warmup_steps,
    max_steps,
    save_path,
):
    device = get_device()
    model.to(device)
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler_fn = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max_steps
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    scaler = GradScaler()

    step = 0

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for input_ids, labels in loop:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(enabled=device.type == "cuda"):
                logits, _ = model(input_ids)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            step += 1
            lr_scale = scheduler_fn(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * lr_scale

            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

            if step >= max_steps:
                save_checkpoint(model, optimizer, step, save_path)
                return

        save_checkpoint(model, optimizer, step, save_path)
