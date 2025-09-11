import math
import time

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer


def prepare_input(cfg, text):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.training.max_len,
        padding="max_length",
        truncation=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, _v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(
    fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.logging.apex)
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    for step, (inputs1, inputs2, position, labels) in enumerate(train_loader):
        # inputs1 = collate(inputs1)
        for k, v in inputs1.items():
            inputs1[k] = v.to(device)
        # inputs2 = collate(inputs2)
        for k, v in inputs2.items():
            inputs2[k] = v.to(device)
        position = position.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.logging.apex):
            y_preds = model(inputs1, inputs2, position)
            loss = criterion(y_preds, labels)
        if cfg.training.gradient_accumulation_steps > 1:
            loss = loss / cfg.training.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.max_grad_norm
        )
        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.model.batch_scheduler:
                scheduler.step()
        if step % cfg.logging.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
        if cfg.logging.wandb:
            wandb.log(
                {
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0],
                }
            )

        if cfg.debug.fast_debug and (step + 1) >= cfg.debug.debug_steps:
            print(
                f"[DEBUG] Остановлен ранний выход после {cfg.debug.debug_steps} шагов"
            )
            break

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, cfg):
    """
    Возвращает:
      avg_val_loss: float (NaN, если не было ни одного батча)
      predictions : np.ndarray формы (N, C) или (0, C) при пустом наборе
      n_seen      : int — сколько объектов реально прошло через валид
    """
    model.eval()

    losses = AverageMeter()
    preds = []
    n_seen = 0
    start = time.time()

    # число таргетов (на случай пустого infer/раннего выхода)
    try:
        n_targets = max(1, len(getattr(cfg.training, "target_cols", [])))
    except Exception:
        n_targets = 1

    with torch.no_grad():
        for step, (inputs1, inputs2, position, labels) in enumerate(valid_loader):
            # на устройство
            for k, v in inputs1.items():
                inputs1[k] = v.to(device)
            for k, v in inputs2.items():
                inputs2[k] = v.to(device)
            position = position.to(device)
            labels = labels.to(device)

            bs = labels.size(0)

            # forward + loss
            y_preds = model(inputs1, inputs2, position)
            loss = criterion(y_preds, labels)
            loss_value = float(loss.detach().item())

            # Валидация не нуждается в делении на gradient_accumulation_steps,
            # но если хочешь сохранить семантику — оставь строку ниже закомментированной:
            # if cfg.training.gradient_accumulation_steps > 1:
            #     loss_value /= cfg.training.gradient_accumulation_steps

            losses.update(loss_value, bs)
            preds.append(y_preds.detach().cpu().numpy())
            n_seen += bs

            if step % cfg.logging.print_freq == 0 or step == (len(valid_loader) - 1):
                print(
                    "EVAL: [{0}/{1}] "
                    "Elapsed {remain:s} "
                    "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                        step,
                        len(valid_loader),
                        loss=losses,
                        remain=timeSince(
                            start, float(step + 1) / max(1, len(valid_loader))
                        ),
                    )
                )

            # ранний выход в debug-режиме
            if getattr(cfg.debug, "fast_debug", False) and (step + 1) >= getattr(
                cfg.debug, "debug_val_steps", 1
            ):
                print(f"[DEBUG] valid early-exit after {step+1} steps (seen={n_seen})")
                break

    # аккуратно склеиваем предсказания
    if preds:
        predictions = np.concatenate(preds, axis=0)
    else:
        # пустой валид/очень ранний выход: вернём корректной формы пустой массив
        predictions = np.empty((0, n_targets), dtype=np.float32)

    # средний лосс: если не было батчей — вернём NaN (train_loop это учтёт)
    avg_val_loss = (
        float(losses.avg) if getattr(losses, "count", 0) > 0 else float("nan")
    )

    return avg_val_loss, predictions, n_seen
