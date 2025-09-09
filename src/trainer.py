import gc
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.dataset import TrainDataset
from src.helpers import train_fn, valid_fn
from src.losses import RMSELoss
from src.model import CustomModel
from src.utils import get_score


def train_loop(folds, fold, config, logger, mlflow_run=None):
    logger.info(f"========== fold: {fold} training ==========")

    # =======================
    # loaders
    # =======================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds[config.training.target_cols].values
    print("### train shape:", train_folds.shape)

    train_dataset = TrainDataset(config, train_folds)
    valid_dataset = TrainDataset(config, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.model.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # =======================
    # model & optimizer
    # =======================
    model = CustomModel(config, config_path=None, pretrained=True)
    torch.save(model.config, config.model.config_path)

    # freeze encoder blocks
    if config.model.num_freeze_blocks > 0:
        print(
            f"### Freezing first {config.model.num_freeze_blocks} blocks.",
            f"Leaving {config.model.total_blocks - config.model.num_freeze_blocks} blocks unfrozen",
        )
        init_layers = config.model.initial_layers
        layers_len = (
            init_layers + config.model.layers_per_block * config.model.num_freeze_blocks
        )
        for _, param in list(model.named_parameters())[:layers_len]:
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=config.training.encoder_lr,
        decoder_lr=config.training.decoder_lr,
        weight_decay=config.training.weight_decay,
    )
    optimizer = AdamW(
        optimizer_parameters,
        lr=config.training.encoder_lr,
        eps=config.training.eps,
        betas=config.training.betas,
    )

    # =======================
    # scheduler
    # =======================
    def get_scheduler(config, optimizer, num_train_steps):
        if config.model.scheduler == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.model.num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        elif config.model.scheduler == "constant":
            return get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=config.model.num_warmup_steps
            )
        elif config.model.scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.model.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=config.training.num_cycles,
            )
        else:
            return None

    num_train_steps = int(
        len(train_folds) / config.training.batch_size * config.training.epochs
    )
    scheduler = get_scheduler(config, optimizer, num_train_steps)

    # =======================
    # loop
    # =======================
    criterion = RMSELoss(reduction="mean")

    history = []  # (epoch, train_loss, val_loss, score, lr)
    best_score = np.inf

    max_epochs = (
        config.debug.debug_epochs if config.debug.fast_debug else config.training.epochs
    )
    for epoch in range(max_epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(
            fold,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
            config,
        )

        # eval
        avg_val_loss, predictions_part, n_seen = valid_fn(
            valid_loader, model, criterion, device, config
        )

        # score (partial в fast_debug)
        if config.debug.fast_debug:
            full_preds = np.full((len(valid_folds), 1), np.nan, dtype=np.float32)
            take = min(n_seen, len(valid_folds), predictions_part.shape[0])
            if take > 0:
                full_preds[:take] = predictions_part[:take]
            labels_subset = valid_labels[:take]
            preds_subset = full_preds[:take]
            try:
                score, scores = get_score(labels_subset, preds_subset)
                logger.info(
                    f"[DEBUG] Partial Score on first {take}: {score:.4f}  Scores: {scores}"
                )
            except Exception as e:
                logger.info(f"[DEBUG] Partial Score skipped: {e}")
                score, scores = float("nan"), [float("nan")]

            # сохраним частичный чекпоинт (как было)
            model_wgths_pth = config.model.model_weights_path
            model = config.model.model
            torch.save(
                {
                    "model": model.state_dict(),
                    "predictions_partial": full_preds,
                    "n_seen": int(take),
                },
                model_wgths_pth + f"{model.replace('/', '-')}_fold{fold}_best.pth",
            )

            # предикты в DF (частичные)
            for i, c in enumerate(config.training.target_cols):
                valid_folds[f"pred_{c}"] = (
                    full_preds[:, i] if full_preds.ndim > 1 else full_preds[:, 0]
                )

        else:
            score, scores = get_score(valid_labels, predictions_part)

        # lr текущий
        try:
            last_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
        except Exception:
            last_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  "
            f"avg_val_loss: {avg_val_loss:.4f}  time: {int(elapsed):d}s"
        )
        logger.info(f"Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}")

        # ——— история для графиков и дальнейшего сохранения ———
        history.append(
            (
                epoch + 1,
                float(avg_loss) if np.isfinite(avg_loss) else float("nan"),
                float(avg_val_loss) if np.isfinite(avg_val_loss) else float("nan"),
                float(score) if np.isfinite(score) else float("nan"),
                float(last_lr),
            )
        )

        # ——— MLflow: логируем метрики КАЖДУЮ эпоху (даже в fast_debug) ———
        if mlflow_run is not None:
            import mlflow

            step = epoch + 1
            if np.isfinite(avg_loss):
                mlflow.log_metric("train_loss", float(avg_loss), step=step)
            if np.isfinite(avg_val_loss):
                mlflow.log_metric("val_loss", float(avg_val_loss), step=step)
            if np.isfinite(score):
                mlflow.log_metric("score", float(score), step=step)
            mlflow.log_metric("lr", float(last_lr), step=step)

        # ——— сохранение лучшего чекпоинта ———
        if not config.debug.fast_debug:
            if best_score > score:
                best_score = score
                logger.info(
                    f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model"
                )
                model_wgths_pth = config.model.model_weights_path
                model = config.model.model
                torch.save(
                    {"model": model.state_dict(), "predictions": predictions_part},
                    model_wgths_pth + f"{model.replace('/', '-')}_fold{fold}_best.pth",
                )
                # предикты в DF (полные)
                for i, c in enumerate(config.training.target_cols):
                    valid_folds[f"pred_{c}"] = (
                        predictions_part[:, i]
                        if predictions_part.ndim > 1
                        else predictions_part[:, 0]
                    )

    torch.cuda.empty_cache()
    gc.collect()
    return valid_folds, history
