import gc
import time

import numpy as np
import torch
import wandb
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


def train_loop(folds, fold, config, logger):

    logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
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

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(config, config_path=None, pretrained=True)
    torch.save(model.config, config.model.config_path)

    # FREEZE LAYERS
    if config.model.num_freeze_blocks > 0:
        print(
            f"### Freezing first {config.model.num_freeze_blocks} blocks.",
            f"Leaving {config.model.total_blocks-config.model.num_freeze_blocks} blocks unfrozen",
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

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(config, optimizer, num_train_steps):
        if config.model.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.model.num_warmup_steps,
                num_training_steps=num_train_steps,
            )
        elif config.model.scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=config.model.num_warmup_steps
            )
        elif config.model.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.model.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=config.training.num_cycles,
            )
        return scheduler

    num_train_steps = int(
        len(train_folds) / config.training.batch_size * config.training.epochs
    )
    scheduler = get_scheduler(config, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = RMSELoss(reduction="mean")

    best_score = np.inf

    for epoch in range(
        config.debug.debug_epochs if config.debug.fast_debug else config.training.epochs
    ):

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

        if config.debug.fast_debug:
            # создаём "полный" массив предиктов под размер фолда и кладём первые n_seen
            full_preds = np.full((len(valid_folds), 1), np.nan, dtype=np.float32)
            take = min(n_seen, len(valid_folds), predictions_part.shape[0])
            if take > 0:
                full_preds[:take] = predictions_part[:take]

            # partial score (только по тем, кто предсказан)
            labels_subset = valid_labels[:take]
            preds_subset = full_preds[:take]
            try:
                pscore, pscores = get_score(labels_subset, preds_subset)
                logger.info(
                    f"[DEBUG] Partial Score on first {take}: {pscore:.4f}  Scores: {pscores}"
                )
            except Exception as e:
                logger.info(f"[DEBUG] Partial Score skipped: {e}")

            # сохранить веса (и опционально частичные предикты)
            cfg_pth = config.model.model_weights_path
            torch.save(
                {
                    "model": model.state_dict(),
                    "predictions_partial": full_preds,
                    "n_seen": int(take),
                },
                cfg_pth + f"{config.model.model.replace('/', '-')}_fold{fold}_best.pth",
            )

            # положить колонки в DF (частично)
            for i, c in enumerate(config.training.target_cols):
                valid_folds[f"pred_{c}"] = (
                    full_preds[:, i] if full_preds.ndim > 1 else full_preds[:, 0]
                )

        else:
            # Обычный путь на полном валиде
            score, scores = get_score(valid_labels, predictions_part)
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
            )
            logger.info(f"Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}")
            if config.logging.wandb:
                wandb.log(
                    {
                        f"[fold{fold}] epoch": epoch + 1,
                        f"[fold{fold}] avg_train_loss": avg_loss,
                        f"[fold{fold}] avg_val_loss": avg_val_loss,
                        f"[fold{fold}] score": score,
                    }
                )
            if best_score > score:
                best_score = score
                logger.info(
                    f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model"
                )
                cfg_pth = config.model.model_weights_path
                cfg_model = config.model.model
                torch.save(
                    {"model": model.state_dict(), "predictions": predictions_part},
                    cfg_pth + f"{cfg_model.replace('/', '-')}_fold{fold}_best.pth",
                )
                # и положим в DF для этого фолда
                for i, c in enumerate(config.training.target_cols):
                    valid_folds[f"pred_{c}"] = (
                        predictions_part[:, i]
                        if predictions_part.ndim > 1
                        else predictions_part[:, 0]
                    )

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
