import gc
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import wandb

from conf.config import CFG
from src.dataset import TrainDataset
from src.losses import RMSELoss
from src.model import CustomModel
from src.helpers import train_fn, valid_fn
from src.utils import LOGGER, get_score

def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    print('### train shape:',train_folds.shape)
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)


    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG.path +'config.pth')
    INIT_CKPT = CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_init.pth"
    torch.save({'model': model.state_dict()}, INIT_CKPT)
    LOGGER.info(f"Saved init checkpoint -> {INIT_CKPT}")

    
    # FREEZE LAYERS
    if CFG.num_freeze_blocks>0:
        print(f'### Freezing first {CFG.num_freeze_blocks} blocks.',
              f'Leaving {CFG.total_blocks-CFG.num_freeze_blocks} blocks unfrozen')
        for name, param in list(model.named_parameters())\
            [:CFG.initial_layers+CFG.layers_per_block*CFG.num_freeze_blocks]:     
                param.requires_grad = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = RMSELoss(reduction="mean") #nn.SmoothL1Loss(reduction='mean')
    
    best_score = np.inf

    for epoch in range(CFG.debug_epochs if CFG.fast_debug else CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions_part, n_seen = valid_fn(valid_loader, model, criterion, device)

        if CFG.fast_debug:
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
                LOGGER.info(f"[DEBUG] Partial Score on first {take}: {pscore:.4f}  Scores: {pscores}")
            except Exception as e:
                LOGGER.info(f"[DEBUG] Partial Score skipped: {e}")

            # сохранить веса (и опционально частичные предикты)
            torch.save({'model': model.state_dict(),
                        'predictions_partial': full_preds,
                        'n_seen': int(take)},
                    CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

            # положить колонки в DF (частично)
            for i, c in enumerate(CFG.target_cols):
                valid_folds[f"pred_{c}"] = full_preds[:, i] if full_preds.ndim > 1 else full_preds[:, 0]

        else:
            # Обычный путь на полном валиде
            score, scores = get_score(valid_labels, predictions_part)
            elapsed = time.time() - start_time
            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
            if CFG.wandb:
                wandb.log({f"[fold{fold}] epoch": epoch+1, 
                        f"[fold{fold}] avg_train_loss": avg_loss, 
                        f"[fold{fold}] avg_val_loss": avg_val_loss,
                        f"[fold{fold}] score": score})
            if best_score > score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': predictions_part},
                        CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
                # и положим в DF для этого фолда
                for i, c in enumerate(CFG.target_cols):
                    valid_folds[f"pred_{c}"] = predictions_part[:, i] if predictions_part.ndim > 1 else predictions_part[:, 0]

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds