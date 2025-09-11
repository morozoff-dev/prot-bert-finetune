# src/pl_module.py
from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch

from protein_stability.models.losses import RMSELoss
from protein_stability.models.model import CustomModel
from protein_stability.utils.utils import get_score


class ProteinLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.model = CustomModel(cfg, config_path=None, pretrained=True)
        self.criterion = RMSELoss(reduction="mean")
        self.val_preds = []
        self.val_trues = []

        # freeze блоков
        if self.cfg.model.num_freeze_blocks > 0:
            init_layers = self.cfg.model.initial_layers
            layers_per_block = self.cfg.model.layers_per_block
            num_freeze_blocks = self.cfg.model.num_freeze_blocks
            upto = init_layers + layers_per_block * num_freeze_blocks
            for _, p in list(self.model.named_parameters())[:upto]:
                p.requires_grad = False

    def forward(self, inputs1, inputs2, position):
        return self.model(inputs1, inputs2, position)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        enc = [
            {
                "params": [
                    p
                    for n, p in self.model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": self.cfg.training.encoder_lr,
                "weight_decay": self.cfg.training.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": self.cfg.training.encoder_lr,
                "weight_decay": 0.0,
            },
        ]
        dec = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if "model" not in n
                ],
                "lr": self.cfg.training.decoder_lr,
                "weight_decay": 0.0,
            }
        ]
        optimizer = torch.optim.AdamW(
            enc + dec,
            lr=self.cfg.training.encoder_lr,
            eps=self.cfg.training.eps,
            betas=self.cfg.training.betas,
        )

        # scheduler
        total_steps = self.trainer.estimated_stepping_batches
        if self.cfg.model.scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            sch = get_linear_schedule_with_warmup(
                optimizer, self.cfg.model.num_warmup_steps, total_steps
            )
        elif self.cfg.model.scheduler == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            sch = get_cosine_schedule_with_warmup(
                optimizer,
                self.cfg.model.num_warmup_steps,
                total_steps,
                self.cfg.training.num_cycles,
            )
        else:
            from transformers import get_constant_schedule_with_warmup

            sch = get_constant_schedule_with_warmup(
                optimizer, self.cfg.model.num_warmup_steps
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sch, "interval": "step", "name": "lr"},
        }

    def training_step(self, batch, batch_idx):
        inputs1, inputs2, position, labels = batch
        preds = self(inputs1, inputs2, position)
        loss = self.criterion(preds, labels)

        # лог лосса
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        # self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # лог LR (по шагам)
        try:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log(
                "lr",
                float(lr),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
        except Exception:
            pass

        return loss

    def on_validation_epoch_start(self):
        self.val_preds.clear()
        self.val_trues.clear()

    def validation_step(self, batch, batch_idx):
        inputs1, inputs2, position, labels = batch
        with torch.no_grad():
            preds = self(inputs1, inputs2, position)
            loss = self.criterion(preds, labels)

        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        score_step, _ = get_score(labels_np, preds_np)

        self.val_preds.append(preds_np)
        self.val_trues.append(labels_np)

        # шаговые кривые
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "score_step",
            float(score_step),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        preds = np.concatenate(self.val_preds, axis=0)
        trues = np.concatenate(self.val_trues, axis=0)
        score_epoch, _ = get_score(trues, preds)
        self.log(
            "score",
            float(score_epoch),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
