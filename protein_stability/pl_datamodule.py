# src/pl_datamodule.py
from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.dataset import TrainDataset  # используй твой класс

# предполагается, что ты уже подготовил train_df с колонкой 'fold' извне


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_df, fold: int):
        super().__init__()
        self.cfg = cfg
        self.df = train_df
        self.fold = fold
        self.ds_train = None
        self.ds_valid = None

    def setup(self, stage: Optional[str] = None):
        tr = self.df[self.df["fold"] != self.fold].reset_index(drop=True)
        va = self.df[self.df["fold"] == self.fold].reset_index(drop=True)

        if self.cfg.debug.fast_debug:
            bs = max(1, int(self.cfg.training.batch_size))
            tr = tr.iloc[: max(bs * self.cfg.debug.debug_steps, 1)]
            va = va.iloc[: max(bs * self.cfg.debug.debug_val_steps, 1)]

        self.ds_train = TrainDataset(self.cfg, tr)
        self.ds_valid = TrainDataset(self.cfg, va)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.training.batch_size,
            shuffle=not self.cfg.debug.fast_debug,  # для воспроизводимости в debug можно false
            num_workers=self.cfg.model.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.model.num_workers,
            pin_memory=True,
            drop_last=False,
        )
