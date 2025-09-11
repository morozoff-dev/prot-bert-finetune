import numpy as np
import torch
from torch.utils.data import Dataset

from protein_stability.utils.helpers import prepare_input


def _prepare_positions(series, max_len: int) -> np.ndarray:
    """
    Приводит колонку position к целому индексу [0..max_len-1].
    Вход может быть 1-based (как в Kaggle-пайплайне), поэтому
    вычитаем 1. NaN заменяем на 1 (=> индекс 0 после -1).
    """
    pos = series.astype("Int64").fillna(1).astype(np.int64).values - 1
    pos = np.clip(pos, 0, max_len - 1)
    return pos


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts1 = df["sequence"].values
        self.texts2 = df["mutant_seq"].values
        self.labels = df[cfg.training.target_cols].values

        self.max_len = int(cfg.training.max_len)
        self.position = _prepare_positions(df["position"], self.max_len)

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        inputs1 = prepare_input(self.cfg, self.texts1[item])
        inputs2 = prepare_input(self.cfg, self.texts2[item])

        position_vec = torch.zeros(self.max_len, dtype=torch.float32)
        position_vec[int(self.position[item])] = 1.0

        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return inputs1, inputs2, position_vec, label


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts1 = df["sequence"].values
        self.texts2 = df["mutant_seq"].values

        self.max_len = int(cfg.training.max_len)
        self.position = _prepare_positions(df["position"], self.max_len)

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        inputs1 = prepare_input(self.cfg, self.texts1[item])
        inputs2 = prepare_input(self.cfg, self.texts2[item])

        position_vec = torch.zeros(self.max_len, dtype=torch.float32)
        position_vec[int(self.position[item])] = 1.0

        return inputs1, inputs2, position_vec
