import numpy as np
import torch
from torch.utils.data import Dataset

from src.helpers import prepare_input


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts1 = df["sequence"].values
        self.texts2 = df["mutant_seq"].values
        self.labels = df[cfg.training.target_cols].values
        self.position = df["position"].values

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        inputs1 = prepare_input(self.cfg, self.texts1[item])
        inputs2 = prepare_input(self.cfg, self.texts2[item])
        position = np.zeros(self.cfg.training.max_len)
        position[self.position[item]] = 1
        position = torch.tensor(position, dtype=torch.int)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs1, inputs2, position, label


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts1 = df["sequence"].values
        self.texts2 = df["mutant_seq"].values
        self.position = df["position"].values

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, item):
        inputs1 = prepare_input(self.cfg, self.texts1[item])
        inputs2 = prepare_input(self.cfg, self.texts2[item])
        position = np.zeros(self.cfg.training.max_len)
        position[self.position[item]] = 1
        position = torch.tensor(position, dtype=torch.int)
        return inputs1, inputs2, position
