import os
import random
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import numpy as np
import torch

from conf.config import CFG


def _rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE без sklearn, с защитой от NaN."""
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))


def MCRMSE(y_trues: np.ndarray, y_preds: np.ndarray):
    """
    Mean Columnwise RMSE.
    Работает, если у тебя 1 или несколько таргетов.
    Игнорит строки, где предикт NaN (частичные предикты в fast-режиме).
    """
    y_trues = np.asarray(y_trues, dtype=np.float32)
    y_preds = np.asarray(y_preds, dtype=np.float32)

    # выровняем формы до (N, C)
    if y_trues.ndim == 1:
        y_trues = y_trues[:, None]
    if y_preds.ndim == 1:
        y_preds = y_preds[:, None]

    C = y_trues.shape[1]
    scores = []

    for i in range(C):
        # маска валидных элементов (оба не NaN)
        mask = ~np.isnan(y_trues[:, i]) & ~np.isnan(y_preds[:, i])
        if mask.sum() == 0:
            scores.append(np.nan)  # нет ни одной валидной строки для этой колонки
        else:
            scores.append(_rmse_np(y_trues[mask, i], y_preds[mask, i]))

    mcrmse = float(np.nanmean(scores))  # среднее по колонкам, игноря NaN
    return mcrmse, scores


def get_score(y_trues: np.ndarray, y_preds: np.ndarray):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


def get_logger(
    filename=CFG.path + "train",
):  # только в трейне получается логгер используем
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)
