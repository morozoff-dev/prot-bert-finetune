import os
import random
import subprocess
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    filename,
):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_git_commit(default: str = "unknown") -> str:
    """Возвращает текущий git commit SHA. Если git недоступен — 'unknown'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode("utf-8").strip()
    except Exception:
        return default


class MetricRecorder:
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.score = []
        self.lr = []

    def add(self, epoch, tr, vl, sc, lr):
        # сохраняем как float, NaN пропускаем в графиках
        self.epochs.append(int(epoch))
        self.train_loss.append(float(tr) if tr is not None else np.nan)
        self.val_loss.append(float(vl) if vl is not None else np.nan)
        self.score.append(float(sc) if sc is not None else np.nan)
        self.lr.append(float(lr) if lr is not None else np.nan)

    def _plot_series(self, xs, ys, title, ylabel, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)

        fig = plt.figure()
        ax = plt.gca()
        if mask.sum() >= 1:
            ax.plot(xs[mask], ys[mask], marker="o")  # нарисует и одиночную точку
        else:
            ax.text(
                0.5,
                0.5,
                "no finite data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_plots(self, out_dir: str, prefix: str = "train"):
        """
        Рисует 3 графика: loss (train/val), score, lr.
        Возвращает dict {name: path}.
        """
        paths = {}
        if len(self.epochs) == 0:
            return paths

        os.makedirs(out_dir, exist_ok=True)
        xs = np.asarray(self.epochs, dtype=float)

        # --- общий график лоссов на одном поле
        p_loss = os.path.join(out_dir, f"{prefix}_loss.png")
        fig = plt.figure()
        ax = plt.gca()
        for series, label in [
            (self.train_loss, "train_loss"),
            (self.val_loss, "val_loss"),
        ]:
            ys = np.asarray(series, dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if mask.sum() >= 1:
                ax.plot(xs[mask], ys[mask], marker="o", label=label)

        if not ax.has_data():
            ax.text(
                0.5,
                0.5,
                "no finite data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Loss curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(p_loss, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["loss"] = p_loss

        # --- score
        p_score = os.path.join(out_dir, f"{prefix}_score.png")
        self._plot_series(self.epochs, self.score, "Score (MCRMSE)", "score", p_score)
        paths["score"] = p_score

        # --- lr
        p_lr = os.path.join(out_dir, f"{prefix}_lr.png")
        self._plot_series(self.epochs, self.lr, "Learning Rate", "lr", p_lr)
        paths["lr"] = p_lr

        return paths
