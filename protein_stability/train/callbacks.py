from __future__ import annotations

import io
import os
from typing import List

import matplotlib.pyplot as plt
import mlflow
from hydra.utils import get_original_cwd
from pytorch_lightning import Callback


class PlotAndArtifactsCallback(Callback):
    """
    Копит метрики по шагам и сохраняет 4 графика:
    - train_loss (по шагам)
    - val_loss (по шагам)
    - score (по шагам)
    - lr (по шагам)
    Логирует их и в локальную папку plots/, и в MLflow Артефакты.
    """

    def __init__(self, plots_dir: str = "plots", enable_mlflow: bool = True):
        super().__init__()
        self.enable_mlflow = enable_mlflow
        self.plots_dir_cfg = plots_dir

        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.score: List[float] = []
        self.lr: List[float] = []

    # собираем значения со шагов
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = trainer.logged_metrics  # «сырые» метрики на текущем шаге
        if "train_loss_step" in s:
            self.train_loss.append(float(s["train_loss_step"]))
        if "lr" in s:
            self.lr.append(float(s["lr"]))

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        s = trainer.logged_metrics
        if "val_loss" in s:
            self.val_loss.append(float(s["val_loss"]))
        if "score_step" in s:
            self.score.append(float(s["score_step"]))

    def _plot_series(self, y: List[float], title: str, ylabel: str):
        if not y:
            return None
        fig = plt.figure()
        plt.plot(range(1, len(y) + 1), y)
        plt.title(title)
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.tight_layout()
        return fig

    def on_fit_end(self, trainer, pl_module):
        # куда сохраняем локально
        root = get_original_cwd()
        plots_dir = os.path.join(root, self.plots_dir_cfg)
        os.makedirs(plots_dir, exist_ok=True)

        plots = [
            (
                "train_loss.png",
                self._plot_series(self.train_loss, "Train Loss (step)", "loss"),
            ),
            (
                "val_loss.png",
                self._plot_series(self.val_loss, "Val Loss (step)", "loss"),
            ),
            ("score.png", self._plot_series(self.score, "Score (step)", "MCRMSE")),
            ("lr.png", self._plot_series(self.lr, "Learning Rate", "lr")),
        ]

        for name, fig in plots:
            if fig is None:
                continue
            # 1) локально
            local_path = os.path.join(plots_dir, name)
            fig.savefig(local_path, dpi=140)
            # 2) в MLflow
            if self.enable_mlflow:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=140)
                buf.seek(0)
                mlflow.log_figure(fig, name)
            plt.close(fig)
