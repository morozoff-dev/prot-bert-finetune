# src/train_pl.py
from __future__ import annotations

import os
import subprocess

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.callbacks import PlotAndArtifactsCallback
from src.pl_datamodule import ProteinDataModule
from src.pl_module import ProteinLightningModule
from src.preprocessing import add_cv_folds, preprocess_train_data
from src.utils import seed_everything


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)

    # === подготовка train ===
    raw_csv = cfg.data_loading.train_data_path
    train = preprocess_train_data(
        raw_csv_path=raw_csv, remove_ct=20, source_substr="jin", verbose=False
    )
    train = add_cv_folds(
        train_df=train,
        n_splits=cfg.training.n_fold,
        target_cols=cfg.training.target_cols,
        group_col="PDB",
        verbose=False,
    )

    # === MLflow-логер
    mlf_logger = None
    if getattr(cfg.logging.mlflow, "enable", False) and getattr(
        cfg.logging.mlflow, "tracking_uri", None
    ):
        exp_name = (
            cfg.logging.mlflow.experiment or f"train-{cfg.model.model.replace('/','_')}"
        )
        run_name = cfg.logging.mlflow.run_name or (
            f"train-debug-fold{cfg.debug.debug_fold}"
            if cfg.debug.fast_debug
            else f"train-folds_{','.join(map(str, cfg.training.trn_fold))}"
        )
        mlf_logger = MLFlowLogger(
            experiment_name=exp_name,
            tracking_uri=cfg.logging.mlflow.tracking_uri,
            run_name=run_name,
        )
        # логируем гиперпараметры (весь конфиг)
        mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        # git commit id как параметр или тег
        git_sha = _get_git_commit()
        if git_sha:
            # как параметр
            try:
                mlf_logger.experiment.log_param(
                    mlf_logger.run_id, "git_commit", git_sha
                )
            except Exception:
                pass
            # и как тег (удобно для поиска)
            try:
                mlf_logger.experiment.set_tag(mlf_logger.run_id, "git_commit", git_sha)
            except Exception:
                pass

    # === чекпоинты — можно отключить, чтобы не было 2ГБ файлов
    # либо enable_checkpointing=False в Trainer,
    # либо save_top_k=0:
    ckpt_dir = os.path.join(get_original_cwd(), cfg.model.model_weights_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{cfg.model.model.replace('/','-')}" + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=0,  # <— не сохраняем модельные файлы
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    # === Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.debug.debug_epochs
        if cfg.debug.fast_debug
        else cfg.training.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if cfg.logging.apex else "32-true",
        gradient_clip_val=cfg.training.max_grad_norm,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        logger=(mlf_logger if mlf_logger is not None else False),
        callbacks=[
            ckpt,
            lrmon,
            PlotAndArtifactsCallback(
                plots_dir=cfg.logging.plots_dir, enable_mlflow=True
            ),
        ],
        log_every_n_steps=max(1, cfg.logging.print_freq),
        enable_progress_bar=True,
        enable_checkpointing=True,  # оставить True — но save_top_k=0 не будет сохранять файлы
    )

    # === запуск по фолдам
    folds = [cfg.debug.debug_fold] if cfg.debug.fast_debug else cfg.training.trn_fold
    for fold in folds:
        dm = ProteinDataModule(cfg, train, fold)
        dm.setup()
        pl_module = ProteinLightningModule(cfg)
        trainer.fit(pl_module, datamodule=dm)


if __name__ == "__main__":
    main()
