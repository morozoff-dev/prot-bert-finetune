# src/train_pl.py
from __future__ import annotations

import glob
import os
import shutil
import subprocess

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from protein_stability.data.preprocessing import add_cv_folds, preprocess_train_data
from protein_stability.train.callbacks import PlotAndArtifactsCallback
from protein_stability.train.pl_datamodule import ProteinDataModule
from protein_stability.train.pl_module import ProteinLightningModule
from protein_stability.utils.utils import seed_everything


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
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
        try:
            mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            pass
        # git commit id как параметр и тег
        git_sha = _get_git_commit()
        if git_sha:
            try:
                mlf_logger.experiment.log_param(
                    mlf_logger.run_id, "git_commit", git_sha
                )
            except Exception:
                pass
            try:
                mlf_logger.experiment.set_tag(mlf_logger.run_id, "git_commit", git_sha)
            except Exception:
                pass

    # === директория чекпоинтов
    ckpt_dir = os.path.join(get_original_cwd(), cfg.model.model_weights_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    # === базовый чекпоинтер Lightning
    # сохраняем лучший по val_loss (save_top_k=1), без автоматического добавления метрик в имя
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{cfg.model.model.replace('/','-')}_fold{{fold}}",  # шаблон, ниже подставим fold
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    lrmon = LearningRateMonitor(logging_interval="step")
    plot_cb = PlotAndArtifactsCallback(
        plots_dir=cfg.logging.plots_dir, enable_mlflow=True
    )

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
        callbacks=[checkpoint_cb, lrmon, plot_cb],
        log_every_n_steps=max(1, cfg.logging.print_freq),
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    # === запуск по фолдам
    folds = [cfg.debug.debug_fold] if cfg.debug.fast_debug else cfg.training.trn_fold
    model_stub = cfg.model.model.replace("/", "-")

    for fold in folds:
        # Подставим номер фолда в имя файла чекпоинта
        checkpoint_cb.filename = f"{model_stub}_fold{fold}"

        # Почистим старые версии для этого фолда (чтобы Lightning не делал -v1, -v2)
        for p in glob.glob(os.path.join(ckpt_dir, f"{model_stub}_fold{fold}*.ckpt")):
            try:
                os.remove(p)
            except OSError:
                pass
        # Удалим фиксированный best, чтобы его корректно перезаписать по итогу
        fixed_best = os.path.join(ckpt_dir, f"{model_stub}_fold{fold}_best.ckpt")
        if os.path.exists(fixed_best):
            try:
                os.remove(fixed_best)
            except OSError:
                pass

        # Данные и модель
        dm = ProteinDataModule(cfg, train, fold)
        dm.setup()
        pl_module = ProteinLightningModule(cfg)

        # Тренировка
        trainer.fit(pl_module, datamodule=dm)

        # Лучший чекпоинт за ран — копируем в фиксированное имя для DVC
        best_path = checkpoint_cb.best_model_path
        if best_path and os.path.exists(best_path):
            shutil.copy2(best_path, fixed_best)

            # (опционально) подчистим все прочие .ckpt для этого фолда, кроме fixed_best
            for p in glob.glob(
                os.path.join(ckpt_dir, f"{model_stub}_fold{fold}*.ckpt")
            ):
                if os.path.abspath(p) != os.path.abspath(fixed_best):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # На этом этапе в outputs/best/ будут лежать ровно:
    #   <model>_foldN_best.ckpt  — по одному на каждый фолд (перезаписываются между запусками)
    # Их удобно трекать в DVC.


if __name__ == "__main__":
    main()
