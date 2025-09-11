import os

import hydra
import mlflow
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig

from protein_stability.preprocessing import add_cv_folds, preprocess_train_data
from protein_stability.trainer import train_loop
from protein_stability.utils import (
    MetricRecorder,
    get_git_commit,
    get_logger,
    get_score,
    seed_everything,
)


def get_result(oof_df, config, logger):
    """
    Подсчёт финального CV score.
    """
    pred_cols = [f"pred_{c}" for c in config.training.target_cols]
    mask = ~np.any(oof_df[pred_cols].isna().values, axis=1)
    if mask.sum() == 0:
        logger.info("[DEBUG] Skip scoring: no complete predictions yet")
        return
    labels = oof_df.loc[mask, config.training.target_cols].values
    preds = oof_df.loc[mask, pred_cols].values
    score, scores = get_score(labels, preds)
    logger.info(f"Score: {score:<.4f}  Scores: {scores}  (on {mask.sum()} rows)")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig):
    seed_everything(config.training.seed)
    logger = get_logger(config.logging.train_log_path)

    # --- MLflow init ---
    mlflow_run = None
    if config.logging.mlflow.enable:
        mlflow.set_tracking_uri(config.logging.mlflow.tracking_uri)
        mlflow.set_experiment(config.logging.mlflow.experiment)
        mlflow_run = mlflow.start_run(run_name=config.logging.mlflow.run_name)

        # MLflow не любит вложенность/слишком большие dict — но обычно ок.
        mlflow.log_params({f"training.{k}": v for k, v in config.training.items()})
        mlflow.log_param("model.model", config.model.model)
        mlflow.log_param("git_sha", get_git_commit())

    # где хранить графики
    plots_dir = config.logging.plots_dir

    raw_csv = config.data_loading.train_data_path

    # tokenizer = AutoTokenizer.from_pretrained(config.model.model)
    # tokenizer.save_pretrained(config.model.tokenizer_dir)

    train = preprocess_train_data(
        raw_csv_path=raw_csv,
        remove_ct=20,
        source_substr="jin",
        verbose=False,
    )
    train = add_cv_folds(
        train_df=train,
        n_splits=config.training.n_fold,
        target_cols=config.training.target_cols,
        group_col="PDB",
        verbose=False,
    )

    # === запуск обучения ===
    if config.training.train:
        oof_df = pd.DataFrame()
        for fold in range(config.training.n_fold):
            if fold in config.training.trn_fold:
                # recorder для графиков по конкретному фолду
                recorder = MetricRecorder()
                _oof_df, history = train_loop(
                    train, fold, config, logger, mlflow_run=mlflow_run
                )
                oof_df = pd.concat([oof_df, _oof_df])

                for (ep, tr, vl, sc, lr) in history:
                    recorder.add(ep, tr, vl, sc, lr)

                paths = {}
                if len(recorder.epochs) > 0:
                    paths = recorder.save_plots(plots_dir, prefix=f"fold{fold}")
                    if mlflow_run is not None:
                        for p in paths.values():
                            mlflow.log_artifact(p)

                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, config, logger)

    oof_df = oof_df.reset_index(drop=True)
    logger.info("========== CV ==========")
    get_result(oof_df, config, logger)

    # можно сохранить общий OOF и залогировать в MLflow
    oof_path = os.path.join(plots_dir, "oof_df.pkl")
    oof_df.to_pickle(oof_path)
    if mlflow_run is not None:
        mlflow.log_artifact(oof_path)

    if config.logging.wandb:
        wandb.finish()

    if config.logging.mlflow.enable:
        mlflow.end_run()


if __name__ == "__main__":
    main()
