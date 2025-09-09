import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.preprocessing import add_cv_folds, preprocess_train_data
from src.trainer import train_loop
from src.utils import get_logger, get_score, seed_everything


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

    raw_csv = config.data_loading.train_data_path

    tokenizer = AutoTokenizer.from_pretrained(config.model.model)
    tokenizer.save_pretrained(config.model.tokenizer_dir)

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
                _oof_df = train_loop(train, fold, config, logger)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df, config, logger)
        oof_df = oof_df.reset_index(drop=True)
        logger.info("========== CV ==========")
        get_result(oof_df, config, logger)

    if config.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
