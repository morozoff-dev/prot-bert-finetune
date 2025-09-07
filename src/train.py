import os

import numpy as np
import pandas as pd
import wandb
from transformers import AutoTokenizer

from conf.config import CFG
from src.preprocessing import add_cv_folds, preprocess_train_data
from src.trainer import train_loop
from src.utils import LOGGER, get_score


def get_result(oof_df):
    """
    Подсчёт финального CV score.
    """
    pred_cols = [f"pred_{c}" for c in CFG.target_cols]
    mask = ~np.any(oof_df[pred_cols].isna().values, axis=1)
    if mask.sum() == 0:
        LOGGER.info("[DEBUG] Skip scoring: no complete predictions yet")
        return
    labels = oof_df.loc[mask, CFG.target_cols].values
    preds = oof_df.loc[mask, pred_cols].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f"Score: {score:<.4f}  Scores: {scores}  (on {mask.sum()} rows)")


def main():
    # === подготовка train ===
    raw_csv = "data/all_train_data_v17.csv"
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Не найден {raw_csv}")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(CFG.path + "tokenizer/")
    CFG.tokenizer = tokenizer

    train = preprocess_train_data(
        raw_csv_path=raw_csv,
        remove_ct=20,
        source_substr="jin",
        verbose=False,
    )
    train = add_cv_folds(
        train_df=train,
        n_splits=CFG.n_fold,
        target_cols=CFG.target_cols,
        group_col="PDB",
        verbose=False,
    )

    # === запуск обучения ===
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info("========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(CFG.path + "oof_df.pkl")

    if CFG.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
