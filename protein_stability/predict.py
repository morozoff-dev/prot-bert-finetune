# src/predict.py
from __future__ import annotations

import gc
import os
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from protein_stability.dataset import TestDataset
from protein_stability.model import CustomModel


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for (inputs1, inputs2, position) in tk0:
        for k, v in inputs1.items():
            inputs1[k] = v.to(device)
        for k, v in inputs2.items():
            inputs2[k] = v.to(device)
        position = position.to(device)
        with torch.no_grad():
            y_preds = model(inputs1, inputs2, position)
        preds.append(y_preds.to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions


# =========================
# УТИЛИТЫ ПОДГОТОВКИ ДАННЫХ
# =========================
def add_spaces(x: str) -> str:
    return " ".join(list(x)) if isinstance(x, str) else x


def first_diff_pos(wt: str, mut: str) -> int:
    """Возвращает 1-базную позицию первой разницы (как в ноутбуке)."""
    s = wt.replace(" ", "")
    m = mut.replace(" ", "")
    i = 0
    L = min(len(s), len(m))
    while i < L and s[i] == m[i]:
        i += 1
    return i + 1  # 1-based


def ensure_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Если нет колонки position — создаём её из пары (sequence, mutant_seq)."""
    if "position" in df.columns:
        return df
    pos = []
    for s, m in zip(df["sequence"].tolist(), df["mutant_seq"].tolist()):
        pos.append(first_diff_pos(s, m))
    out = df.copy()
    out["position"] = pos
    return out


def make_kaggle_test_like(df: pd.DataFrame, base_wt: str) -> pd.DataFrame:
    """
    Преобразует test.csv (protein_sequence/seq_id) в (sequence, mutant_seq, position).
    """

    def get_test_mutation(row):
        for i, (a, b) in enumerate(zip(row.protein_sequence, base_wt)):
            if a != b:
                row["wildtype"] = base_wt[i]
                row["mutation"] = row.protein_sequence[i]
                row["position"] = i + 1
                break
        return row

    df = df.apply(get_test_mutation, axis=1)
    df = df.copy()
    df["sequence"] = base_wt
    df["sequence"] = df["sequence"].map(add_spaces)
    df = df.rename(columns={"protein_sequence": "mutant_seq"})
    df["mutant_seq"] = df["mutant_seq"].map(add_spaces)
    return df


# =========================
# PIPELINE HELPERS
# =========================
def load_folds(cfg: DictConfig) -> List[int]:
    """Список фолдов с учётом fast_debug."""
    if cfg.debug.fast_debug:
        return [cfg.debug.debug_fold]
    return list(cfg.training.trn_fold)


def detect_kaggle_input(df: pd.DataFrame, force_flag: bool) -> bool:
    if force_flag:
        return True
    return {"protein_sequence", "seq_id"} <= set(df.columns)


def prepare_dataframe(cfg: DictConfig) -> Tuple[pd.DataFrame, bool]:
    df_head = pd.read_csv(cfg.infer.input_csv, nrows=1)
    is_kaggle = detect_kaggle_input(df_head, cfg.infer.as_kaggle_test)

    # читаем весь файл
    df = pd.read_csv(cfg.infer.input_csv)
    if is_kaggle:
        df = make_kaggle_test_like(df, base_wt=cfg.infer.base_wt)
    else:
        needed = {"sequence", "mutant_seq"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(
                f"Входной CSV должен содержать {needed}. Отсутствуют: {missing}"
            )
        df = df.copy()
        df["sequence"] = df["sequence"].map(add_spaces)
        df["mutant_seq"] = df["mutant_seq"].map(add_spaces)
        df = ensure_positions(df)

    if cfg.debug.fast_debug:
        df = df.iloc[:16].reset_index(drop=True)

    return df, is_kaggle


def _load_weights_into_model(model: torch.nn.Module, ckpt_path: str):
    """
    Поддерживает:
      - Lightning .checkpoint/.ckpt (ключ 'state_dict', имена с префиксом 'model.')
      - Старые .pth (ключ 'model' со state_dict модели)
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        # Lightning checkpoint: вытащим только веса self.model.* и снимем префикс 'model.'
        pl_sd = state["state_dict"]
        cleaned = {}
        for k, v in pl_sd.items():
            if k.startswith("model."):
                cleaned[k[len("model.") :]] = v
        model.load_state_dict(cleaned, strict=False)
    elif isinstance(state, dict) and "model" in state:
        # Старый формат
        model.load_state_dict(state["model"], strict=True)
    else:
        # На всякий случай — пробуем загрузить как есть
        model.load_state_dict(state, strict=False)


def _resolve_ckpt_path(checkpoint_dir: str, base_name: str, fold: int) -> str:
    """
    Пробуем .ckpt, иначе старый _best.pth.
    """
    candidates = [
        os.path.join(checkpoint_dir, f"{base_name}_fold{fold}_best.ckpt"),
        os.path.join(checkpoint_dir, f"{base_name}_fold{fold}_best.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Не найден чекпоинт для фолда {fold}. Искомые файлы:\n" + "\n".join(candidates)
    )


def load_and_predict(cfg: DictConfig, df: pd.DataFrame, folds: List[int]) -> np.ndarray:
    checkpoint_dir = cfg.model.model_weights_path
    base_name = cfg.model.model.replace("/", "-")

    test_dataset = TestDataset(cfg, df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.model.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds_folds = []

    for fold in folds:
        model = CustomModel(cfg, config_path=cfg.model.config_path, pretrained=False)

        ckpt_path = _resolve_ckpt_path(checkpoint_dir, base_name, fold)
        _load_weights_into_model(model, ckpt_path)

        preds = inference_fn(test_loader, model, device)
        preds_folds.append(preds.astype(np.float32))

        del model, preds
        gc.collect()
        torch.cuda.empty_cache()

    predictions = np.mean(preds_folds, axis=0)
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    return predictions


def write_output(
    cfg: DictConfig, df_in: pd.DataFrame, preds: np.ndarray, is_kaggle: bool
):
    if is_kaggle:
        out = df_in.copy()
        out["tm"] = preds
        if "seq_id" in out.columns:
            out = out[["seq_id", "tm"]]
        out.to_csv(cfg.infer.output_csv, index=False)
    else:
        out = df_in.copy()
        out["prediction"] = preds
        out.to_csv(cfg.infer.output_csv, index=False)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1) подготовить датафрейм
    df, is_kaggle = prepare_dataframe(cfg)

    # 2) фолды
    folds = load_folds(cfg)

    # 3) инференс
    preds = load_and_predict(cfg, df, folds)

    # 4) записать вывод
    write_output(cfg, df, preds, is_kaggle)
    print(f"[OK] Saved predictions to {cfg.infer.output_csv}")


if __name__ == "__main__":
    main()
