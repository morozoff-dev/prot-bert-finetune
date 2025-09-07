import argparse
import gc
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.dataset import TestDataset
from src.model import CustomModel
from src.utils import CFG


# =========================
# ИНФЕРЕНС (как в ноутбуке)
# =========================
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
        preds.append(y_preds.to('cpu').numpy())
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
    return i + 1  # 1-based, как в ноутбуке

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
    Преобразует test.csv из соревнования (protein_sequence/seq_id)
    в формат (sequence, mutant_seq, position), как в ноутбуке.
    """
    # вычисляем wildtype/mutation/position (позиция 1-базная)
    def get_test_mutation(row):
        for i, (a, b) in enumerate(zip(row.protein_sequence, base_wt)):
            if a != b:
                break
        row['wildtype'] = base_wt[i]
        row['mutation'] = row.protein_sequence[i]
        row['position'] = i + 1
        return row

    df = df.apply(get_test_mutation, axis=1)
    df = df.copy()
    df["sequence"] = base_wt
    df["sequence"] = df["sequence"].map(add_spaces)
    df = df.rename(columns={'protein_sequence': 'mutant_seq'})
    df["mutant_seq"] = df["mutant_seq"].map(add_spaces)
    return df


# =========================
# АРГУМЕНТЫ CLI
# =========================
DEFAULT_BASE = (
    "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSA"
    "QDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDP"
    "AKWAYQYDEKNNKFNYVGK"
)

def parse_args():
    p = argparse.ArgumentParser(description="Protein BERT inference (совместимо с ноутбуком + общий CSV)")
    p.add_argument("--input_csv", required=True,
                   help="Либо Kaggle test.csv (protein_sequence, seq_id), либо CSV с sequence, mutant_seq, [position]")
    p.add_argument("--output_csv", default="predictions.csv",
                   help="Выходной CSV. Для Kaggle-формата будет ['seq_id','tm'], для общего — добавится колонка 'prediction'.")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Папка с чекпоинтами *_foldX_best.pth. Если не задано — возьмётся CFG.path")
    p.add_argument("--folds", default=None,
                   help="Список фолдов через запятую, напр. '0,1,2'. Если не задано — CFG.trn_fold")
    p.add_argument("--batch_size", type=int, default=None, help="Переопределить CFG.batch_size для инференса")
    p.add_argument("--num_workers", type=int, default=None, help="Переопределить CFG.num_workers")
    p.add_argument("--as_kaggle_test", action="store_true",
                   help="Явно укажи, что вход — kaggle test.csv (иначе детектится по колонкам)")
    p.add_argument("--base_wt", default=DEFAULT_BASE,
                   help="Базовая WT-последовательность для Kaggle-режима (если нужна другая)")
    return p.parse_args()


# =========================
# ОСНОВНОЙ PIPELINE
# =========================
def load_folds(args) -> List[int]:
    if args.folds is not None:
        return [int(x) for x in args.folds.split(",") if x.strip() != ""]
    # по умолчанию — как в тренировке
    return list(CFG.trn_fold)

def detect_kaggle_input(df: pd.DataFrame, force_flag: bool) -> bool:
    if force_flag:
        return True
    return {"protein_sequence", "seq_id"} <= set(df.columns)

def prepare_dataframe(args) -> Tuple[pd.DataFrame, bool]:
    df = pd.read_csv(args.input_csv)
    is_kaggle = detect_kaggle_input(df, args.as_kaggle_test)

    if is_kaggle:
        # режим ноутбука: строим (sequence, mutant_seq, position) из protein_sequence + base_wt
        df = make_kaggle_test_like(df, base_wt=args.base_wt)
    else:
        # общий режим: ждём sequence, mutant_seq, опц. position
        needed = {"sequence", "mutant_seq"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Входной CSV должен содержать {needed}. Отсутствуют: {missing}")
        # гарантируем пробелы между АА (как в ноутбуке)
        df = df.copy()
        df["sequence"] = df["sequence"].map(add_spaces)
        df["mutant_seq"] = df["mutant_seq"].map(add_spaces)
        # если позиции нет — посчитаем как первую разницу
        df = ensure_positions(df)

    return df, is_kaggle

def load_and_predict(df: pd.DataFrame, folds: List[int], checkpoint_dir: Optional[str]) -> np.ndarray:
    # Переопределить параметры из CLI при необходимости
    if checkpoint_dir is None:
        checkpoint_dir = CFG.path

    # dataloader
    test_dataset = TestDataset(CFG, df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds_folds = []

    if CFG.fast_debug:
        folds = [CFG.debug_fold]
    else:
        folds = CFG.trn_fold

    for fold in folds:
        model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
        ckpt_path = os.path.join(checkpoint_dir, f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
        state = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state['model'])
        preds = inference_fn(test_loader, model, device)
        preds_folds.append(preds.astype(np.float32))
        del model, state, preds; gc.collect()
        torch.cuda.empty_cache()

    predictions = np.mean(preds_folds, axis=0)
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    return predictions

def write_output(df_in: pd.DataFrame, preds: np.ndarray, is_kaggle: bool, output_csv: str):
    if is_kaggle:
        # как в ноутбуке: колонка tm + seq_id
        out = df_in.copy()
        out["tm"] = preds
        # если в исходнике был seq_id — соберём сабмишн как в ноутбуке
        if "seq_id" in out.columns:
            out = out[["seq_id", "tm"]]
        out.to_csv(output_csv, index=False)
    else:
        # общий режим: просто добавим колонку prediction
        out = df_in.copy()
        out["prediction"] = preds
        out.to_csv(output_csv, index=False)

def main():
    args = parse_args()

    # опциональные переопределения размера батча/воркеров
    if args.batch_size is not None:
        CFG.batch_size = args.batch_size
    if args.num_workers is not None:
        CFG.num_workers = args.num_workers

    tokenizer = AutoTokenizer.from_pretrained(CFG.tokenizer_dir)
    CFG.tokenizer = tokenizer

    df, is_kaggle = prepare_dataframe(args)
    if CFG.fast_debug:
        df = df.iloc[:16]

    folds = load_folds(args)
    preds = load_and_predict(df, folds, args.checkpoint_dir)
    write_output(df, preds, is_kaggle, args.output_csv)
    print(f"[OK] Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
