import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold


def add_spaces(x: str) -> str:
    """Добавляет пробелы между аминокислотами: 'ACD' -> 'A C D'."""
    return " ".join(list(x))


def preprocess_train_data(
    raw_csv_path: str,
    remove_ct: int = 20,
    source_substr: str = "jin",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Читаем CSV, фильтруем по source, добавляем пробелы,
    выкидываем маленькие PDB-группы, рангово нормализуем target (dTm или ddG).
    Возвращает итоговый DataFrame.
    """
    train = pd.read_csv(raw_csv_path)
    if verbose:
        print("Train has shape:", train.shape)

    # 1) фильтр по источнику
    if "source" in train.columns:
        train = train.loc[train["source"].astype(str).str.contains(source_substr, na=False)]
        if verbose:
            print("After source filter:", train.shape)

    # 2) add_spaces для sequence и mutant_seq
    train["sequence"] = train["sequence"].map(add_spaces)
    train["mutant_seq"] = train["mutant_seq"].map(add_spaces)
    if verbose:
        print("After add_spaces:", train.shape)

    # 3) выкидываем PDB-группы с n <= REMOVE_CT
    REMOVE_CT = remove_ct
    train["n"] = train.groupby("PDB")["PDB"].transform("count")
    train = train.loc[train["n"] > REMOVE_CT]
    if verbose:
        print(f"After removing groups with <= {REMOVE_CT} mutations:", train.shape)

    # 4) ранговая нормализация таргета внутри каждой PDB-группы
    train["target"] = 0.5
    for p in train["PDB"].unique():
        target_col = "dTm"
        tmp = train.loc[train["PDB"] == p, "dTm"]
        if tmp.isna().sum() > len(tmp) / 2:
            target_col = "ddG"
        train.loc[train["PDB"] == p, "target"] = (
            rankdata(train.loc[train["PDB"] == p, target_col]) /
            len(train.loc[train["PDB"] == p, target_col])
        )
    train = train.reset_index(drop=True)

    if verbose:
        print("Unique mutation groups:", train["PDB"].nunique())
    return train


def add_cv_folds(
    train_df,
    n_splits: int,
    target_cols,
    group_col: str = "PDB",
    verbose: bool = False
):
    """
    Модифицирует train_df: добавляет столбец 'fold' (int) по GroupKFold.
    Возвращает тот же DataFrame.
    """
    Fold = GroupKFold(n_splits=n_splits)
    train_df = train_df.copy()
    for n, (train_index, val_index) in enumerate(
        Fold.split(train_df, train_df[target_cols], train_df[group_col])
    ):
        train_df.loc[val_index, "fold"] = int(n)
    train_df["fold"] = train_df["fold"].astype(int)

    if verbose:
        print(train_df.groupby("fold").size())

    return train_df
