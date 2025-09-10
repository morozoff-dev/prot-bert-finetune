# ProtBERT Finetune

## Описание проекта

Этот проект посвящён дообучению большой языковой модели для работы с белковыми
последовательностями (например, Facebook ESM2 / ProtBERT). Задача —
предсказывать стабильность белковых мутаций (регрессионная задача).

Основные возможности и этапы:

- Предобработка данных (добавление фолдов, подготовка последовательностей,
  генерация позиций).
- Обучение модели с использованием PyTorch Lightning.
- Логирование экспериментов и метрик в MLflow.
- Отслеживание и версионирование артефактов через DVC.
- Предсказание на новых данных в двух режимах:
  - Kaggle-совместимый формат (`protein_sequence`, `seq_id`).
  - Свободный формат (`sequence`, `mutant_seq`, `position`).

---

## Технические детали

### Setup

1. Клонировать репозиторий:

   ```bash
   git clone <url>
   cd prot-bert-finetune
   ```

2. Установить Poetry и зависимости:

   ```bash
   poetry install
   ```

3. Настроить DVC и подтянуть данные и модели:

   ```bash
   dvc pull -r data     # скачать датасеты
   dvc pull -r models   # скачать модели (чекпоинты .ckpt)
   ```

4. Запустить MLflow сервер (локально):
   ```bash
   poetry run mlflow server        --backend-store-uri sqlite:///mlflow.db        --default-artifact-root ./mlruns        --host 127.0.0.1 --port 8080
   ```

---

### Train

Обучение модели запускается через PyTorch Lightning:

```bash
poetry run python -m src.train_pl
```

Полезные параметры:

- `debug.fast_debug=true` — отладочный режим.
- `training.trn_fold="[0,1,2,3,4]"` — список фолдов.
- `logging.mlflow.enable=true` — включить логирование в MLflow.

Пример запуска:

```bash
poetry run python -m src.train_pl   logging.mlflow.enable=true   logging.mlflow.tracking_uri=http://127.0.0.1:8080   logging.mlflow.experiment=protein-stability   logging.mlflow.run_name=train-esm2   training.trn_fold="[0,1,2,3,4]"
```

После обучения в `outputs/best/` появятся файлы:

```
<model>_foldN_best.ckpt
```

Их удобно версионировать через DVC.

Запушить новые чекпоинты в удалённое хранилище:

```bash
dvc add outputs/best/*.ckpt
git add outputs/best/*.dvc
git commit -m "add new checkpoints"
dvc push -r models
```

---

### Production preparation

Для деплоя:

- Использовать лучшие чекпоинты (`outputs/best/*_best.ckpt`).
- При необходимости — перевести модель в ONNX:
  ```bash
  torch.onnx.export(...)
  ```
- Обязательные артефакты:
  - чекпоинты моделей,
  - код (`src/`),
  - конфиги (`conf/`).

---

### Infer

Предсказания запускаются через:

```bash
poetry run python -m src.predict infer.input_csv=data/test.csv infer.output_csv=outputs/preds.csv
```

#### Поддерживаемые форматы данных

1. Kaggle-совместимый формат:

```csv
seq_id,protein_sequence
id1,MKVLWAALLVTFLAGCQAKVE...
id2,MKVLWAALLVTFLAGCQAKVF...
```

2. Свободный формат:

```csv
sequence,mutant_seq,position
M K V L W A A L L V T F L A G C Q A K V E, M K V L W A A L L V T F L A G C Q A K V F, 23
M K V L W A A L L V T F L A G C Q A K V E, M K V L W A A L L V T F L A G C Q A K I E, 45
```

Выходной CSV:

- для Kaggle: `seq_id,tm`
- для свободного формата: `sequence,mutant_seq,position,prediction`

---

## Итог

README покрывает:

- установку окружения (Poetry, DVC, MLflow),
- запуск обучения и предсказаний,
- использование DVC для управления данными и моделями,
- подготовку артефактов для продакшена,
- примеры входных/выходных данных.
