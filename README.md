# protein-stability

## Описание проекта

Цель проекта — дообучение модели машинного обучения для предсказания
термостабильности вариантов ферментов на основе их аминокислотных
последовательностей. Термостабильность (измеряемая через температуру плавления,
tm) критически важна для промышленного применения ферментов, так как определяет
их устойчивость к высоким температурам и другим агрессивным условиям. Улучшение
стабильности ферментов позволит снизить затраты на производство и повысить
эффективность биотехнологических процессов (например, в производстве биоэтанола
или моющих средств).

Основные возможности и этапы:

- Предобработка данных (добавление фолдов, подготовка последовательностей,
  генерация позиций).
- Обучение модели с использованием **PyTorch Lightning**.
- Логирование экспериментов и метрик в **MLflow**.
- Отслеживание и версионирование артефактов через **DVC**.
- Предсказание на новых данных в двух режимах:
  - Kaggle-совместимый формат (`protein_sequence`, `seq_id`).
  - Свободный формат (`sequence`, `mutant_seq`, `position`).

---

## Технические детали

### Структура проекта (важные модули)

```
protein_stability/
  data/
    dataset.py           # TrainDataset / TestDataset
    preprocessing.py     # функции подготовки данных и разбиения на фолды
  models/
    model.py             # CustomModel
    losses.py            # RMSELoss и др.
  train/
    train_pl.py          # точка входа обучения (Lightning)
    pl_module.py
    pl_datamodule.py
    callbacks.py
  infer/
    predict.py           # точка входа инференса
  utils/
    helpers.py           # prepare_input и пр.
    utils.py             # seed_everything, get_score, логгеры и т.д.
conf/
outputs/
plots/
```

---

### Setup

1. Клонировать репозиторий:

   ```bash
   git clone <url>
   cd protein-stability
   ```

2. Установить Poetry и зависимости:

   ```bash
   poetry install
   ```

3. Настроить **pre-commit** (автопроверка стиля):

   ```bash
   poetry run pre-commit install
   # по желанию — проверить все файлы сразу
   poetry run pre-commit run -a
   ```

4. Настроить **DVC** и подтянуть данные и модели:

   ```bash
   dvc pull -r data     # скачать датасеты
   dvc pull -r models   # скачать модели (чекпоинты .ckpt)
   ```

   > Для удалённого Google Drive потребуется `./.dvc/gdrive-service.json`.
   > Попросите файл у владельца репозитория и положите его по указанному пути.

5. Запустить **MLflow** сервер (локально):

   ```bash
   poetry run mlflow server \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns \
     --host 127.0.0.1 --port 8080
   ```

---

### Train

Запуск обучения через **PyTorch Lightning**:

```bash
poetry run python -m protein_stability.train.train_pl
```

Полезные параметры (Hydra):

- `debug.fast_debug=true` — отладочный режим.
- `training.trn_fold="[0,1,2,3,4]"` — список фолдов.
- `logging.mlflow.enable=true` — включить логирование в MLflow.

Пример запуска с MLflow:

```bash
poetry run python -m protein_stability.train.train_pl \
  logging.mlflow.enable=true \
  logging.mlflow.tracking_uri=http://127.0.0.1:8080 \
  logging.mlflow.experiment=protein-stability \
  logging.mlflow.run_name=train-esm2 \
  training.trn_fold="[0,1,2,3,4]"
```

После обучения в `outputs/best/` появятся файлы:

```
<model>_foldN_best.ckpt
```

Их удобно версионировать через **DVC**.

Запушить новые чекпоинты в удалённое хранилище моделей:

```bash
dvc add outputs/best/*.ckpt
git add outputs/best/*.dvc
git commit -m "add new checkpoints"
dvc push -r models
```

---

### Production preparation

Для деплоя:

- Использовать лучшие чекпоинты: `outputs/best/*_best.ckpt`.
- Обязательные артефакты:
  - чекпоинты моделей,
  - код (`protein_stability/`),
  - конфиги (`conf/`).

_(Опционально)_ Экспорт в ONNX/другие форматы при необходимости.

---

### Infer

Запуск предсказаний:

```bash
poetry run python -m protein_stability.infer.predict \
  infer.input_csv=data/test.csv \
  infer.output_csv=outputs/preds.csv
```

#### Поддерживаемые форматы данных

1. **Kaggle-совместимый**:

```csv
seq_id,protein_sequence
id1,MKVLWAALLVTFLAGCQAKVE...
id2,MKVLWAALLVTFLAGCQAKVF...
```

2. **Свободный формат**:

```csv
sequence,mutant_seq,position
M K V L W A A L L V T F L A G C Q A K V E, M K V L W A A L L V T F L A G C Q A K V F, 23
M K V L W A A L L V T F L A G C Q A K V E, M K V L W A A L L V T F L A G C Q A K I E, 45
```

**Выходной CSV**:

- для Kaggle: `seq_id,tm`
- для свободного формата: `sequence,mutant_seq,position,prediction`
