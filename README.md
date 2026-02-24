# CV module: mice spinal cord feedback

Подробное описание текущего CV-модуля (на базе конфигурации DeepLabCut) и инструкции по встраиванию в Python-плагин с **локальными весами** и **локальными видео**.

## 1) Что это за модуль

В репозитории хранится конфигурация проекта pose-estimation (DeepLabCut):
- задача/проект: `r_tm_side`
- архитектура сети: `resnet_50`
- batch size: `8`
- порог уверенности: `pcutoff = 0.6`
- список ключевых точек (bodyparts): 15 штук
- мультиживотные: `false` (single-animal pipeline)

Основной файл: `config.yaml`.

---

## 2) Какие ключевые точки детектируются

Текущий список bodyparts:

- `nose`
- `eye_l`, `eye_r`
- `fl_toes_l`, `fl_toes_r`
- `hl_toes_l`, `hl_ankle_l`, `hl_hip_l`, `hl_iliac_l`
- `hl_toes_r`, `hl_ankle_r`, `hl_hip_r`, `hl_iliac_r`
- `spine`
- `tail`

Это важно при интеграции в плагин: downstream-логика (метрики, биомеханика, обратная связь) должна опираться на **эти точные имена ключевых точек**.

---

## 3) Что нужно для встраивания в Python-плагин

Минимальный набор:

1. `config.yaml` (из этого репозитория)
2. локальная директория с весами/снимками обучения (snapshots/exported model)
3. локальные видео (`.avi` или другой поддерживаемый формат)
4. установленный Python + зависимости DeepLabCut/TensorFlow

### Рекомендуемая структура директорий

```text
my_plugin/
  __init__.py
  cv_runner.py
  models/
    mice_spinal_cord/
      config.yaml
      dlc-models/
        iteration-0/
          r_tm_sideOct25-trainset95shuffle1/
            train/
              snapshot-XXXXXX.meta
              snapshot-XXXXXX.index
              snapshot-XXXXXX.data-00000-of-00001
  videos/
    input_001.avi
    input_002.avi
  outputs/
```

> Если у вас модель экспортирована иначе (например, TensorFlow SavedModel), храните её в `models/mice_spinal_cord/exported/` и адаптируйте вызов инференса.

---

## 4) Важные нюансы текущего `config.yaml`

В конфиге есть пути формата Windows (`G:\...`) в `project_path` и `video_sets`. Для плагина это обычно не подходит, потому что:
- среда запуска может быть Linux или контейнер;
- видео должны подбираться динамически, а не из фиксированного списка обучающего проекта.

### Что делать правильно

- Использовать `config.yaml` как базовый шаблон для модели/ключевых точек.
- Для runtime-инференса передавать **локальный путь к конкретному видео** через аргументы плагина.
- Не полагаться на `video_sets` при обработке новых файлов.

---

## 5) Пример интеграции в Python-плагин

Ниже пример адаптера, который можно встроить в ваш плагин.

```python
from pathlib import Path
from typing import Optional
import deeplabcut


class MiceSpinalCordCV:
    """
    Обёртка над DeepLabCut-моделью для инференса на локальных видео.
    """

    def __init__(
        self,
        config_path: str | Path,
        pcutoff: float = 0.6,
        gputouse: Optional[int] = 0,
    ):
        self.config_path = str(Path(config_path).resolve())
        self.pcutoff = pcutoff
        self.gputouse = gputouse

    def analyze_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        save_as_csv: bool = True,
        dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    ) -> None:
        video_path = str(Path(video_path).resolve())
        output_dir = str(Path(output_dir).resolve())

        deeplabcut.analyze_videos(
            config=self.config_path,
            videos=[video_path],
            destfolder=output_dir,
            save_as_csv=save_as_csv,
            dynamic=dynamic,
            gputouse=self.gputouse,
        )

    def filter_predictions(
        self,
        video_path: str | Path,
        output_dir: str | Path,
    ) -> None:
        """Опционально сглаживает предсказания перед дальнейшим анализом."""
        deeplabcut.filterpredictions(
            self.config_path,
            [str(Path(video_path).resolve())],
            destfolder=str(Path(output_dir).resolve()),
        )
```

### Вызов из plugin entrypoint

```python
def run_plugin(video_file: str, work_dir: str):
    model_config = f"{work_dir}/models/mice_spinal_cord/config.yaml"
    out_dir = f"{work_dir}/outputs"

    cv = MiceSpinalCordCV(config_path=model_config, pcutoff=0.6, gputouse=0)
    cv.analyze_video(video_file, out_dir)
    cv.filter_predictions(video_file, out_dir)
```

---

## 6) Контракт входа/выхода для плагина

### Вход
- `video_path`: путь к локальному видео.
- `config_path`: путь к локальному `config.yaml`.
- `weights_dir`/snapshots: локальная директория весов, на которую ссылается DLC-проект.

### Выход
Обычно DeepLabCut формирует:
- `.h5` с координатами keypoints (`x`, `y`, `likelihood`)
- `.csv` (если `save_as_csv=True`)
- служебные артефакты инференса

Рекомендуется после инференса нормализовать результат в внутренний формат плагина, например:

```json
{
  "frame": 120,
  "keypoints": {
    "nose": {"x": 123.4, "y": 51.2, "p": 0.98},
    "spine": {"x": 140.1, "y": 70.5, "p": 0.91}
  }
}
```

---

## 7) Практические рекомендации для production

1. **Пинning версий**: зафиксируйте версии `deeplabcut`, `tensorflow`, `numpy`, `pandas`.
2. **GPU/CPU fallback**: если GPU недоступен, предусмотрите запуск на CPU (медленнее).
3. **Валидация входов**: проверять существование файла видео и `config.yaml` до запуска инференса.
4. **Логи и тайминги**: логировать длительность на видео/кадр и процент кадров с низким `likelihood`.
5. **Постобработка**: фильтрация, интерполяция пропусков, отбрасывание точек ниже порога доверия.
6. **Стабильные имена точек**: downstream-код должен опираться на фиксированный словарь bodyparts.

---

## 8) Быстрый чек-лист интеграции

- [ ] `config.yaml` лежит локально и доступен на чтение.
- [ ] Веса модели (snapshots/exported model) лежат локально.
- [ ] Видео читается из локального пути.
- [ ] Плагин сохраняет `.h5/.csv` в собственный `output_dir`.
- [ ] Добавлена проверка порога уверенности (`pcutoff`).
- [ ] Добавлен fallback на CPU.

---

## 9) Что в этом репозитории сейчас

На текущий момент в репозитории присутствует только конфигурация проекта (`config.yaml`) без кода плагина и без файлов весов/снимков обучения. Поэтому интеграция в плагин выполняется как обёртка вокруг DeepLabCut с указанием локальных путей к модели и видео.

