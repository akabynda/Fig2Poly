# CurveForge

Генератор синтетического датасета для semantic- и instance-сегментации кривых на графиках.
Он рисует случайные полиномы и реалистичный графический «мусор», а затем применяет одни и те
же геометрические преобразования к изображению и маскам.

## Быстрый старт

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
curveforge --count 10000 --output dataset --workers 8
```

Пробный запуск без установки пакета:

```powershell
pip install -r requirements.txt
python -m curveforge --count 20 --output dataset_preview
```

Воспроизводимый JSON со всеми настройками:

```powershell
python -m curveforge --write-default-config config.json
python -m curveforge --config config.json --count 100000 --output dataset --workers 8 --resume
```

`--resume` безопасно продолжает совместимый прерванный запуск. Готовые образцы
проверяются и переиспользуются, а каждый новый файл записывается атомарно. Если
конфигурация, seed, число образцов или split отличаются, генератор завершится с
ошибкой вместо смешивания разных версий датасета.

## Формат

```text
dataset/
  dataset.json
  generation_state.json
  train.jsonl, val.jsonl, test.jsonl
  images/{train,val,test}/00000000.jpg
  semantic_masks/{train,val,test}/00000000.png
  instance_masks/{train,val,test}/00000000.png
  curve_masks/{train,val,test}/00000000/curve_001.png
  curve_masks/{train,val,test}/00000000/curve_002.png
  metadata/{train,val,test}/00000000.json
```

- `semantic_masks`: 8-bit PNG, фон `0`, любой пиксель кривой `255`.
- `curve_masks`: главная instance-разметка — отдельный 8-bit PNG для каждой видимой кривой.
  Маски независимы, поэтому пиксель пересечения может принадлежать сразу нескольким кривым.
- `instance_masks`: вспомогательный 16-bit PNG, фон `0`, значения `1..N`. В точках
  пересечения он хранит только один ID, поэтому для обучения instance segmentation следует
  использовать `curve_masks`.
- `metadata`: коэффициенты полинома в степенном базисе (от свободного члена), цвет,
  толщина, стиль, bbox, площадь, подпись, путь к индивидуальной маске, параметры графика
  и искажений. `curve_count` равен числу непустых масок после crop.
- Маска содержит только реально видимые пиксели кривой. Участки под непрозрачной легендой,
  текстом, аннотациями и случайными перекрытиями из каждой индивидуальной маски вычитаются.
  В пересечении двух кривых пиксель может оставаться сразу в двух независимых масках.
  Коэффициенты исходного полного полинома сохраняются в `metadata`, поэтому полную кривую
  можно восстанавливать позднее по видимым точкам.

`--workers N` включает параллельную генерацию. Результат детерминирован seed-ом и не зависит
от числа worker-процессов.

## Вариативность

- 0–10 кривых: полиномы, смеси синусоид, гауссовы и лоренцевы пики, сплайны,
  ступенчатые и затухающие сигналы;
- одинаковые цвета, разные цвета и цветовые группы, включая grayscale и colorblind palette;
- семейства близких, параллельных и почти совпадающих кривых с общим `relation_group`;
- solid/dashed/dotted/dash-dot, 1–4 px, круги/квадраты/кресты;
- белый, тёплый, холодный, тёмный и бумажный фон;
- график на всём кадре или уменьшенный график внутри журнальной страницы с текстовыми колонками;
- 2–6 независимых subplot на одном изображении: 1×N, 2×2, 3+2, 2+3, 3×2
  и вертикальные компоновки; каждая кривая сохраняет собственную маску и `panel_id`;
- box/open/cross/minimal/arrow/no axes;
- major/minor/both/horizontal/vertical/no grid;
- случайные пределы, деления, подписи, заголовки, легенды, аннотации;
- пустые графики и hard negatives: reference lines, scatter, error bars, стрелки и водяные знаки;
- перекрытия, поворот, перспектива, смещённый агрессивный crop;
- blur, шум, contrast/brightness, grayscale, понижение разрешения, scanlines,
  sharpening и JPEG-артефакты.

## Практическая стратегия обучения

Начните с semantic segmentation (кривая/фон), затем добавьте instance head или постобработку.
Для переноса на реальные картинки смешивайте синтетику с небольшим вручную размеченным real-набором.
Тестовый real-набор должен быть полностью отделён: синтетический test измеряет корректность пайплайна,
но не реальную обобщающую способность.

## Mask2Former baseline

Для обучения instance segmentation на независимых PNG-масках:

```powershell
pip install -r requirements-train.txt
python -m training.mask2former train --dataset dataset --output runs/mask2former
python -m training.mask2former evaluate --model runs/mask2former/final --dataset dataset --split test
```

Pipeline напрямую передаёт в Mask2Former тензор `(N, H, W)`, поэтому пересекающиеся и несвязные
маски кривых не теряются при конвертации в polygon или единую ID-карту.

## YOLO26 baseline

```powershell
python -m training.convert_yolo --dataset dataset --workers 10
python -m training.train_yolo --data dataset/curve_yolo.yaml --epochs 30

# Продолжить прерванное обучение с последней завершённой эпохи:
python -m training.train_yolo --resume runs/yolo26/full_visible_384/weights/last.pt
```

Конвертер объединяет пунктирные компоненты каждой кривой в одну polygon-строку YOLO. Для тонких
кривых обязательно используются `mask_ratio=1` и `overlap_mask=False`.
