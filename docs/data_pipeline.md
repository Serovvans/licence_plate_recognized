# Формат данных для обучения yolo 

## Структура папок

data/processed/detector/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    └── val/
    └── test/

## Структура разметки

Для каждого изображения images/train/img001.jpg должен существовать файл labels/train/img001.txt с тем же именем.

class_id  cx  cy  w  h
Все значения нормализованы от 0 до 1 относительно размера изображения. Например, для изображения 640×480 с боксом номера:
0 0.512 0.334 0.241 0.089

Если на изображении два номера — две строки. Если объектов нет — файл пустой (но должен существовать).

## dataset.yaml

```yaml
# config/dataset.yaml
path: data/processed/detector  # путь к корню датасета
train: images/train
val: images/val
test: images/test

nc: 1          # количество классов
names: ['plate']
```