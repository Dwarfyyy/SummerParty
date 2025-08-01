# Домашнее задание к уроку 5: Аугментации и работа с изображениями

## Описание проекта
Данный проект реализует шесть заданий по работе с изображениями героев из датасета, расположенного в папке `data/`.  Все результаты сохраняются в папку `results/`.

## Задание 1: Стандартные аугментации torchvision (15 баллов)
Создан пайплайн стандартных аугментаций в `standart_augmentations.py` с использованием `torchvision.transforms`, включая `RandomHorizontalFlip`, `RandomResizedCrop`, `ColorJitter`, `RandomRotation` и `RandomGrayscale`. Аугментации применены к 5 изображениям из разных классов папки `train`. Результаты визуализированы: оригинал, каждая аугментация отдельно и комбинированный результат.

### Результаты
- Оригинальные изображения и результаты аугментаций сохранены в файл `results/standart_augmentation`.
- Пример результата:  
  ![Результаты задания 1](results/task1.png)

## Задание 2: Кастомные аугментации (20 баллов)
Реализованы четыре кастомные аугментации в `custom_augmentations.py`: реверсивные цвета (`ReverseColors`), гауссово размытие (`GaussianBlur`), увеличение контрастности (`IncreaseContrast`) и понижение яркости (`DecreaseBrightness`). Эти аугментации применены к изображениям из `train` и визуально сравнены с аугментациями из файла `extra_augs.py` (гауссов шум, случайное затирание, соляризация, автоконтраст).

### Результаты
- Сравнение кастомных аугментаций с аугментациями из `extra_augs.py` сохранено в файл `results/custom_augmentation`.
- Пример результата:  
  ![Результаты задания 2](results/task2.png)

## Задание 3: Анализ датасета (10 баллов)
Подсчитано количество изображений в `analyze_dataset.py`в каждом классе, найдены минимальный, максимальный и средний размеры изображений, а также визуализировано распределение размеров и гистограмма по классам.

### Результаты
- Количество изображений в каждом классе:
  Гароу: 30 изображений
  Генос: 30 изображений
  Сайтама: 30 изображений
  Соник: 30 изображений
  Татсумаки: 30 изображений
  Фубуки: 30 изображений
Минимальный размер: 210x240
Максимальный размер: 736x1308
Средний размер: 538.89x623.56
- Гистограмма распределения изображений по классам:  
  ![Гистограмма по классам](results/task3lol.png)
- Распределение размеров изображений:  
  ![Распределение размеров](results/task3.png)

## Задание 4: Pipeline аугментаций (20 баллов)
Реализован класс `AugmentationPipeline` с методами `add_augmentation`, `remove_augmentation`, `apply` и `get_augmentations`в `augmentation_pipeline.py`. Создано три уникальные конфигурации аугментаций: `light`, `medium` и `heavy`. Каждая конфигурация применена к изображениям из `train`, а также добавлено сравнение с кастомными аугментациями для визуального анализа.

### Реализация и изменения
- **Класс `AugmentationPipeline`**: Обеспечивает добавление, удаление и применение аугментаций, с корректной обработкой типов данных (`PIL.Image` и `torch.Tensor`).
- **Конфигурации**:
  - `light`: Случайный поворот (15 градусов) и гауссово размытие (радиус 1).
  - `medium`: Случайное затирание (вероятность 0.5, масштаб 0.05-0.15), увеличение контрастности (фактор 1.2) и оттенки серого (вероятность 0.3).
  - `heavy`: Соляризация (порог 100), автоконтраст (вероятность 0.7), понижение яркости (фактор 0.6) и реверсивные цвета.
- **Сравнение с кастомными аугментациями**: Добавлено визуальное сравнение с кастомными аугментациями (реверсивные цвета, гауссово размытие, увеличение контрастности, понижение яркости) для каждой конфигурации.

### Результаты
- Изображения с примененными конфигурациями и сравнением кастомных аугментаций сохранены в файлы `results/pipeline`.
- Пример результата:  
  ![Результаты задания 4](results/task4.png)

## Задание 5: Эксперимент с размерами (10 баллов)
Проведен эксперимент с разными размерами изображений (64x64, 128x128, 224x224, 512x512, 1024x1024) в `experiment_sizes.py`. Для каждого размера измерено время загрузки и применения аугментаций к 100 изображениям, а также потребление памяти. Построены графики зависимости времени и памяти от размера.

### Результаты
- Вывод эксперимента:
  - Обработка размера 64x64... Время: 0.16 сек, Память: 0.71 МБ
  - Обработка размера 128x128... Время: 0.17 сек, Память: 0.02 МБ
  - Обработка размера 224x224... Время: 0.18 сек, Память: 0.01 МБ
  - Обработка размера 512x512... Время: 0.27 сек, Память: 0.01 МБ
  - Обработка размера 1024x1024... Время: 0.64 сек, Память: 0.00 МБ
- График зависимости времени и памяти от размера:  
  ![Результаты задания 5](results/task5.png)

## Задание 6: Дообучение предобученных моделей (25 баллов)
Использована предобученная модель `resnet18` из `torchvision`в `train_model.py`. Последний слой заменен на количество классов датасета. Модель дообучена на `train`, проверено качество на `test`, а процесс обучения визуализирован (потери и точность).

### Результаты
- Вывод обучения:
  - Эпоха 1/5: Потери = 1.6711, Точность (train) = 28.89%, Точность (test) = 69.33%
  - Эпоха 2/5: Потери = 0.4882, Точность (train) = 97.78%, Точность (test) = 80.00%
  - Эпоха 3/5: Потери = 0.1677, Точность (train) = 100.00%, Точность (test) = 82.83%
  - Эпоха 4/5: Потери = 0.0665, Точность (train) = 100.00%, Точность (test) = 85.67%
  - Эпоха 5/5: Потери = 0.0383, Точность (train) = 100.00%, Точность (test) = 86.67%
  - Графики сохранены в папке results
- Графики потерь и точности:  
  ![Результаты задания 6](results/task6.png)

## Общий вывод
Проект успешно реализовал комплексный анализ и обработку датасета изображений героев, включая аугментации, анализ размеров, эксперименты с производительностью и дообучение нейронной сети. Задание 1 и 2 продемонстрировали эффективность стандартных и кастомных аугментаций, улучшая разнообразие данных. Задание 3 выявило равномерное распределение классов и широкий диапазон размеров изображений. Задание 4 показало гибкость пайплайна аугментаций с уникальными конфигурациями, а задание 5 подтвердило, что увеличение размера изображения приводит к росту времени обработки (с 0.16 сек до 0.64 сек) при минимальном изменении потребления памяти (от 0.71 МБ до 0.00 МБ). Задание 6 продемонстрировало успешное дообучение модели `resnet18`, достигнув 100% точности на обучающей выборке и 86.67% на тестовой после 5 эпох, с устойчивым снижением потерь с 1.6711 до 0.0383. Графики (task5.png и task6.png) наглядно отобразили зависимости и прогресс обучения, что подтверждает эффективность предложенных подходов и их потенциал для дальнейшего улучшения.