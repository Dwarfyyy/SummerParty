# README.md

## Описание домашней работы

Данная домашняя работа представляет собой задачу по улучшению модели машинного обучения, обучаемой на текстах книг серии "Гарри Поттер" на русском языке. Цель — разработка и оптимизация модели на основе трансформеров для токенизации текста, обучения и генерации текста. Работа включает создание модульной структуры кода и проведение экспериментов с разными наборами данных.

## Файлы и структура домашней работы

- `tokenizer.json` - конфигурация токенизатора с поддержкой Byte-Level BPE.
- `data.py` - модуль для обработки данных.
- `model.py` - определение модели трансформера.
- `train.py` - скрипт для обучения модели.
- `generate.py` - скрипт для генерации текста.
- `results` - папка с результатами
- `text.txt` - Гарри Поттер и орден феникса
- `text1.txt` - Гарри Поттер и философский камень

## Эксперименты

### Эксперимент 1: Гарри Поттер и Философский Камень
#### Обучение модели
Запустите `train.py`:
```
PS C:\Users\79022\Desktop\SummerParty\homework6> & C:/Users/79022/Desktop/SummerParty/homework6/env/Scripts/python.exe c:/Users/79022/Desktop/SummerParty/homework6/train.py
[00:00:00] Предобработка файлов (0 Мо)    ██████████████████████████████████████████████████████████████████                100%
[00:00:00] Токенизация слов               ██████████████████████████████████████████████████████████████████ 15166    /    15166
[00:00:00] Подсчет пар                    ██████████████████████████████████████████████████████████████████ 15166    /    15166
[00:00:00] Вычисление слияний             ██████████████████████████████████████████████████████████████████ 24444    /    24444
💡 Совет: Установите litmodels для работы с облачными загрузками.
GPU доступен: True (cuda), используется: True
...
Epoch 2: 100%|████████████████████████████████████████████████████| 472/472 [00:45<00:00, 10.34it/s, v_num=4, train_loss=6.320]
`Trainer.fit` остановлен: достигнуто `max_epochs=3`.
```

#### Генерация текста
Запустите `generate.py`:
```
PS C:\Users\79022\Desktop\SummerParty\homework6> & C:/Users/79022/Desktop/SummerParty/homework6/env/Scripts/python.exe c:/Users/79022/Desktop/SummerParty/homework6/generate.py
Введите 'quit' для выхода.
Вы: Гарри Поттер?
Бот (обычная генерация): Гарри Поттер? ,? диковатого,, Нимбус для Снэйп лакричные:, часто футбольный
 Гарри И что, над:: в были под палочку попал чудовищного Петуния. всеуслышание
Бот (beam search): Гарри Поттер?,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Вы:
```

### Эксперимент 2: Гарри Поттер и Орден Феникса
#### Обучение модели
Запустите `train.py`:
```
PS C:\Users\79022\Desktop\SummerParty\homework6> & C:/Users/79022/Desktop/SummerParty/homework6/env/Scripts/python.exe c:/Users/79022/Desktop/SummerParty/homework6/train.py
[00:00:00] Pre-processing files (2 Mo)    ██████████████████████████████████████████████████████████████████                100%
[00:00:00] Tokenize words                 ██████████████████████████████████████████████████████████████████ 34097    /    34097
[00:00:00] Count pairs                    ██████████████████████████████████████████████████████████████████ 34097    /    34097
[00:00:00] Compute merges                 ██████████████████████████████████████████████████████████████████ 29901    /    29901
💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
C:\Users\79022\Desktop\SummerParty\homework6\env\Lib\site-packages\pytorch_lightning\trainer\connectors\logger_connector\logger_connector.py:76: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name                | Type               | Params | Mode
-------------------------------------------------------------------
0 | embedding           | Embedding          | 15.4 M | train
1 | pos_encoder         | PositionalEncoding | 0      | train
2 | transformer_decoder | TransformerDecoder | 25.2 M | train
3 | fc_out              | Linear             | 15.4 M | train
4 | dropout             | Dropout            | 0      | train
-------------------------------------------------------------------
56.0 M    Trainable params
0         Non-trainable params
56.0 M    Total params
223.897   Total estimated model params size (MB)
90        Modules in train mode
0         Modules in eval mode
C:\Users\79022\Desktop\SummerParty\homework6\env\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:425: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.
Epoch 9: 100%|██████████████████████████████████████████████████| 1709/1709 [02:55<00:00,  9.75it/s, v_num=7, train_loss=7.030]
`Trainer.fit` stopped: `max_epochs=10` reached.
Epoch 9: 100%|██████████████████████████████████████████████████| 1709/1709 [02:59<00:00,  9.54it/s, v_num=7, train_loss=7.030]
```

#### Генерация текста
Запустите `generate.py`:
```
PS C:\Users\79022\Desktop\SummerParty\homework6> & C:/Users/79022/Desktop/SummerParty/homework6/env/Scripts/python.exe c:/Users/79022/Desktop/SummerParty/homework6/generate.py
Введите 'quit' для выхода.
Вы: Северус Снейп
Бот (обычная генерация): Северус Снейп,лен, и,,,,, на и
,, и. опять
 егоетной глаза и ,,, . Гарри возможность. ,
 заставить на он.то, на в! будет и
  Гарри когда,....-. его змея —? громкий вопль с! кого в свча. до. я на — людей подолуст что него свете из-. ужаса институт с. — неожиданно к ей.... иыш грозным коридора занятий с зал,...- —,,. — сильно. от? в в на Бедные
 мгновение это и стали вчить.й во почер,, В добрался
 низко конце...

 наоркой ПРОИСХОЖДЕНИЯ стола.., крохот и, местотывалоось они Уэсли. камина шумно бы и.:,!, едва появился
 — что что глад — резко сал Это снова, в всех деревянного Джордж которыеку два,ившиеся какое в:алась чтобы снова, вестибюля в ц привычкуце  накрыла плотной капитан,
Бот (beam search): Северус Снейп,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Вы:
```
## Вывод

После выполнения домашней работы установлено, что моя текущая модель демонстрирует сомнительную способность к генерации осмысленных текстов. Эксперимент 1 ("Гарри Поттер и Философский Камень") показал ограниченную способность модели к генерации осмысленных текстов с большим количеством лишних символов. Эксперимент 2 ("Гарри Поттер и Орден Феникса") с увеличенным объемом данных (2 МБ) и 10 эпохами обучения улучшил связность текста, но сохранил проблему с лишними знаками и вырождением beam search. Необходимо доработать токенизатор (увеличить словарь до 50,000 токенов), увеличить число слоев модели до 8 и добавить постобработку текста для удаления шумов. Обычная генерация дает результат с лишними символами ("Гарри Поттер? ,? диковатого,,..."), а beam search полностью вырождается в повторяющиеся символы. Это указывает на необходимость доработки токенизатора, увеличения числа эпох обучения и оптимизации гиперпараметров модели. Дальнейшая работа должна сосредоточиться на постобработке текста и тестировании с более длинными запросами.
