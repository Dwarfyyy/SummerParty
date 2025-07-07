import os
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from PIL import Image
import numpy as np

DATA_DIR = 'data/train'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None)

# Подсчет количества изображений в каждом классе
class_counts = {}
for class_name in dataset.classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    class_counts[class_name] = len(os.listdir(class_dir))

# Сбор размеров изображений
sizes = []
for img_path in dataset.images:
    with Image.open(img_path) as img:
        sizes.append(img.size)

# Расчет статистики размеров
widths, heights = zip(*sizes)
min_width, max_width = min(widths), max(widths)
min_height, max_height = min(heights), max(heights)
avg_width = np.mean(widths)
avg_height = np.mean(heights)

# Вывод результатов
print("Количество изображений в каждом классе:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} изображений")
print(f"Минимальный размер: {min_width}x{min_height}")
print(f"Максимальный размер: {max_width}x{max_height}")
print(f"Средний размер: {avg_width:.2f}x{avg_height:.2f}")

# Визуализация гистограммы по классам
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Классы')
plt.ylabel('Количество изображений')
plt.title('Распределение изображений по классам')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'))
plt.close()

# Визуализация распределения размеров
plt.figure(figsize=(10, 6))
plt.hist(widths, bins=20, alpha=0.5, label='Ширина')
plt.hist(heights, bins=20, alpha=0.5, label='Высота')
plt.xlabel('Размер (пиксели)')
plt.ylabel('Частота')
plt.title('Распределение размеров изображений')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'size_distribution.png'))
plt.close()

print(f"Графики сохранены в папке {RESULTS_DIR}")