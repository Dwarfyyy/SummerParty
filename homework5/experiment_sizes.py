import os
import time
import psutil
import torch
from torchvision import transforms
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Установка путей
DATA_DIR = 'data/train'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None)

# Выбор 100 случайных изображений
all_indices = list(range(len(dataset)))
random_indices = np.random.choice(all_indices, 100, replace=False)
selected_dataset = torch.utils.data.Subset(dataset, random_indices)

# Конфигурация аугментаций
aug_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=15)
])

# Функция для измерения времени и памяти
def measure_performance(size):
    # Изменение размера датасета
    transform_resize = transforms.Resize(size)
    augmented_dataset = [(transform_resize(img), label) for img, label in selected_dataset]
    
    # Измерение времени и памяти
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 ** 2  # МБ
    
    for img, _ in augmented_dataset:
        aug_img = aug_pipeline(img)
    
    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 ** 2  # МБ
    
    load_time = end_time - start_time
    mem_usage = mem_after - mem_before
    
    return load_time, mem_usage

# Эксперимент с разными размерами
sizes = [(64, 64), (128, 128), (224, 224), (512, 512), (1024, 1024)]
times = []
memories = []

for size in sizes:
    print(f"Обработка размера {size[0]}x{size[1]}...")
    load_time, mem_usage = measure_performance(size)
    times.append(load_time)
    memories.append(mem_usage)
    print(f"Время: {load_time:.2f} сек, Память: {mem_usage:.2f} МБ")

# Построение графиков
plt.figure(figsize=(10, 5))

# График времени
plt.subplot(1, 2, 1)
plt.plot([s[0] for s in sizes], times, marker='o', color='#1E90FF')
plt.xlabel('Размер (пиксели)')
plt.ylabel('Время (сек)')
plt.title('Зависимость времени от размера')
plt.grid(True)

# График памяти
plt.subplot(1, 2, 2)
plt.plot([s[0] for s in sizes], memories, marker='o', color='#FF4500')
plt.xlabel('Размер (пиксели)')
plt.ylabel('Память (МБ)')
plt.title('Зависимость памяти от размера')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'task5_performance.png'))
plt.close()

print(f"Графики сохранены в папке {RESULTS_DIR}")