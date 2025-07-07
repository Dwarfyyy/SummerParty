import os
import torch
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, RandomErasingCustom, Solarize, AutoContrast
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# Установка путей
DATA_DIR = 'data/train'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Кастомные аугментации
class ReverseColors:
    """Инвертирует цвета изображения."""
    def __call__(self, img):
        return ImageOps.invert(img)

class GaussianBlur:
    """Применяет гауссово размытие к изображению."""
    def __init__(self, radius=2):
        self.radius = radius
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class IncreaseContrast:
    """Увеличивает контрастность изображения."""
    def __init__(self, factor=1.5):
        self.factor = factor
    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.factor)

class DecreaseBrightness:
    """Понижает яркость изображения."""
    def __init__(self, factor=0.5):
        self.factor = factor
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)

# Аугментации
custom_augs = {
    'Реверсивные цвета': ReverseColors(),
    'Гауссово размытие': GaussianBlur(radius=2),
    'Увеличение контрастности': IncreaseContrast(factor=1.5),
    'Понижение яркости': DecreaseBrightness(factor=0.5)
}

extra_augs = {
    'Гауссов шум': AddGaussianNoise(mean=0., std=0.1),
    'Случайное затирание': RandomErasingCustom(p=1.0, scale=(0.02, 0.2)),
    'Соляризация': Solarize(threshold=128),
    'Автоконтраст': AutoContrast(p=1.0)
}

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Случайный выбор одного изображения из каждой папки класса
selected_images = []
selected_labels = []
classes_covered = set()

for class_name in class_names[:5]:  # Берем первые 5 классов
    class_idx = dataset.class_to_idx[class_name]
    class_images = [i for i, lbl in enumerate(dataset.labels) if lbl == class_idx]
    random_idx = random.choice(class_images)
    img, label = dataset[random_idx]
    selected_images.append(img)
    selected_labels.append(class_name)

# Функция для визуализации
def visualize_augmentations(image, class_name, img_idx):
    # Подготовка изображения
    image_tensor = transforms.ToTensor()(image)
    
    # Создание сетки для визуализации
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Оригинал')
    axes[0].axis('off')
    
    # Применение кастомных аугментаций
    for idx, (aug_name, aug) in enumerate(custom_augs.items(), 1):
        aug_image = aug(image)
        aug_image = transforms.ToTensor()(aug_image)
        axes[idx].imshow(aug_image.permute(1, 2, 0).numpy())
        axes[idx].set_title(aug_name)
        axes[idx].axis('off')
    
    # Применение аугментаций из extra_augs.py
    for idx, (aug_name, aug) in enumerate(extra_augs.items(), 5):
        aug_image = aug(image_tensor)
        axes[idx].imshow(aug_image.permute(1, 2, 0).numpy())
        axes[idx].set_title(aug_name)
        axes[idx].axis('off')
    
    plt.suptitle(f'Класс: {class_name}')
    plt.tight_layout()
    
    # Сохранение результата
    output_path = os.path.join(RESULTS_DIR, f'custom_augs_comparison_{img_idx}_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

# Применение аугментаций и визуализация
for idx, (image, class_name) in enumerate(zip(selected_images, selected_labels)):
    visualize_augmentations(image, class_name, idx)

print(f"Результаты сохранены в папке {RESULTS_DIR}")