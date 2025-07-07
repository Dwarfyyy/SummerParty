import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
import numpy as np
from PIL import Image

# Установка путей
DATA_DIR = 'data/train'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Определение аугментаций
augmentations = {
    'Горизонтальный поворот': transforms.RandomHorizontalFlip(p=1.0),
    'Случайная обрезка': transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    'Цветовые изменения': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    'Случайный поворот': transforms.RandomRotation(degrees=30),
    'Оттенки серого': transforms.RandomGrayscale(p=1.0)
}

# Комбинированный пайплайн аугментаций
combined_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor()  # Для корректной обработки в PyTorch
])

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Выбор по одному изображению из первых 5 классов
selected_images = []
selected_labels = []
classes_covered = set()

for idx in range(len(dataset)):
    img, label = dataset[idx]
    class_name = class_names[label]
    if class_name not in classes_covered:
        selected_images.append(img)
        selected_labels.append(class_name)
        classes_covered.add(class_name)
    if len(selected_images) == 5:
        break

# Функция для визуализации
def visualize_augmentations(image, class_name, img_idx):
    # Подготовка изображения для обработки
    image_tensor = transforms.ToTensor()(image)
    
    # Создание сетки для визуализации
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Оригинальное изображение
    axes[0].imshow(image)
    axes[0].set_title('Оригинал')
    axes[0].axis('off')
    
    # Применение каждой аугментации отдельно
    for idx, (aug_name, aug) in enumerate(augmentations.items(), 1):
        aug_image = aug(image)
        aug_image = transforms.ToTensor()(aug_image)  # Конвертация в тензор для отображения
        axes[idx].imshow(aug_image.permute(1, 2, 0).numpy())
        axes[idx].set_title(aug_name)
        axes[idx].axis('off')
    
    # Применение комбинированных аугментаций
    combined_image = combined_transform(image)
    axes[5].imshow(combined_image.permute(1, 2, 0).numpy())
    axes[5].set_title('Комбинированные аугментации')
    axes[5].axis('off')
    
    plt.suptitle(f'Класс: {class_name}')
    plt.tight_layout()
    
    # Сохранение результата
    output_path = os.path.join(RESULTS_DIR, f'augmentation_{img_idx}_{class_name}.png')
    plt.savefig(output_path)
    plt.close()

# Применение аугментаций и визуализация для каждого изображения
for idx, (image, class_name) in enumerate(zip(selected_images, selected_labels)):
    visualize_augmentations(image, class_name, idx)

print(f"Результаты сохранены в папке {RESULTS_DIR}")