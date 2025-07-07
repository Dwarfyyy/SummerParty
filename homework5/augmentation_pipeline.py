import os
import torch
import random
from torchvision import transforms
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
from PIL import Image
from extra_augs import AddGaussianNoise, RandomErasingCustom, Solarize, AutoContrast
from custom_augmentations import ReverseColors, GaussianBlur, IncreaseContrast, DecreaseBrightness

# Класс для пайплайна аугментаций
class AugmentationPipeline:
    """Класс для управления пайплайном аугментаций"""
    
    def __init__(self):
        self.augmentations = {}
    
    def add_augmentation(self, name, aug):
        """Добавляет аугментацию с указанным именем"""
        self.augmentations[name] = aug
    
    def remove_augmentation(self, name):
        """Удаляет аугментацию по имени"""
        if name in self.augmentations:
            del self.augmentations[name]
    
    def apply(self, image):
        """Применяет все аугментации к изображению"""
        img = image
        for name, aug in self.augmentations.items():
            if isinstance(img, Image.Image):
                if name in ['Соляризация', 'Автоконтраст', 'Гауссов шум', 'Случайное затирание']:
                    img_tensor = transforms.ToTensor()(img)
                    img = aug(img_tensor)
                else:
                    img = aug(img)
            elif isinstance(img, torch.Tensor):
                # Преобразование обратно в PIL.Image для кастомных аугментаций
                if name in ['Гауссово размытие', 'Увеличение контрастности', 'Понижение яркости', 'Реверсивные цвета', 'Случайный поворот']:
                    img = transforms.ToPILImage()(img)
                    img = aug(img)
                else:
                    img = aug(img)
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)  # Возвращаем обратно в PIL.Image для визуализации
        return img
    
    def get_augmentations(self):
        """Возвращает список имен аугментаций"""
        return list(self.augmentations.keys())

# Установка путей
DATA_DIR = 'data/train'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Инициализация датасета
dataset = CustomImageDataset(root_dir=DATA_DIR, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Выбор по одному изображению из каждого класса
selected_images = []
selected_labels = []
for class_name in class_names[:5]:  # Берем первые 5 классов
    class_idx = dataset.class_to_idx[class_name]
    class_images = [i for i, lbl in enumerate(dataset.labels) if lbl == class_idx]
    random_idx = random.choice(class_images)
    img, _ = dataset[random_idx]
    selected_images.append(img)
    selected_labels.append(class_name)

# Создание уникальных конфигураций
light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation('Случайный поворот', transforms.RandomRotation(degrees=15))
light_pipeline.add_augmentation('Гауссово размытие', GaussianBlur(radius=1))

medium_pipeline = AugmentationPipeline()
medium_pipeline.add_augmentation('Случайное затирание', RandomErasingCustom(p=0.5, scale=(0.05, 0.15)))
medium_pipeline.add_augmentation('Увеличение контрастности', IncreaseContrast(factor=1.2))
medium_pipeline.add_augmentation('Оттенки серого', transforms.RandomGrayscale(p=0.3))

heavy_pipeline = AugmentationPipeline()
heavy_pipeline.add_augmentation('Соляризация', Solarize(threshold=100))
heavy_pipeline.add_augmentation('Автоконтраст', AutoContrast(p=0.7))
heavy_pipeline.add_augmentation('Понижение яркости', DecreaseBrightness(factor=0.6))
heavy_pipeline.add_augmentation('Реверсивные цвета', ReverseColors())

# Функция для визуализации
def visualize_pipeline(image, pipeline, class_name, config_name, img_idx):
    print(f"\nОбработка класса: {class_name} с конфигурацией: {config_name}")
    print("Примененные аугментации:", ", ".join(pipeline.get_augmentations()))
    
    img_tensor = transforms.ToTensor()(image)
    aug_image = pipeline.apply(image)
    aug_image_tensor = transforms.ToTensor()(aug_image)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axes[0].set_title('Оригинал')
    axes[0].axis('off')
    axes[1].imshow(aug_image_tensor.permute(1, 2, 0).numpy())
    axes[1].set_title(f'{config_name}')
    axes[1].axis('off')
    plt.suptitle(f'Класс: {class_name}')
    plt.tight_layout()
    
    # Сохранение с информативным именем файла
    output_filename = f'pipeline_{config_name}_class_{class_name}_img_{img_idx}.png'
    output_path = os.path.join(RESULTS_DIR, output_filename)
    plt.savefig(output_path)
    print(f"Результат сохранен: {output_filename}")
    plt.close()

# Применение конфигураций и визуализация
pipelines = {'light': light_pipeline, 'medium': medium_pipeline, 'heavy': heavy_pipeline}
for config_name, pipeline in pipelines.items():
    for idx, (image, class_name) in enumerate(zip(selected_images, selected_labels)):
        visualize_pipeline(image, pipeline, class_name, config_name, idx)

print(f"\nВсе результаты сохранены в папке {RESULTS_DIR}")