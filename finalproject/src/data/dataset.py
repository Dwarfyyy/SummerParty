import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class CarLogoDataset(Dataset):
    """Класс для загрузки и аугментации датасета логотипов автомобилей."""
    
    def __init__(self, data_dir, transform=None):
        """Инициализация датасета.
        
        Args:
            data_dir (str): Путь к директории с данными (Train или Test).
            transform (callable, optional): Трансформации для аугментации.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
    
    def _load_images(self):
        """Загрузка путей к изображениям."""
        images = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.data_dir, cls_name)
            for img_name in os.listdir(class_dir):
                images.append((os.path.join(class_dir, img_name), self.class_to_idx[cls_name]))
        return images
    
    def __len__(self):
        """Возвращает общее количество изображений."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Возвращает изображение и метку."""
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label

# Трансформации для аугментации
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

if __name__ == "__main__":
    dataset = CarLogoDataset(data_dir="data/Train", transform=train_transform)
    print(f"Количество изображений: {len(dataset)}")