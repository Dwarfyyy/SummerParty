import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class CarLogoCNN(nn.Module):
    """Класс для модели классификации логотипов автомобилей с использованием Transfer Learning."""
    
    def __init__(self, num_classes=8, pretrained=True):
        """Инициализация модели.
        
        Args:
            num_classes (int): Количество классов (8 марок автомобилей).
            pretrained (bool): Использовать предобученную модель.
        """
        super(CarLogoCNN, self).__init__()
        # Загружаем предобученную модель EfficientNet-B0 с современным параметром weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        # Заменяем финальный слой для нашего числа классов
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        """Прямой проход через модель."""
        return self.model(x)

if __name__ == "__main__":
    model = CarLogoCNN()
    print(model)