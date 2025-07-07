import os
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
import numpy as np

# Установка путей
DATA_DIR = 'data'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Подготовка датасета
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(os.path.join(DATA_DIR, 'train'), transform=transform)
test_dataset = CustomImageDataset(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Загрузка предобученной модели
model = models.resnet18(weights='IMAGENET1K_V1')
num_classes = len(train_dataset.get_class_names())
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Перенос на GPU, если доступен
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Оптимизатор и функция потерь
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Обучение
num_epochs = 5
train_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # Проверка качества на тесте
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    test_acc = 100 * correct / total
    test_accs.append(test_acc)
    
    print(f'Эпоха {epoch+1}/{num_epochs}: Потери = {epoch_loss:.4f}, Точность (train) = {epoch_acc:.2f}%, Точность (test) = {test_acc:.2f}%')

# Визуализация
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Потери (train)')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Потери во время обучения')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accs, label='Точность (train)')
plt.plot(range(1, num_epochs + 1), test_accs, label='Точность (test)')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.title('Точность во время обучения')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'task6_training_curves.png'))
plt.close()

print(f"Графики сохранены в папке {RESULTS_DIR}")