import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import get_dataloader, CustomImageDataset
from model import Resnet18
from tqdm import tqdm
import os
import time

def train_model(image_size, epochs=5, batch_size=16, weights=True):
    """
    Обучает модель ResNet-18 для заданного размера изображения с оптимизациями для ускорения.
    
    Args:
        image_size (int): Размер изображения.
        epochs (int): Количество эпох.
        batch_size (int): Размер батча.
        weights (bool): Использовать ли предобученные веса.
    """
    # Активация CUDNN для оптимизации
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CustomImageDataset(root_dir='./data/train').get_class_names())
    model = Resnet18(num_classes=num_classes).to(device)
    if weights:
        pretrained_model = Resnet18(num_classes=1000).to(device)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    train_loader = get_dataloader(
        root_dir='./data/train',
        batch_size=batch_size,
        target_size=(image_size, image_size),
        train=True
    )

    # Логирование в файл
    os.makedirs('./results', exist_ok=True)
    log_file = f'./results/training_log_{image_size}.txt'
    with open(log_file, 'w') as f:
        f.write(f"Обучение модели для размера изображения {image_size}x{image_size}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        log_message = (f"Эпоха {epoch+1}, Потеря: {avg_loss:.4f}, "
                       f"Текущая скорость обучения: {scheduler.get_last_lr()[0]:.6f}\n")
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message)

    # Сохранение модели
    os.makedirs('./weights', exist_ok=True)
    torch.save(model.state_dict(), f'./weights/best_resnet18_{image_size}.pth')

if __name__ == "__main__":
    # Проверка доступной памяти GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} МБ")
    
    image_sizes = [224, 256, 384, 512]
    for size in image_sizes:
        print(f"Обучение модели для размера изображения {size}x{size}")
        start_time = time.time()
        train_model(size, epochs=5, batch_size=16, weights=True)
        elapsed_time = time.time() - start_time
        with open(f'./results/training_time_{size}.txt', 'w') as f:
            f.write(f"Время обучения для {size}x{size}: {elapsed_time:.2f} секунд")