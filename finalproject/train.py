import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.model import CarLogoCNN
from src.data.dataset import CarLogoDataset, train_transform, test_transform
from src.utils.utils import save_model, save_plot
import argparse

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Обучение модели.
    
    Args:
        model (torch.nn.Module): Модель для обучения.
        train_loader (DataLoader): Даталоader для тренировочных данных.
        criterion (nn.Module): Функция потерь.
        optimizer (torch.optim): Оптимизатор.
        num_epochs (int): Количество эпох.
    """
    train_losses = []
    device = next(model.parameters()).device  # Получаем устройство модели
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Переносим данные на устройство модели
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Эпоха [{epoch+1}/{num_epochs}], Потери: {epoch_loss:.4f}")
    return train_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/Train", help="Путь к тренировочным данным")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    args = parser.parse_args()

    # Инициализация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarLogoCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Датасет и DataLoader
    dataset = CarLogoDataset(args.data_dir, transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Обучение
    train_losses = train_model(model, train_loader, criterion, optimizer, args.epochs)

    # Сохранение результатов
    save_model(model, "results/models/efficientnet_b0.pth")
    save_plot(train_losses, "Потери на тренировке", "results/plots/train_loss.png")