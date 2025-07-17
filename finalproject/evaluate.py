import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.model import CarLogoCNN
from src.data.dataset import CarLogoDataset, test_transform
from src.utils.utils import save_plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
import seaborn as sns
import argparse
from datetime import datetime

def evaluate_model(model, test_loader, criterion, device):
    """Оценка модели на тестовом датасете.
    
    Args:
        model (torch.nn.Module): Обученная модель.
        test_loader (DataLoader): Даталоader для тестовых данных.
        criterion (nn.Module): Функция потерь.
        device (torch.device): Устройство для вычислений (CPU/GPU).
    
    Returns:
        float: Точность модели.
        float: Потери на тесте.
        float: F1-Score.
        float: Weighted Precision.
        list: Предсказанные метки.
        list: Истинные метки.
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    test_loss /= len(test_loader)
    print(f"Потери на тесте: {test_loss:.4f}, Точность: {accuracy:.4f}, F1-Score: {f1:.4f}, Weighted Precision: {precision:.4f}")
    
    # Вывод precision для каждого класса
    classes = test_loader.dataset.classes
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    for cls, prec in zip(classes, precision_per_class):
        print(f"Precision для {cls}: {prec:.4f}")
    
    return accuracy, test_loss, f1, precision, all_preds, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/models/efficientnet_b0.pth", help="Путь к сохраненной модели")
    parser.add_argument("--data_dir", type=str, default="data/Test", help="Путь к тестовым данным")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    args = parser.parse_args()

    # Инициализация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarLogoCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    # Датасет и DataLoader
    dataset = CarLogoDataset(args.data_dir, transform=test_transform)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Оценка
    accuracy, test_loss, f1, precision, preds, labels = evaluate_model(model, test_loader, criterion, device)

    # Визуализация матрицы ошибок
    cm = confusion_matrix(labels, preds)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title("Матрица ошибок")
    plt.xlabel("Предсказанные классы")
    plt.ylabel("Истинные классы")
    plt.savefig(f"results/plots/confusion_matrix_{timestamp}.png")
    plt.close()

    # Сохранение метрик
    with open(f"results/predictions/evaluation_metrics_{timestamp}.txt", "w") as f:
        f.write(f"Дата и время: {timestamp} (CEST)\n")
        f.write(f"Потери на тесте: {test_loss:.4f}\n")
        f.write(f"Точность: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write("Precision по классам:\n")
        for cls, prec in zip(dataset.classes, precision_score(labels, preds, average=None)):
            f.write(f"{cls}: {prec:.4f}\n")