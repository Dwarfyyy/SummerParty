import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import os
from utils.experiment_utils import train_model, evaluate_model
from utils.visualization_utils import plot_learning_curves, plot_weight_distribution
from utils.model_utils import FullyConnectedNet

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Параметры эксперимента
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DATASET_PATH = 'C:/Users/79022/Desktop/SummerParty/homework3/data'  # Абсолютный путь
RESULTS_PATH = './results/regularization_experiments'
PLOTS_PATH = './plots'
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# Проверка существования пути к данным
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH, exist_ok=True)
    raise RuntimeError(f"Папка {DATASET_PATH} не существует. Создайте её и поместите файлы датасета вручную.")

# Загрузка данных (MNIST без скачивания)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(DATASET_PATH, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(DATASET_PATH, train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 3.1 Сравнение техник регуляризации
base_layers = [784, 256, 128, 10]  # Сокращённая архитектура с тремя скрытыми слоями
reg_configs = [
    {"name": "no_reg", "dropout": 0.0, "batch_norm": False, "weight_decay": 0.0},
    {"name": "dropout_0.1", "dropout": 0.1, "batch_norm": False, "weight_decay": 0.0},
    {"name": "dropout_0.3", "dropout": 0.3, "batch_norm": False, "weight_decay": 0.0},
    {"name": "dropout_0.5", "dropout": 0.5, "batch_norm": False, "weight_decay": 0.0},
    {"name": "batch_norm", "dropout": 0.0, "batch_norm": True, "weight_decay": 0.0},
    {"name": "dropout_batch_norm", "dropout": 0.3, "batch_norm": True, "weight_decay": 0.0},
    {"name": "l2_reg", "dropout": 0.0, "batch_norm": False, "weight_decay": 0.0001}
]

def run_regularization_comparison():
    """Сравнение различных техник регуляризации."""
    results = []
    
    for config in reg_configs:
        logger.info(f"Запуск эксперимента с регуляризацией: {config['name']}")
        model = FullyConnectedNet(base_layers, dropout=config['dropout'], batch_norm=config['batch_norm']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
        )
        training_time = time.time() - start_time
        
        # Оценка модели
        _, train_acc = evaluate_model(model, train_loader, DEVICE)
        _, test_acc = evaluate_model(model, test_loader, DEVICE)
        
        results.append({
            "name": config['name'],
            "train_acc": train_acc,
            "test_acc": test_acc,
            "training_time": training_time,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs
        })
        
        # Визуализация
        plot_learning_curves(
            train_losses, test_losses, train_accs, test_accs,
            title=f"Кривые обучения для {config['name']}",
            save_path=f"{PLOTS_PATH}/learning_curve_reg_{config['name']}.png"
        )
        plot_weight_distribution(
            model, title=f"Распределение весов для {config['name']}",
            save_path=f"{PLOTS_PATH}/weights_reg_{config['name']}.png"
        )
        
        # Анализ стабильности (стандартное отклонение потерь)
        test_loss_std = torch.tensor(test_losses).std().item()
        logger.info(f"Стабильность обучения ({config['name']}): Стандартное отклонение потерь: {test_loss_std:.4f}")
        
        logger.info(f"Регуляризация {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Сохранение результатов
    with open(f"{RESULTS_PATH}/reg_comparison_results.txt", "w") as f:
        for res in results:
            f.write(f"Регуляризация: {res['name']}\n")
            f.write(f"Точность на тренировке: {res['train_acc']:.4f}\n")
            f.write(f"Точность на тесте: {res['test_acc']:.4f}\n")
            f.write(f"Время обучения: {res['training_time']:.2f} сек\n\n")
    
    return results

# 3.2 Адаптивная регуляризация
def adaptive_regularization():
    """Реализация адаптивных техник регуляризации."""
    results = []
    
    # Адаптивный Dropout с изменяющимся коэффициентом
    for epoch_drop in [0.1, 0.3, 0.5]:
        config = {"name": f"adaptive_dropout_{epoch_drop}", "batch_norm": False, "weight_decay": 0.0}
        model = FullyConnectedNet(base_layers, dropout=epoch_drop, batch_norm=False).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
        )
        training_time = time.time() - start_time
        
        _, train_acc = evaluate_model(model, train_loader, DEVICE)
        _, test_acc = evaluate_model(model, test_loader, DEVICE)
        
        results.append({
            "name": config['name'],
            "train_acc": train_acc,
            "test_acc": test_acc,
            "training_time": training_time,
            "test_accs": test_accs
        })
        
        logger.info(f"Адаптивный Dropout {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # BatchNorm с различными momentum
    for momentum in [0.1, 0.5, 0.9]:
        config = {"name": f"batch_norm_momentum_{momentum}", "dropout": 0.0, "weight_decay": 0.0}
        model = FullyConnectedNet(base_layers, dropout=0.0, batch_norm=True, momentum=momentum).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
        )
        training_time = time.time() - start_time
        
        _, train_acc = evaluate_model(model, train_loader, DEVICE)
        _, test_acc = evaluate_model(model, test_loader, DEVICE)
        
        results.append({
            "name": config['name'],
            "train_acc": train_acc,
            "test_acc": test_acc,
            "training_time": training_time,
            "test_accs": test_accs
        })
        
        logger.info(f"BatchNorm momentum {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Комбинирование техник (пример: адаптивный Dropout + BatchNorm с разным momentum по слоям)
    class AdaptiveCombinedNet(nn.Module):
        def __init__(self, layers, initial_dropout=0.1, momentum_values=[0.1, 0.5, 0.9]):
            super(AdaptiveCombinedNet, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(len(layers) - 1):
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                if i < len(layers) - 2:  # Не добавляем нормализацию перед выходным слоем
                    self.layers.append(nn.BatchNorm1d(layers[i + 1], momentum=momentum_values[i % len(momentum_values)]))
                if i < len(layers) - 2:  # Применяем Dropout только к скрытым слоям
                    self.layers.append(nn.Dropout(initial_dropout * (i + 1) / (len(layers) - 2)))
                self.layers.append(nn.ReLU())
            self.layers = self.layers[:-1]  # Удаляем последний ReLU
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = AdaptiveCombinedNet(base_layers).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    train_losses, test_losses, train_accs, test_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
    )
    training_time = time.time() - start_time
    
    _, train_acc = evaluate_model(model, train_loader, DEVICE)
    _, test_acc = evaluate_model(model, test_loader, DEVICE)
    
    results.append({
        "name": "combined_adaptive",
        "train_acc": train_acc,
        "test_acc": test_acc,
        "training_time": training_time,
        "test_accs": test_accs
    })
    
    logger.info(f"Комбинированная модель combined_adaptive: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Сохранение результатов
    with open(f"{RESULTS_PATH}/reg_adaptive_results.txt", "w") as f:
        for res in results:
            f.write(f"Модель: {res['name']}\n")
            f.write(f"Точность на тренировке: {res['train_acc']:.4f}\n")
            f.write(f"Точность на тесте: {res['test_acc']:.4f}\n")
            f.write(f"Время обучения: {res['training_time']:.2f} сек\n\n")

if __name__ == "__main__":
    # Выполнение части 3.1
    reg_results = run_regularization_comparison()
    # Выполнение части 3.2
    adaptive_regularization()