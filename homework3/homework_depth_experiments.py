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
EPOCHS = 20
LEARNING_RATE = 0.001
DATASET_PATH = 'C:/Users/79022/Desktop/SummerParty/homework3/data'  # Абсолютный путь
RESULTS_PATH = './results/depth_experiments'
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

# Конфигурации моделей
depth_configs = [
    {"name": "1_layer", "layers": [784, 10]},  # Линейный классификатор
    {"name": "2_layers", "layers": [784, 128, 10]},  # 1 скрытый слой
    {"name": "3_layers", "layers": [784, 128, 64, 10]},  # 2 скрытых слоя
    {"name": "5_layers", "layers": [784, 256, 128, 64, 32, 10]},  # 4 скрытых слоя
    {"name": "7_layers", "layers": [784, 512, 256, 128, 64, 32, 16, 10]}  # 6 скрытых слоев
]

# Конфигурации с регуляризацией
reg_configs = [
    {"name": "no_reg", "dropout": 0.0, "batch_norm": False},
    {"name": "dropout_0.3", "dropout": 0.3, "batch_norm": False},
    {"name": "batch_norm", "dropout": 0.0, "batch_norm": True},
    {"name": "dropout_batch_norm", "dropout": 0.3, "batch_norm": True}
]

def run_depth_experiments():
    """Проведение экспериментов с различной глубиной сети."""
    results = []
    
    for config in depth_configs:
        logger.info(f"Запуск эксперимента для модели: {config['name']}")
        model = FullyConnectedNet(config['layers'], dropout=0.0, batch_norm=False).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
        )
        training_time = time.time() - start_time
        
        # Оценка модели (извлекаем только точность из кортежа)
        _, train_acc = evaluate_model(model, train_loader, DEVICE)  # Игнорируем loss, берем accuracy
        _, test_acc = evaluate_model(model, test_loader, DEVICE)    # Игнорируем loss, берем accuracy
        
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
        
        # Визуализация кривых обучения
        plot_learning_curves(
            train_losses, test_losses, train_accs, test_accs,
            title=f"Кривые обучения для {config['name']}",
            save_path=f"{PLOTS_PATH}/learning_curve_{config['name']}.png"
        )
        
        # Визуализация распределения весов
        plot_weight_distribution(
            model, title=f"Распределение весов для {config['name']}",
            save_path=f"{PLOTS_PATH}/weights_{config['name']}.png"
        )
        
        logger.info(f"Модель {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Сохранение результатов
    with open(f"{RESULTS_PATH}/depth_results.txt", "w") as f:
        for res in results:
            f.write(f"Модель: {res['name']}\n")
            f.write(f"Точность на тренировке: {res['train_acc']:.4f}\n")
            f.write(f"Точность на тесте: {res['test_acc']:.4f}\n")
            f.write(f"Время обучения: {res['training_time']:.2f} сек\n\n")
    
    # Анализ переобучения
    analyze_overfitting(results)

def run_regularization_experiments():
    """Проведение экспериментов с регуляризацией."""
    base_layers = [784, 128, 64, 10]  # Базовая архитектура с 3 слоями
    results = []
    
    for config in reg_configs:
        logger.info(f"Запуск эксперимента с регуляризацией: {config['name']}")
        model = FullyConnectedNet(base_layers, dropout=config['dropout'], batch_norm=config['batch_norm']).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001 if config['name'] == "l2_reg" else 0.0)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        train_losses, test_losses, train_accs, test_accs = train_model(
            model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE
        )
        training_time = time.time() - start_time
        
        # Оценка модели (извлекаем только точность из кортежа)
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
            title=f"Кривые обучения с {config['name']}",
            save_path=f"{PLOTS_PATH}/learning_curve_reg_{config['name']}.png"
        )
        plot_weight_distribution(
            model, title=f"Распределение весов с {config['name']}",
            save_path=f"{PLOTS_PATH}/weights_reg_{config['name']}.png"
        )
        
        logger.info(f"Регуляризация {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Сохранение результатов
    with open(f"{RESULTS_PATH}/reg_results.txt", "w") as f:
        for res in results:
            f.write(f"Регуляризация: {res['name']}\n")
            f.write(f"Точность на тренировке: {res['train_acc']:.4f}\n")
            f.write(f"Точность на тесте: {res['test_acc']:.4f}\n")
            f.write(f"Время обучения: {res['training_time']:.2f} сек\n\n")

def analyze_overfitting(results):
    """Анализ переобучения на основе результатов экспериментов."""
    plt.figure(figsize=(10, 6))
    for res in results:
        gap = [train - test for train, test in zip(res['train_accs'], res['test_accs'])]
        plt.plot(gap, label=f"Разрыв {res['name']}")
    plt.title("Разрыв между тренировочной и тестовой точностью")
    plt.xlabel("Эпоха")
    plt.ylabel("Разрыв точности")
    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOTS_PATH}/overfitting_analysis.png")
    plt.close()
    
    # Определение оптимальной глубины
    best_model = max(results, key=lambda x: x['test_acc'])
    logger.info(f"Оптимальная глубина: {best_model['name']} с тестовой точностью {best_model['test_acc']:.4f}")

if __name__ == "__main__":
    run_depth_experiments()
    run_regularization_experiments()