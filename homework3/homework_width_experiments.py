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
RESULTS_PATH = './results/width_experiments'
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

# 2.1 Сравнение моделей разной ширины
width_configs = [
    {"name": "narrow", "layers": [784, 64, 32, 16, 10]},    # Узкие слои
    {"name": "medium", "layers": [784, 256, 128, 64, 10]},  # Средние слои
    {"name": "wide", "layers": [784, 1024, 512, 256, 10]},  # Широкие слои
    {"name": "very_wide", "layers": [784, 2048, 1024, 512, 10]}  # Очень широкие слои
]

def count_parameters(model):
    """Подсчёт числа обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_width_comparison():
    """Сравнение моделей с различной шириной слоёв."""
    results = []
    
    for config in width_configs:
        logger.info(f"Запуск эксперимента для модели: {config['name']}")
        model = FullyConnectedNet(config['layers'], dropout=0.0, batch_norm=False).to(DEVICE)
        param_count = count_parameters(model)
        logger.info(f"Количество параметров: {param_count}")
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            "param_count": param_count,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs
        })
        
        # Визуализация
        plot_learning_curves(
            train_losses, test_losses, train_accs, test_accs,
            title=f"Кривые обучения для {config['name']}",
            save_path=f"{PLOTS_PATH}/learning_curve_width_{config['name']}.png"
        )
        plot_weight_distribution(
            model, title=f"Распределение весов для {config['name']}",
            save_path=f"{PLOTS_PATH}/weights_width_{config['name']}.png"
        )
        
        logger.info(f"Модель {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Сохранение результатов
    with open(f"{RESULTS_PATH}/width_comparison_results.txt", "w") as f:
        for res in results:
            f.write(f"Модель: {res['name']}\n")
            f.write(f"Точность на тренировке: {res['train_acc']:.4f}\n")
            f.write(f"Точность на тесте: {res['test_acc']:.4f}\n")
            f.write(f"Количество параметров: {res['param_count']}\n")
            f.write(f"Время обучения: {res['training_time']:.2f} сек\n\n")
    
    return results

# 2.2 Оптимизация архитектуры
def grid_search_width_configs():
    """Grid search для поиска оптимальной комбинации ширины слоёв."""
    layer_sizes = [64, 128, 256, 512, 1024]
    schemes = [
        {"name": "expanding", "layers": lambda x: [784, x, 2*x, 4*x, 10]},  # Расширение
        {"name": "constant", "layers": lambda x: [784, x, x, x, 10]},       # Постоянная
        {"name": "narrowing", "layers": lambda x: [784, 4*x, 2*x, x, 10]}   # Сужение
    ]
    results = []
    
    for scheme in schemes:
        for size in layer_sizes:
            config = {
                "name": f"{scheme['name']}_{size}",
                "layers": scheme['layers'](size)
            }
            logger.info(f"Запуск grid search для: {config['name']}")
            model = FullyConnectedNet(config['layers'], dropout=0.0, batch_norm=False).to(DEVICE)
            param_count = count_parameters(model)
            logger.info(f"Количество параметров: {param_count}")
            
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
                "param_count": param_count,
                "test_accs": test_accs  # Для heatmap
            })
            
            logger.info(f"Модель {config['name']}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Время: {training_time:.2f} сек")
    
    # Визуализация результатов в виде heatmap
    plt.figure(figsize=(10, 6))
    sizes = [str(s) for s in layer_sizes]
    schemes_names = [s['name'] for s in schemes]
    heatmap_data = [[0 for _ in layer_sizes] for _ in schemes]
    
    for res in results:
        name_parts = res['name'].split('_')
        scheme_idx = schemes_names.index(name_parts[0])
        size_idx = layer_sizes.index(int(name_parts[1]))
        heatmap_data[scheme_idx][size_idx] = res['test_acc']
    
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", xticklabels=sizes, yticklabels=schemes_names,
                cmap="YlOrRd", cbar_kws={'label': 'Точность на тесте'})
    plt.title("Heatmap точности на тесте для разных схем и ширины")
    plt.xlabel("Базовая ширина слоя")
    plt.ylabel("Схема изменения ширины")
    plt.savefig(f"{PLOTS_PATH}/width_heatmap.png")
    plt.close()
    
    # Сохранение результатов grid search
    with open(f"{RESULTS_PATH}/width_grid_search_results.txt", "w") as f:
        for res in results:
            f.write(f"Модель: {res['name']}\n")
            f.write(f"Точность на тренировке: {res['train_acc']:.4f}\n")
            f.write(f"Точность на тесте: {res['test_acc']:.4f}\n")
            f.write(f"Количество параметров: {res['param_count']}\n")
            f.write(f"Время обучения: {res['training_time']:.2f} сек\n\n")
    
    # Определение оптимальной архитектуры
    best_model = max(results, key=lambda x: x['test_acc'])
    logger.info(f"Оптимальная архитектура: {best_model['name']} с тестовой точностью {best_model['test_acc']:.4f}")

if __name__ == "__main__":
    # Выполнение части 2.1
    width_results = run_width_comparison()
    # Выполнение части 2.2
    grid_search_width_configs()