import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Добавлен импорт numpy как np

def plot_learning_curves(train_losses, test_losses, train_accs, test_accs, title, save_path):
    """Визуализация кривых обучения."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Тренировочные потери")
    plt.plot(test_losses, label="Тестовые потери")
    plt.title(f"{title} - Потери")
    plt.xlabel("Эпоха")
    plt.ylabel("Потери")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Тренировочная точность")
    plt.plot(test_accs, label="Тестовая точность")
    plt.title(f"{title} - Точность")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_weight_distribution(model, title, save_path):
    """Визуализация распределения весов модели."""
    weights = []
    for param in model.parameters():
        if param.requires_grad:
            weights.append(param.data.cpu().numpy().flatten())
    
    weights = np.concatenate(weights)
    plt.figure(figsize=(8, 6))
    sns.histplot(weights, bins=50, kde=True)
    plt.title(f"{title}")
    plt.xlabel("Значения весов")
    plt.ylabel("Частота")
    plt.grid()
    plt.savefig(save_path)
    plt.close()