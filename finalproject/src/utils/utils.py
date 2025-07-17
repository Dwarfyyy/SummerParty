import torch
import matplotlib.pyplot as plt
import os

def save_plot(data, title, filepath):
    """Сохранение графика в файл.
    
    Args:
        data (list): Данные для графика.
        title (str): Название графика.
        filepath (str): Путь для сохранения.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Эпохи")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def save_model(model, filepath):
    """Сохранение модели.
    
    Args:
        model (torch.nn.Module): Модель для сохранения.
        filepath (str): Путь для сохранения.
    """
    torch.save(model.state_dict(), filepath)

if __name__ == "__main__":
    # Пример использования (для тестирования)
    pass