# predict.py
import torch
import torch.nn as nn
from src.models.model import CarLogoCNN
from src.data.dataset import test_transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import random

def predict_image(model, image_path, device):
    """Выполнение предсказания на одном изображении.
    
    Args:
        model (torch.nn.Module): Обученная модель.
        image_path (str): Путь к изображению.
        device (torch.device): Устройство для вычислений (CPU/GPU).
    
    Returns:
        tuple: (предсказанный класс, изображение как массив numpy).
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image=np.array(image))["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
        class_idx = pred.item()
        classes = ["hyundai", "lexus", "mazda", "mercedes", "opel", "skoda", "toyota", "volkswagen"]
        predicted_class = classes[class_idx]
    
    # Конвертация изображения в массив для коллажа
    img_array = np.array(Image.open(image_path).resize((100, 100), Image.Resampling.LANCZOS))  # Уменьшение размера для коллажа
    return predicted_class, img_array

def get_random_images(data_dir, num_images=2):
    """Получение списка случайных изображений из каждой папки.
    
    Args:
        data_dir (str): Путь к директории с данными (Train или Test).
        num_images (int): Количество случайных изображений из каждой папки.
    
    Returns:
        list: Список путей к случайным изображениям.
    """
    image_paths = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            jpg_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(jpg_files) >= num_images:
                selected_files = random.sample(jpg_files, num_images)
                image_paths.extend([os.path.join(class_dir, f) for f in selected_files])
    return image_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/models/efficientnet_b0.pth", help="Путь к сохраненной модели")
    parser.add_argument("--data_dir", type=str, default="data", help="Путь к корневой директории данных (содержит Train и Test)")
    args = parser.parse_args()

    # Инициализация
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarLogoCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Получение путей к случайным изображениям
    train_images = get_random_images(os.path.join(args.data_dir, "Train"))
    test_images = get_random_images(os.path.join(args.data_dir, "Test"))
    all_images = train_images + test_images

    # Список для хранения результатов
    predictions = []
    images = []

    # Предсказание для каждого изображения
    for image_path in all_images:
        try:
            predicted_class, img_array = predict_image(model, image_path, device)
            print(f"Изображение: {image_path}, Предсказанный класс: {predicted_class}")
            predictions.append(predicted_class)
            images.append(img_array)
        except FileNotFoundError as e:
            print(e)

    # Создание коллажа
    num_images = len(images)
    cols = 4  # 4 колонки для 4x4 сетки
    rows = (num_images + cols - 1) // cols  # Вычисление строк

    plt.figure(figsize=(cols * 2, rows * 2))  # Увеличенный размер для читаемости
    for i, (img, pred) in enumerate(zip(images, predictions)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{pred}")
        plt.axis("off")

    # Сохранение коллажа
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    collage_filename = f"results/predictions/collage_{timestamp}.png"
    plt.savefig(collage_filename)
    plt.close()

    # Сохранение текстового отчета
    with open(f"results/predictions/collage_{timestamp}.txt", "w") as f:
        f.write(f"Дата и время: {timestamp} (CEST)\n")
        for i, (image_path, pred) in enumerate(zip(all_images, predictions), 1):
            f.write(f"Изображение {i}: {image_path}, Предсказанный класс: {pred}\n")