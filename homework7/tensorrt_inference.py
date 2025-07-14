import torch
import torch_tensorrt
import time
import numpy as np
from model import Resnet18
from datasets import CustomImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def measure_tensorrt_inference(image_size, batch_size, num_runs=100):
    """
    Измеряет производительность инференса Torch-TensorRT.
    
    Args:
        image_size (int): Размер изображения.
        batch_size (int): Размер батча.
        num_runs (int): Количество прогонов.
    
    Returns:
        dict: Метрики производительности или None при ошибке.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_classes = len(CustomImageDataset(root_dir='./data/test').get_class_names())
        model = Resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(f'./weights/best_resnet18_{image_size}.pth'))
        model.eval()
        
        inputs = torch.randn(batch_size, 3, image_size, image_size).to(device)
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                min_shape=(1, 3, image_size, image_size),
                opt_shape=(batch_size, 3, image_size, image_size),
                max_shape=(batch_size, 3, image_size, image_size),
                dtype=torch.float16
            )],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30
        )
        
        dataloader = DataLoader(
            CustomImageDataset(root_dir='./data/test', target_size=(image_size, image_size)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        times = []
        
        with torch.no_grad():
            # Прогрев
            for _ in tqdm(range(10), desc="Прогрев Torch-TensorRT"):
                dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
                trt_model(dummy_input)
            
            for _ in tqdm(range(num_runs), desc=f"Тестирование Torch-TensorRT (размер {image_size}, батч {batch_size})"):
                start_time = time.time()
                for inputs, _ in dataloader:
                    inputs = inputs.to(device)
                    trt_model(inputs)
                end_time = time.time()
                times.append((end_time - start_time) / len(dataloader))
        
        times = np.array(times)
        times = times[int(0.1 * len(times)):int(0.9 * len(times))]
        
        # Логирование
        os.makedirs('./results', exist_ok=True)
        with open(f'./results/tensorrt_log_{image_size}_{batch_size}.txt', 'w') as f:
            f.write(f"Torch-TensorRT, размер: {image_size}, батч: {batch_size}\n")
            f.write(f"Среднее время: {np.mean(times) * 1000:.2f} мс\n")
            f.write(f"STD времени: {np.std(times) * 1000:.2f} мс\n")
            f.write(f"FPS: {batch_size / np.mean(times):.2f}\n")
            f.write(f"Использовано памяти GPU: {torch.cuda.memory_allocated() / 1024**2:.2f} МБ\n")
        
        return {
            "mean_time": np.mean(times) * 1000,
            "std_time": np.std(times) * 1000,
            "fps": batch_size / np.mean(times),
            "gpu_memory": torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0
        }
    except Exception as e:
        print(f"Ошибка Torch-TensorRT для размера {image_size}, батча {batch_size}: {str(e)}")
        return None