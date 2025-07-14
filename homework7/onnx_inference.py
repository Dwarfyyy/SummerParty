import onnx
import onnxruntime as ort
import torch
import numpy as np
import time
from model import Resnet18
from datasets import CustomImageDataset
from torch.utils.data import DataLoader

def export_to_onnx(image_size, model_path=f'./weights/best_resnet18_{{size}}.pth'):
    """
    Экспортирует модель в ONNX.
    
    Args:
        image_size (int): Размер изображения.
        model_path (str): Шаблон пути к сохраненной модели.
    """
    model_path = model_path.format(size=image_size)
    model = Resnet18(num_classes=len(CustomImageDataset(root_dir='./data/test').get_class_names()))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, 
        dummy_input, 
        f'./weights/resnet18_{image_size}.onnx', 
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def measure_onnx_inference(image_size, batch_size, num_runs=100):
    """
    Измеряет производительность инференса ONNX Runtime.
    
    Args:
        image_size (int): Размер изображения.
        batch_size (int): Размер батча.
        num_runs (int): Количество прогонов.
    
    Returns:
        dict: Метрики производительности.
    """
    session = ort.InferenceSession(
        f'./weights/resnet18_{image_size}.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    dataloader = DataLoader(
        CustomImageDataset(root_dir='./data/test', target_size=(image_size, image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        for inputs, _ in dataloader:
            inputs = inputs.numpy()
            session.run(None, {'input': inputs})
        end_time = time.time()
        times.append((end_time - start_time) / len(dataloader))
    
    times = np.array(times)
    times = times[int(0.1 * len(times)):int(0.9 * len(times))]
    
    return {
        "mean_time": np.mean(times) * 1000,
        "std_time": np.std(times) * 1000,
        "fps": batch_size / np.mean(times),
        "gpu_memory": 0  # ONNX Runtime не предоставляет прямой доступ к памяти GPU
    }