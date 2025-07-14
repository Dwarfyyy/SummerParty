import torch
import time
import numpy as np
from model import Resnet18
from datasets import CustomImageDataset
from torch.utils.data import DataLoader

def measure_pytorch_inference(image_size, batch_size, num_runs=100):
    """
    Измеряет производительность инференса PyTorch.
    
    Args:
        image_size (int): Размер изображения.
        batch_size (int): Размер батча.
        num_runs (int): Количество прогонов.
    
    Returns:
        dict: Метрики производительности.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CustomImageDataset(root_dir='./data/test').get_class_names())
    model = Resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f'./weights/best_resnet18_{image_size}.pth'))
    model.eval()
    
    dataloader = DataLoader(
        CustomImageDataset(root_dir='./data/test', target_size=(image_size, image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                model(inputs)
            end_time = time.time()
            times.append((end_time - start_time) / len(dataloader))
    
    times = np.array(times)
    times = times[int(0.1 * len(times)):int(0.9 * len(times))]
    
    return {
        "mean_time": np.mean(times) * 1000,
        "std_time": np.std(times) * 1000,
        "fps": batch_size / np.mean(times),
        "gpu_memory": torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0
    }