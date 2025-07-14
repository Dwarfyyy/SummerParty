import torch
import torch_tensorrt
import os
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from model import Resnet18
from datasets import CustomImageDataset, RandomImageDataset
from utils import run_test

def convert_to_torch_trt(
    model_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'fp16',
    workspace_size: int = 1 << 30,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    opt_batch_size: int = 1,
    **kwargs
):
    """
    Конвертирует PyTorch модель в TensorRT через torch-tensorrt
    
    Args:
        model_path: Путь к сохраненной PyTorch модели
        input_shape: Форма входного тензора
        precision: Точность (fp32, fp16)
        workspace_size: Размер рабочего пространства в байтах
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        opt_batch_size: Оптимальный размер батча
    
    Returns:
        Скомпилированная TensorRT модель
    """
    # Загружаем модель
    num_classes = len(CustomImageDataset(root_dir='./data/test').get_class_names())
    model = Resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to('cuda')
    model.eval()
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    precision_dtype = torch.float16 if precision == 'fp16' else torch.float32
    min_shape = (min_batch_size, *input_shape)
    opt_shape = (opt_batch_size, *input_shape)
    max_shape = (max_batch_size, *input_shape)
    
    # Конвертируем модель
    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float16
            )],
            enabled_precisions={precision_dtype},
            workspace_size=workspace_size
        )
        
        # Сохраняем модель
        output_path = model_path.replace('.pth', '.trt')
        inputs = [
            torch.randn(min_shape, device='cuda'),
            torch.randn(opt_shape, device='cuda'),
            torch.randn(max_shape, device='cuda')
        ]
        torch_tensorrt.save(trt_model, output_path, inputs=inputs)
        
        print(f"Модель успешно конвертирована в TensorRT: {output_path}")
        return trt_model
    except Exception as e:
        print(f"Ошибка конвертации в TensorRT: {str(e)}")
        return None

def test_torch_trt_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 100,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    **kwargs
) -> dict[Tuple[int, int, int], float]:
    """
    Тестирует torch-tensorrt модель
    
    Args:
        model: Скомпилированная TensorRT модель
        input_shape: Форма входного тензора
        num_runs: Количество прогонов
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        dataset: Датасет для тестирования
    
    Returns:
        Словарь с результатами тестирования
    """
    return run_test(
        model_wrapper=model,
        input_shape=input_shape,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_step=batch_step,
        dataset=dataset,
        timer_type='cuda'
    )

if __name__ == '__main__':
    # Инициализируем CUDA контекст
    torch.cuda.init()
    
    # Создаем датасет
    dataset = CustomImageDataset(root_dir='./data/test', target_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    
    # Конвертируем и тестируем модель
    model_path = './weights/best_resnet18_224.pth'
    if os.path.exists(model_path):
        trt_model = convert_to_torch_trt(
            model_path=model_path,
            input_shape=(3, 224, 224),
            precision='fp16',
            min_batch_size=1,
            max_batch_size=16,
            opt_batch_size=16
        )
        if trt_model is not None:
            results = test_torch_trt_model(
                model=trt_model,
                input_shape=(3, 224, 224),
                num_runs=50,
                min_batch_size=1,
                max_batch_size=16,
                batch_step=8,
                dataset=dataset
            )
            for shape, time in results.items():
                print(f"Shape: {shape}, Time: {time:.4f} ms")