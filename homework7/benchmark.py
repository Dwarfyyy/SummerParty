import pandas as pd
import matplotlib.pyplot as plt
import os
from pytorch_inference import measure_pytorch_inference
from onnx_inference import export_to_onnx, measure_onnx_inference
from tensorrt_inference import measure_tensorrt_inference
from utils.plot_utils import plot_fps_vs_image_size, plot_fps_vs_batch_size, plot_acceleration

def run_benchmark():
    """
    Выполняет бенчмарк производительности для всех подходов.
    """
    image_sizes = [224, 256, 384, 512]
    batch_sizes = [1, 8, 16]  # Уменьшены для GTX 1650 (4 ГБ)
    results = []
    
    for image_size in image_sizes:
        export_to_onnx(image_size)  # Экспорт модели в ONNX
        for batch_size in batch_sizes:
            print(f"Тестирование: размер изображения {image_size}, размер батча {batch_size}")
            
            # PyTorch
            pytorch_metrics = measure_pytorch_inference(image_size, batch_size)
            results.append({
                "approach": "PyTorch",
                "image_size": image_size,
                "batch_size": batch_size,
                **pytorch_metrics
            })
            
            # ONNX Runtime
            onnx_metrics = measure_onnx_inference(image_size, batch_size)
            results.append({
                "approach": "ONNX",
                "image_size": image_size,
                "batch_size": batch_size,
                **onnx_metrics
            })
            
            # Torch-TensorRT
            tensorrt_metrics = measure_tensorrt_inference(image_size, batch_size)
            if tensorrt_metrics is not None:
                results.append({
                    "approach": "Torch-TensorRT",
                    "image_size": image_size,
                    "batch_size": batch_size,
                    **tensorrt_metrics
                })
    
    # Сохранение результатов
    df = pd.DataFrame(results)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/benchmark_results.csv', index=False)
    
    # Построение графиков
    plot_fps_vs_image_size(df)
    plot_fps_vs_batch_size(df)
    plot_acceleration(df)

if __name__ == "__main__":
    run_benchmark()