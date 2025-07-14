import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

COLORS = {
    'PyTorch': ['#FF6384', '#FF9F40', '#FFCD56'],  # Красный, оранжевый, желтый для batch_size 1, 8, 16
    'ONNX': ['#36A2EB', '#4BC0C0', '#9966FF'],     # Синий, голубой, фиолетовый
    'Torch-TensorRT': ['#FF66CC', '#FF33FF', '#CC00CC']  # Розовый, пурпурный, темно-пурпурный
}

def plot_fps_vs_image_size(df):
    """
    Строит график FPS в зависимости от размера изображения на черном фоне.
    
    Args:
        df (pd.DataFrame): DataFrame с результатами бенчмарка.
    """
    plt.figure(figsize=(10, 6))
    for approach in df['approach'].unique():
        subset = df[df['approach'] == approach]
        for idx, batch_size in enumerate(subset['batch_size'].unique()):
            subsubset = subset[subset['batch_size'] == batch_size]
            plt.plot(subsubset['image_size'], subsubset['fps'], marker='o', 
                     color=COLORS[approach][idx], linewidth=2, markersize=8,
                     label=f"{approach} (батч {batch_size})")
    
    plt.xlabel('Размер изображения', fontsize=12, color='white')
    plt.ylabel('FPS', fontsize=12, color='white')
    plt.title('Производительность (FPS) в зависимости от размера изображения', fontsize=14, color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.grid(True, color='gray', linestyle='--', alpha=0.7)
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/fps_vs_image_size.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def plot_fps_vs_batch_size(df):
    """
    Строит график FPS в зависимости от размера батча на черном фоне.
    
    Args:
        df (pd.DataFrame): DataFrame с результатами бенчмарка.
    """
    plt.figure(figsize=(10, 6))
    for approach in df['approach'].unique():
        subset = df[df['approach'] == approach]
        for idx, image_size in enumerate(subset['image_size'].unique()):
            subsubset = subset[subset['image_size'] == image_size]
            color_idx = min(idx, len(COLORS[approach]) - 1)  # Ограничение индекса цвета
            plt.plot(subsubset['batch_size'], subsubset['fps'], marker='o', 
                     color=COLORS[approach][color_idx], linewidth=2, markersize=8,
                     label=f"{approach} (размер {image_size})")
    
    plt.xlabel('Размер батча', fontsize=12, color='white')
    plt.ylabel('FPS', fontsize=12, color='white')
    plt.title('Производительность (FPS) в зависимости от размера батча', fontsize=14, color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.grid(True, color='gray', linestyle='--', alpha=0.7)
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/fps_vs_batch_size.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def plot_acceleration(df):
    """
    Строит график ускорения относительно PyTorch (FP32) на черном фоне.
    
    Args:
        df (pd.DataFrame): DataFrame с результатами бенчмарка.
    """
    plt.figure(figsize=(10, 6))
    baseline = df[(df['approach'] == 'PyTorch')][['image_size', 'batch_size', 'mean_time']]
    for approach in df['approach'].unique():
        if approach == 'PyTorch':
            continue
        subset = df[df['approach'] == approach]
        for idx, batch_size in enumerate(subset['batch_size'].unique()):
            subsubset = subset[subset['batch_size'] == batch_size]
            base_times = baseline[baseline['batch_size'] == batch_size][['image_size', 'mean_time']].set_index('image_size')
            accel = base_times['mean_time'] / subsubset.set_index('image_size')['mean_time']
            plt.scatter(accel.index, accel.values, 
                        color=COLORS[approach][idx], s=100,
                        label=f"{approach} (батч {batch_size})")
    
    plt.xlabel('Размер изображения', fontsize=12, color='white')
    plt.ylabel('Ускорение (x)', fontsize=12, color='white')
    plt.title('Ускорение относительно PyTorch (FP32)', fontsize=14, color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.grid(True, color='gray', linestyle='--', alpha=0.7)
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/acceleration.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()