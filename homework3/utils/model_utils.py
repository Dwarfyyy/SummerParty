import torch
import torch.nn as nn

class FullyConnectedNet(nn.Module):
    """Полносвязная нейронная сеть с настраиваемой глубиной и регуляризацией."""
    def __init__(self, layers, dropout=0.0, batch_norm=False):
        super(FullyConnectedNet, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:  # Не добавляем для последнего слоя
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(layers[i+1]))
                self.layers.append(nn.ReLU())
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x