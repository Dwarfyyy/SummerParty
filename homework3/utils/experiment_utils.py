import torch
import logging

logger = logging.getLogger(__name__)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    """Обучение модели с логированием метрик."""
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        logger.info(f"Эпоха {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return train_losses, test_losses, train_accs, test_accs

def evaluate_model(model, loader, device, criterion=None):
    """Оценка модели на датасете."""
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.view(inputs.size(0), -1))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if criterion:
                loss += criterion(outputs, labels).item()
    
    acc = correct / total
    loss = loss / len(loader) if criterion else 0.0
    return loss, acc