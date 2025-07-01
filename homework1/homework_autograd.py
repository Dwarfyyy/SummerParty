import torch

# 2.1 Простые вычисления с градиентами
# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z

# Вычисляем градиенты
f.backward()

# Получаем градиенты
grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

print(f"Градиент по x: {grad_x}")
print(f"Градиент по y: {grad_y}")
print(f"Градиент по z: {grad_z}")

# Градиент по x: 14.0
# Градиент по y: 10.0
# Градиент по z: 10.0

# Аналитическая проверка
def analytical_gradients(x, y, z):
    df_dx = 2*x + 2*y*z
    df_dy = 2*y + 2*x*z
    df_dz = 2*z + 2*x*y
    return df_dx, df_dy, df_dz

analytical_grads = analytical_gradients(x.item(), y.item(), z.item())

print("\nАналитические градиенты:")
print(f"df/dx: {analytical_grads[0]}")
print(f"df/dy: {analytical_grads[1]}")
print(f"df/dz: {analytical_grads[2]}")

# Аналитические градиенты:
# df/dx: 14.0
# df/dy: 10.0
# df/dz: 10.0

# Проверка совпадения
# Автоматически вычисленные градиенты:
# grad_x = 14.0, grad_y = 10.0, grad_z = 10.0
# Аналитические градиенты:
# df/dx = 14.0, df/dy = 10.0, df/dz = 10.0
# # Результаты полностью совпадают, что подтверждает корректность вычислений

# 2.2 Градиент функции потерь
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b
import torch

# Определим входные данные
x = torch.tensor([1.0, 2.0, 3.0, 4.0])  # входные признаки
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])  # истинные значения

# Инициализируем параметры модели с градиентами
w = torch.tensor(1.0, requires_grad=True)  # вес (начальное значение 1.0)
b = torch.tensor(0.0, requires_grad=True)  # смещение (начальное значение 0.0)

# Прямой проход: вычисляем предсказания
y_pred = w * x + b

# Вычисляем MSE
loss = torch.mean((y_pred - y_true) ** 2)

# Вычисление градиентов которое обратное распространение ошибки
loss.backward()

# Получаем градиенты
grad_w = w.grad
grad_b = b.grad

print(f"MSE: {loss.item():.4f}")
print(f"Градиент по w: {grad_w.item():.4f}")
print(f"Градиент по b: {grad_b.item():.4f}")

# MSE: 7.5000
# Градиент по w: -15.0000
# Градиент по b: -5.0000

# 2.3 Цепное правило
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad

# Создаем входной тензор с включенным градиентом
x = torch.tensor(2.0, requires_grad=True)

# Вычисляем функцию f(x) = sin(x^2 + 1)
y = torch.sin(x**2 + 1)

# Вычисляем градиент автоматически
y.backward()
auto_grad = x.grad

# Вычисляем градиент вручную по цепному правилу
manual_grad = torch.cos(x**2 + 1) * 2 * x

# Сравниваем результаты
print(f"Автоматический градиент: {auto_grad.item()}")
print(f"Градиент по цепному правилу: {manual_grad.item()}")
print(f"Совпадают ли результаты: {torch.isclose(auto_grad, manual_grad)}")

# Автоматический градиент: 1.1346487998962402
# Градиент по цепному правилу: 1.1346487998962402
# Совпадают ли результаты: True