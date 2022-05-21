import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def estandarizacion(data, column_1, column_2):
    x = data[column_1]
    y = data[column_2]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    x_standard = ((x - mean_x) / std_x).tolist()
    y_standard = ((y - mean_y) / std_y).tolist()
    return x_standard, y_standard


def get_gradient_at_b(x, y, b, m):
    N = len(x)
    diff = 0
    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        diff += y_val - ((m * x_val) + b)
    b_gradient = -(2 / N) * diff
    return b_gradient


def get_gradient_at_m(x, y, b, m):
    N = len(x)
    diff = 0
    for i in range(N):
        x_val = x[i]
        y_val = y[i]
        diff += x_val * (y_val - ((m * x_val) + b))
    m_gradient = -(2 / N) * diff
    return m_gradient


def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return b, m


def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return b, m


data = pd.read_csv("Insurance_Data.csv")
x_standard, y_standard = estandarizacion(data, "Num Claims", "Amount for Claims")

b, m = gradient_descent(x_standard, y_standard, 0.001, 10000)
y = [m * x + b for x in x_standard]

# Error cuadrático medio
mse = mean_squared_error(y_standard, y)
print("Error cuadrático medio: ")
print(mse)

plt.plot(x_standard, y_standard, "o")
plt.plot(x_standard, y)

plt.show()
