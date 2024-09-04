import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1)
# Datos históricos para el viento regional y la diferencia del viento entre ambos molinos
data = pd.read_excel('wind_data.xlsx')

# Parámetros dados
delta = 1/365

# Calcular theta(t) = 6 + 2 * cos(2 * pi * i / 365)
def theta_func(i):
    return 6 + 2 * np.cos(2 * np.pi * i / 365)

# Calcular variable dependiente Y_i = (S_{i+1} - S_i) / delta
data['Y_wind'] = data['regional wind'].diff().shift(-1).fillna(0)
# Calcular variable dependiente Y_i = (D_{i+1} - D_i) / delta
data['Y_gap'] = data['wind gap norte sur'].diff().shift(-1).fillna(0)

# Construir X_S
X_wind = np.array([((theta_func(i) - data['regional wind'][i]) * delta) for i in range(len(data))]).reshape(-1, 1)
Y_wind = data['Y_wind'].values

X_gap = np.array([((- data['wind gap norte sur'][i]) * delta) for i in range(len(data))]).reshape(-1, 1)
Y_gap = data['Y_gap'].values

# Crear y entrenar el modelo de regresión lineal
model_wind = LinearRegression().fit(X_wind, Y_wind)
model_gap = LinearRegression().fit(X_gap, Y_gap)

## ESTIMADORES ##
kappa = model_wind.coef_[0]
sigma = np.sqrt(np.mean((Y_wind - model_wind.predict(X_wind))**2) / delta)

beta = model_gap.coef_[0]
gamma = np.sqrt(np.mean((Y_gap - model_gap.predict(X_gap))**2) / delta)

print(f"Estimated kappa: {kappa:.2f}")
print(f"Estimated sigma: {sigma:.2f}")
print(f"Estimated beta: {beta:.2f}")
print(f"Estimated gamma: {gamma:.2f}")