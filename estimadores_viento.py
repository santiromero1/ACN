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
data['theta(t)'] = 6 + 2 * np.cos(2 * np.pi * data['dia'] / 365)

# Calcular variable dependiente Y_i = (S_{i+1} - S_i) / delta
data['Y_wind'] = (data['regional wind'].shift(-1) - data['regional wind']) / delta
# Calcular variable dependiente Y_i = (D_{i+1} - D_i) / delta
data['Y_gap'] = (data['wind gap norte sur'].shift(-1) - data['wind gap norte sur']) / delta

# Definir las variables independientes X1 = theta(t) y X2 = S_i
data['X1_wind'] = data['theta(t)']
data['X2_wind'] = data['regional wind']
# Definir las variables independientes X = D_i
data['X1_gap'] = data['wind gap norte sur']

# Eliminar la última fila porque no tiene S_{i+1} (y, por lo tanto, no tiene Y_i)
data = data[:-1]

# Variables independientes y dependiente
X_wind = data[['X1_wind', 'X2_wind']].values
y_wind = data['Y_wind'].values

X_gap = data[['X1_gap']].values
y_gap = data['Y_gap'].values

# Crear y entrenar el modelo de regresión lineal
model_wind = LinearRegression().fit(X_wind, y_wind)
model_gap = LinearRegression().fit(X_gap, y_gap)


## ESTIMADORES ##
# Coeficiente estimado de X1_wind (este es kappa)
kappa = model_wind.coef_[0]

# Coeficiente estimado de X1_gap (este es -β)
beta = -model_gap.coef_[0]

# Estimar sigma usando el error estándar de la regresión
y_pred_wind = model_wind.predict(X_wind)
residuals_wind = y_wind - y_pred_wind
sse_wind = np.sum(residuals_wind**2)  # Suma de los cuadrados de los residuos
n_wind = len(y_wind)
p_wind = X_wind.shape[1]  # Número de parámetros
variance_sigma = sse_wind / (n_wind - p_wind)
sigma_estimate = np.sqrt(variance_sigma)

# Estimar gamma usando la desviación estándar de los residuos
y_pred_gap = model_gap.predict(X_gap)
residuals_gap = y_gap - y_pred_gap
gamma_estimate = np.sqrt(np.mean(residuals_gap**2) / delta)


## ERRORES ##
# Calcular la matriz de covarianza de los coeficientes
X_transpose_X_inv = np.linalg.inv(X_wind.T @ X_wind)
cov_matrix = X_transpose_X_inv * variance_sigma

# Error estándar de kappa
error_kappa = np.sqrt(cov_matrix[0, 0])

# Error estándar de beta
# (Estimar el error estándar de beta usando la desviación estándar de los residuos del gap)
y_pred_gap = model_gap.predict(X_gap)
residuals_gap = y_gap - y_pred_gap
sse_gap = np.sum(residuals_gap**2)  # Suma de los cuadrados de los residuos
n_gap = len(y_gap)
p_gap = X_gap.shape[1]  # Número de parámetros
variance_beta = sse_gap / (n_gap - p_gap)
beta_cov_matrix = np.linalg.inv(X_gap.T @ X_gap) * variance_beta
error_beta = np.sqrt(beta_cov_matrix[0, 0])

# Error estándar de sigma
error_sigma = np.sqrt(np.mean(residuals_wind**2) / n_wind)  # Calcula la desviación estándar de los residuos

# Error estándar de gamma
error_gamma = np.sqrt(variance_beta / (n_gap - p_gap))


## PRINTEAR ESTIMADORES CON RESPECTIVO ERROR ##
print(f"kappa: {kappa:.2f} ± {error_kappa:.2f}")
print(f"sigma: {sigma_estimate:.2f} ± {error_sigma:.2f}")
print(f"beta: {beta:.2f} ± {error_beta:.2f}")
print(f"gamma: {gamma_estimate:.2f} ± {error_gamma:.2f}")