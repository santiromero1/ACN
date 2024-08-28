import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Definimos los parámetros y funciones iniciales
delta = 1/365
n_days = 365
theta = lambda i: 6 + 2 * np.cos(2 * np.pi * i / 365)

# Cargar datos históricos para S y D
# Supongamos que tienes los datos en un DataFrame con columnas 'Si', 'Si+1', 'Di', 'Di+1' --> Entiendo que esto es lo que despues nos pasan
# Por ejemplo:
# data = pd.read_csv("wind_data.csv")
# Para fines de demostración, generaremos datos simulados:

np.random.seed(42)
data = pd.DataFrame({
    'Si': np.random.normal(6, 2, n_days),
    'Si+1': np.random.normal(6, 2, n_days),
    'Di': np.random.normal(0, 1, n_days),
    'Di+1': np.random.normal(0, 1, n_days),
})

# Estimación de κ y σ
Y_s = data['Si+1'] - data['Si']
X_s = np.vstack([
    theta(np.arange(n_days)) * delta - data['Si'] * delta,
    np.sqrt(delta) * np.random.normal(0, 1, n_days)
]).T

reg_s = LinearRegression().fit(X_s, Y_s)
kappa, sigma = reg_s.coef_
sigma_std = np.std(Y_s - reg_s.predict(X_s))

# Estimación de β y γ
Y_d = data['Di+1'] - data['Di']
X_d = np.vstack([
    -data['Di'] * delta,
    np.sqrt(delta) * np.random.normal(0, 1, n_days)
]).T

reg_d = LinearRegression().fit(X_d, Y_d)
beta, gamma = reg_d.coef_
gamma_std = np.std(Y_d - reg_d.predict(X_d))

# Resultados de la estimación
print(f"Estimación de κ: {kappa}")
print(f"Estimación de σ: {sigma} (Error estándar: {sigma_std})")
print(f"Estimación de β: {beta}")
print(f"Estimación de γ: {gamma} (Error estándar: {gamma_std})")

# Simulación diaria para un año
S_sim = np.zeros(n_days)
D_sim = np.zeros(n_days)
S_sim[0] = data['Si'].iloc[0]
D_sim[0] = data['Di'].iloc[0]

for i in range(1, n_days):
    Z = np.random.normal(0, 1)
    W = np.random.normal(0, 1)
    S_sim[i] = S_sim[i-1] + kappa * (theta(i) - S_sim[i-1]) * delta + sigma * np.sqrt(delta) * Z
    D_sim[i] = D_sim[i-1] - beta * D_sim[i-1] * delta + gamma * np.sqrt(delta) * W

# Graficar resultados
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(S_sim, label='Simulado S')
plt.plot(data['Si'], label='Histórico S', linestyle='--')
plt.title('Viento Regional')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(D_sim, label='Simulado D')
plt.plot(data['Di'], label='Histórico D', linestyle='--')
plt.title('Diferencia de Viento entre Molinos')
plt.legend()

plt.tight_layout()
plt.show()
