import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
kappa = 0.5  # Ejemplo de valor para kappa
sigma = 1.0  # Ejemplo de valor para sigma
beta = 0.1   # Ejemplo de valor para beta
gamma = 1.0  # Ejemplo de valor para gamma
Delta = 1/365
n_days = 365

# Inicializar variables
S = np.zeros(n_days)
D = np.zeros(n_days)

# Valores iniciales
S[0] = 6  # Asumimos que comienza en el valor medio estacional
D[0] = 0  # La diferencia inicial es 0

# Simulación del proceso
for i in range(n_days - 1):
    theta = 6 + 2 * np.cos(2 * np.pi * i / 365)
    Z_i = np.random.normal(0, 1)
    W_i = np.random.normal(0, 1)
    
    S[i + 1] = S[i] + kappa * (theta - S[i]) * Delta + sigma * np.sqrt(Delta) * Z_i
    D[i + 1] = D[i] - beta * D[i] * Delta + gamma * np.sqrt(Delta) * W_i

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(S, label='Viento promedio regional (S)')
plt.plot(D, label='Diferencia de viento entre molinos (D)')
plt.xlabel('Día')
plt.ylabel('Valor')
plt.title('Simulación del modelo de viento')
plt.legend()
plt.show()