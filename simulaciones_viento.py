import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos históricos
historical_data = pd.read_excel('wind_data.xlsx')

# Filtrar los datos hasta el día 365
filtered_data = historical_data[historical_data['dia'] <= 365]

# Parámetros estimados
kappa = 109.21
sigma = 16.04
beta = 48.38
gamma = 10.09

# Parámetros de simulación
n_days = 365
delta = 1 / 365  # Paso de tiempo
np.random.seed(42)  # Para reproducibilidad

# Inicializar arrays para los resultados
S = np.zeros(n_days + 1)  # Viento regional
D = np.zeros(n_days + 1)  # Diferencia de viento
V_norte = np.zeros(n_days)  # Viento en molino Norte
V_sur = np.zeros(n_days)    # Viento en molino Sur

# Inicializar el primer día
S[0] = 6  # Valor inicial del viento regional
D[0] = 0  # Diferencia inicial del viento

# Simulación 
Z = np.random.normal(0, 1, n_days)  # Shock para el viento regional
W = np.random.normal(0, 1, n_days)  # Shock para la diferencia del viento

for i in range(n_days):
    # Calcular θ(t_i)
    theta_t = 6 + 2 * np.cos(2 * np.pi * i / 365)
    
    # Simulación del viento regional
    S[i + 1] = S[i] + kappa * (theta_t - S[i]) * delta + sigma * np.sqrt(delta) * Z[i]
    
    # Simulación de la diferencia de viento
    D[i + 1] = D[i] - beta * D[i] * delta + gamma * np.sqrt(delta) * W[i]
    
    # Calcular el viento en cada molino
    V_norte[i] = S[i] + D[i] / 2
    V_sur[i] = S[i] - D[i] / 2

# Dataframe con datos de la simulación diaria
df_simulation = pd.DataFrame({
    'Día': np.arange(n_days),
    'Viento Regional Simulado': S[:-1],
    'Viento Norte Simulado': V_norte,
    'Viento Sur Simulado': V_sur,
    'Wind Gap Simulado': V_norte - V_sur
})

# Guardar los resultados en un archivo Excel
df_simulation.to_excel('simulacion_viento.xlsx', index=False)

# Graficar los resultados
plt.figure(figsize=(14, 10))

# Primer gráfico: Viento Regional Simulado vs Histórico
plt.subplot(2, 1, 1)
plt.plot(df_simulation['Día'], df_simulation['Viento Regional Simulado'], label='Viento Regional Simulado', linestyle='--', color='grey')
plt.plot(filtered_data['dia'], filtered_data['regional wind'], label='Viento Regional Histórico', linestyle='-', color='blue')
plt.xlabel('Día')
plt.ylabel('Velocidad del Viento')
plt.title('Viento Regional Simulado vs Histórico')
plt.legend()
plt.grid(True)

# Segundo gráfico: Viento Norte, Sur, Wind Gap Simulado vs Wind Gap Histórico
plt.subplot(2, 1, 2)
plt.plot(df_simulation['Día'], df_simulation['Viento Norte Simulado'], label='Viento Norte Simulado', linestyle='-', color='green')
plt.plot(df_simulation['Día'], df_simulation['Viento Sur Simulado'], label='Viento Sur Simulado', linestyle='-', color='red')
plt.plot(filtered_data['dia'], filtered_data['wind gap norte sur'], label='Wind Gap Histórico', linestyle='--', color='purple')
plt.plot(df_simulation['Día'], df_simulation['Wind Gap Simulado'], label='Wind Gap Simulado', linestyle='-', color='orange')
plt.xlabel('Día')
plt.ylabel('Velocidad del Viento')
plt.title('Viento Norte y Sur Simulados vs Wind Gap Histórico y Simulado')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
