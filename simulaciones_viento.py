import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Cargar los datos históricos
historical_data = pd.read_excel('wind_data.xlsx')

# Filtrar los datos hasta el día 365
filtered_data = historical_data[historical_data['dia'] <= 365]


# Parámetros estimados
kappa = 105.60 # ± 4.34
sigma = 306.57 # ± 5.07
beta = 48.38 # ± 3.01
gamma = 5855.45 # ± 3.19

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
    'Viento Regional': S[:-1],
    'Viento Norte': V_norte,
    'Viento Sur': V_sur,
})

# Guardar los resultados en un archivo Excel
df_simulation.to_excel('simulacion_viento.xlsx', index=False)

# Graficar los resultados
plt.figure(figsize=(14, 7))
plt.plot(df_simulation['Día'], df_simulation['Viento Norte'], label='Viento Norte')
plt.plot(df_simulation['Día'], df_simulation['Viento Sur'], label='Viento Sur')
plt.plot(df_simulation['Día'], df_simulation['Viento Regional'], label='Viento Regional', linestyle='--', color='grey')
plt.xlabel('Día')
plt.ylabel('Velocidad del Viento')
plt.title('Simulación del Viento en Molinos Norte y Sur')
plt.legend()
plt.grid(True)
plt.show()