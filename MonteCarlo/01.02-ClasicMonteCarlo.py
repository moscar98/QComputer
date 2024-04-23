
# --------------------------  Librerias  --------------------------
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# -------------------------- Lectura de Directorios --------------------------

Directorio = os.getcwd()
carpeta = 'Data'
img = 'Images'
path2data = os.path.join(Directorio,carpeta)
path2img = os.path.join(Directorio,img)

df = pd.read_csv(os.path.join(path2data,'IBM-Data.csv'))

print('DataFrame Informacion Util - Acciones IBM')
print(df)
print('-------------------------','\n')
# -------------------------- MonteCarlo Simulations --------------------------

# Parámetros 
# ----------
S0 = df['Cierre'].iloc[-1] # Precio actual de la acción IBM (ultimo cierre)
K = S0*1.1   # Precio de ejercicio de la opción (agregar precio de pago de la opcion %10)
T = 0.25  # Tiempo hasta el vencimiento en años (3-6-12 meses)
r = 0.015  # Tasa de interés libre de riesgo anual para 3 meses |   NO ENTENDI
sigma = df['Volatilidad Anualizada'].iloc[-1]  # Volatilidad anual estimada
n_simulations = 100000 # Cantidad de Simulaciones 1M+ 
n_steps = int(T * 252)  # Número de pasos (días de trading en 3 meses)
# ----------
# Generalizar para Parametros
dt = T/n_steps #pasos
t = np.linspace(0, T, n_steps)
S = np.zeros((n_simulations, n_steps))
S[:, 0] = S0

nudt = (r - 0.5 * sigma **2) * dt #parametros de MC | (Tasa de Interes - Volatilidad^2 /2)*t
sidt = sigma * np.sqrt(dt) #parametros de MC | Discretizacion del sigma*W(t)

for i in range(1,n_steps):
    z = np.random.standard_normal(n_simulations)
    S[:, i] = S[:, i - 1] * np.exp(nudt + sidt * z)
    
# Visualizacion de la Simulacion
plt.figure(figsize = (10,8))
for i in range(len(S)):
    plt.plot(S[i])
plt.xlabel('Numbers of steps')
plt.ylabel('Stock price')
plt.title('Monte Carlo Simulation for Stock Price')
plt.savefig(os.path.join(path2img,'montecarlo.png'))

# -------------------------- Muestra de Solucion --------------------------
# Cálculo del precio de la opción de compra
payoffs = np.maximum(S[:, -1] - K, 0)
option_price = np.exp(-r * T) * np.mean(payoffs)

print(f"El precio estimado de la opción de compra es {option_price:.2f}")