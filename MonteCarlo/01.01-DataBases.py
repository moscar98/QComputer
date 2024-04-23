"""
Autor: Oscar Andrés Painen Briones
Fecha: Abril 2024
Tesis
"""
# --------------------------  Librerias  --------------------------
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import requests
import os 
import seaborn as sns

import plotly.express as px
import plotly.io as pio

# -------------------------- Lectura de Directorios --------------------------
# Usar en caso de ser Necesario
Directorio = os.getcwd()
imagenes = 'Images'
data = 'Data'
path2img  = os.path.join(Directorio,imagenes)
path2data  = os.path.join(Directorio,data)

# -------------------------- ALPHAVANTANGE --------------------------
# API DE STOCK DIARIOS 100 DIAS 
# Genera un archivo .csv que contiene :
# Series de tiempo diarias (fechas, apertura, maximo, bajo, cierre, volumen) [diario]
# cubre 20+ años de historicos (acá solo hay 100 dias)

# agregar '&outputsize:full' para obtener historicos
# Cantidad maxima de revisiones diarias == 25
"""
api_key = 'HCD1BI5ZR35ZSHGA' 
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey={api_key}'
r = requests.get(url)
data = r.json()
print(data)
time_series = data['Time Series (Daily)']
"""
time_series = os.path.join(path2data,'daily_IBM.csv')
# Convertir los datos a un DataFrame de pandas
df = pd.read_csv(time_series)
# Procesamiento de Columnas
df.columns = ['Fecha','Apertura', 'Maximo', 'Minimo', 'Cierre', 'Volumen']
df['Fecha'] = pd.to_datetime(df['Fecha'])
df[['Apertura', 'Maximo', 'Minimo', 'Cierre', 'Volumen']] = df[['Apertura', 'Maximo', 'Minimo', 'Cierre', 'Volumen']].apply(pd.to_numeric, errors='coerce')


# ------------------------- Muestreo de Datos -------------------------
print('Columnas')
print(df.columns)
print('-------------------------','\n')
print('Datos')
print(df)
print('-------------------------','\n')
print('Analisis Descriptivo de las acciones diarias')
print(df['Cierre'].describe(include='all'))
print(df.shape)
print('-------------------------','\n')
# ------------------------- Muestreo de la Serie -------------------------

# Gráficos de serie temporal de precios de cierre
"""
# Graficos con plt

plt.figure(figsize=(16, 9))
plt.plot(df['Fecha'], df['Cierre'], label='Precio de Cierre')
plt.title('Evolución del Precio de Cierre')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre ($)')
plt.legend()
plt.savefig(os.path.join(path2save,'CierreDiario_plt.png')) #Guardar Figura 
plt.show()
"""
"""
# Graficos con seaborn

plt.figure(figsize=(16, 9))
sns.lineplot(x='Fecha', y='Cierre', data=df[['Fecha','Cierre']], label='Precio de Cierre')
plt.title('Evolución del Precio de Cierre')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre ($)')
plt.legend()
plt.savefig(os.path.join(path2save, 'CierreDiario_sns.png'))  # Guardar Figura
plt.show()
"""

# Graficos con plotly

fig = px.line(df, # Ingreso DataFrame Completo
              x='Fecha', y='Cierre', # Agrega a los ejes las columnas que quiero plotear 
              title='Evolución del Precio de Cierre',
              labels={'Cierre': 'Precio de Cierre ($)'}, # Nombre de los ejes
              markers=True # Agrega marcadores para cada punto de datos
              )  
fig.update_traces(marker=dict(size= 10), # Ajustar el grosor del marcador
                  line=dict(width=4), # Ajustar el grosor de la línea
                  )#fill='tozeroy') 

fig.update_layout(legend_title_text='Legenda',
                  autosize=False,
                  width=1920,
                  height=1080
                  )

# Para guardar el gráfico como una imagen estática
pio.write_image(fig, os.path.join(path2img, 'CierreDiario_pio.png'))

# ------------------------- Muestreo Estadistico de la Serie -------------------------
# Cálculo de estadísticas descriptivas básicas
media = df['Cierre'].mean()
mediana = df['Cierre'].median()
moda = df['Cierre'].mode()[0]  # mode() devuelve una Serie, toma el primer valor
maximo = df['Cierre'].max()
minimo = df['Cierre'].min()
desviacion_std = df['Cierre'].std()
coef_variacion = (desviacion_std / media) * 100  # Coeficiente de variación como porcentaje
asimetria = df['Cierre'].skew()
curtosis = df['Cierre'].kurt()

# Imprimir resultados
print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Moda: {moda}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}")
print(f"Desviación estándar: {desviacion_std}")
print(f"Coeficiente de variación: {coef_variacion:.2f}%")
print(f"Asimetría: {asimetria}")
print(f"Curtosis: {curtosis}")
print('-------------------------','\n')

# ------------------------- Rendimiento y Volatilidad -------------------------

# Calculo del rendimiento diario con el precio de cierre de las acciones
df['Rendimiento'] = np.log(df['Cierre']) - np.log(df['Cierre'].shift(1))
df.loc[0,'Rendimiento'] = 0

# Calculando la volatilidad móvil para una ventana de (100 - 30 días | largo - corto plazo) 
# Para Calculo de la volatilidad anual -> * (252**0.5) == sqrt(252)
# 252 son los dias operables de una accion
df['Volatilidad Anualizada'] = df['Rendimiento'].rolling(window=100).std() * (252**0.5)

print('Rendimiento Diario')
print(df['Rendimiento'])
print('Volatilidad Anualizada: ' , df['Volatilidad Anualizada'].iloc[-1])
print('-------------------------','\n')
print(df)
df.to_csv(os.path.join(path2data,'IBM-Data.csv'),index=False)