# --------------------------  Librerias  --------------------------
import cirq
import os
import matplotlib.pyplot as plt
# -------------------------- Lectura de Directorios ----------------

Directorio = os.getcwd()
carpeta_img = 'Images'
path2img = os.path.join(Directorio,carpeta_img)
# --------------------- Crear un qubit -----------------------------
qubit = cirq.GridQubit(0, 0)


# Crear un circuito
circuit = cirq.Circuit()
# Añadir la puerta de Hadamard al circuito
circuit.append(cirq.H(qubit))
# Añadir una medición al final
circuit.append(cirq.measure(qubit, key='result'))


# Visualizar el circuito
print("Circuito:")
print(circuit)

# Simular el circuito
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)

# Obtener y mostrar resultados
print("\nResultados de la medición:")
print(result.histogram(key='result'))


# Visualización básica 


# Preparar los datos para la gráfica
histogram = result.histogram(key='result')
plt.bar(histogram.keys(), histogram.values(), tick_label=[str(k) for k in histogram.keys()])
plt.xlabel('Estado medido')
plt.ylabel('Frecuencia')
plt.savefig(os.path.join(path2img,'plot1.png'))
plt.show()

help(simulator)  