import cudaq

qubit_count = 2

# Define the simulation target.
cudaq.set_target("nvidia")

# Define a quantum kernel function.
kernel = cudaq.make_kernel()

# Allocate our `qubit_count` to the kernel.
qubits = kernel.qalloc(qubit_count)

# 2-qubit GHZ state.
kernel.h(qubits[0])
for i in range(1, qubit_count):
    kernel.cx(qubits[0], qubits[i])

# If we dont specify measurements, all qubits are measured in
# the Z-basis by default.
kernel.mz(qubits)

result = cudaq.sample(kernel, shots_count=1000)

result.dump()

