import cudaq
from cudaq import spin

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

# Define a Hamiltonian in terms of Pauli Spin operators.
hamiltonian = spin.z(0) + spin.y(1) + spin.x(0) * spin.z(0)

result = cudaq.observe(kernel, hamiltonian)

