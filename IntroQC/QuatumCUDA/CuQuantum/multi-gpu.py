import cudaq

targets = cudaq.get_targets()

# for target in targets:
print(targets)

def ghz_state(qubit_count, target):
    """A function that will generate a variable sized GHZ state (`qubit_count`)."""
    cudaq.set_target(target)

    kernel = cudaq.make_kernel()

    qubits = kernel.qalloc(qubit_count)

    kernel.h(qubits[0])

    for i in range(1, qubit_count):
        kernel.cx(qubits[0], qubits[i])

    kernel.mz(qubits)

    result = cudaq.sample(kernel, shots_count=1000)

    return result

cpu_result = ghz_state(n_qubits=2, target="default")

cpu_result.dump()

gpu_result = ghz_state(n_qubits=25, target="nvidia")

gpu_result.dump()

cudaq.set_target("nvidia-mqpu")

qubit_count = 15
term_count = 100000

kernel = cudaq.make_kernel()

qubits = kernel.qalloc(qubit_count)

kernel.h(qubits[0])

for i in range(1, qubit_count):
    kernel.cx(qubits[0], qubits[i])

# We create a random hamiltonian with 10e5 terms
hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

# The observe calls allows us to calculate the expectation value of the Hamiltonian, batches the terms, and distributes them over the multiple QPU's/GPUs.

# expectation = cudaq.observe(kernel, hamiltonian)  # Single node, single GPU.

expectation = cudaq.observe(
    kernel, hamiltonian,
    execution=cudaq.parallel.thread)  # Single node, multi-GPU.

# expectation = cudaq.observe(kernel, hamiltonian, execution= cudaq.parallel.mpi) # Multi-node, multi-GPU.


