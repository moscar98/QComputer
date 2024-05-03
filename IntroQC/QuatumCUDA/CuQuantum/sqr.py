import cudaq
from cudaq import spin

cudaq.set_target("nvidia")

qubit_count = 1

# Initialize a kernel/ ansatz and variational parameters.
kernel, parameters = cudaq.make_kernel(list)

# Allocate qubits that are initialised to the |0> state.
qubits = kernel.qalloc(qubit_count)

# Define gates and the qubits they act upon.
kernel.rx(parameters[0], qubits[0])
kernel.ry(parameters[1], qubits[0])

# Our hamiltonian will be the Z expectation value of our qubit.
hamiltonian = spin.z(0)

# Initial gate parameters which intialize the qubit in the zero state
initial_parameters = [0, 0]

cost_values = []
#cost_values.append(initial_cost_value)


def cost(parameters):
    """Returns the expectation value as our cost."""
    expectation_value = cudaq.observe(kernel, hamiltonian,
                                      parameters).expectation_z()
    cost_values.append(expectation_value)
    return expectation_value

# We see that the initial value of our cost function is one, demonstrating that our qubit is in the zero state
initial_cost_value = cost(initial_parameters)
print(initial_cost_value)

# Define a CUDA Quantum optimizer.
optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = initial_parameters

result = optimizer.optimize(dimensions=2, function=cost)

# Plotting how the value of the cost function decreases during the minimization procedure.
# !pip install matplotlib
import matplotlib.pyplot as plt

x_values = list(range(len(cost_values)))
y_values = cost_values

plt.plot(x_values, y_values)

plt.xlabel("Epochs")
plt.ylabel("Cost Value")

plt.savefig("output.jpg")

