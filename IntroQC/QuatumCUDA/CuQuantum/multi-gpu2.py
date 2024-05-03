import cudaq
from cudaq import spin
import numpy as np
import timeit 

np.random.seed(1)

cudaq.set_target("nvidia-mqpu")

qubit_count = 5
sample_count = 10000
h = spin.z(0)
parameter_count = qubit_count

# Below we run a circuit for 10000 different input parameters.
parameters = np.random.default_rng(13).uniform(low=0,
                                               high=1,
                                               size=(sample_count,
                                                     parameter_count))

kernel, params = cudaq.make_kernel(list)

qubits = kernel.qalloc(qubit_count)
qubits_list = list(range(qubit_count))

for i in range(qubit_count):
    kernel.rx(params[i], qubits[i])


start = timeit.default_timer()
result = cudaq.observe(kernel, h, parameters)   # Single GPU result.
print("The difference of time is :",timeit.default_timer() - start)


print('We have', parameters.shape[0],'parameters which we would like to execute')

xi = np.split(parameters,2)  # We split our parameters into 4 arrays since we have 4 GPUs available.

print('We split this into', len(xi), 'batches of', xi[0].shape[0], ',',
      xi[1].shape[0])

#%%timeit

# Timing the execution on a single GPU vs 4 GPUs, users will see a 4x performance improvement
start = timeit.default_timer()
print("The start time is :", start)

asyncresults = []

for i in range(len(xi)):
    for j in range(xi[i].shape[0]):
        asyncresults.append(
            cudaq.observe_async(kernel, h, xi[i][j, :], qpu_id=i))

print("The difference of time is :",timeit.default_timer() - start)

