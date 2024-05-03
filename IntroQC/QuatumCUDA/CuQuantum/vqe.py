import cudaq

geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]

molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
    geometry, 'sto-3g', 1, 0)

electron_count = data.n_electrons
qubit_count = 2 * data.n_orbitals

kernel, angles = cudaq.make_kernel(list)
qubits = kernel.qalloc(qubit_count)

# Prepare the Hartree Fock State.
kernel.x(qubits[0])
kernel.x(qubits[1])

# Adds parameterized gates based on the UCCSD ansatz.
cudaq.kernels.uccsd(kernel, qubits, angles, electron_count, qubit_count)

parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,
                                                     qubit_count)

optimizer = cudaq.optimizers.COBYLA()

energy, parameters = cudaq.vqe(kernel,
                               molecule,
                               optimizer,
                               parameter_count=parameter_count)

print(energy)


