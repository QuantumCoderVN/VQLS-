
# VQLS on Qiskit

This repository provides an implementation of the **Variational Quantum Linear Solver (VQLS)** using Qiskit, which solves the linear system \( Ax = b \) using quantum circuits. The solution is obtained by minimizing the cost function that quantifies the residual between the quantum and classical solutions.

## Problem Overview

VQLS aims to solve the linear system \( Ax = b \) by preparing a quantum state \( |xangle \) that approximates the solution vector \( x \). This is done by applying a variational quantum algorithm, where a quantum circuit is used to iteratively optimize the parameters of a quantum ansatz that represents the solution. 

The solution is recovered by performing a **Hadamard test** with controlled Pauli operations and measuring the ancilla qubit for the expectation values of Pauli operators.

## Requirements

To run the code, you need to have Python and the following libraries installed:

- **Qiskit**: Quantum computing framework
- **NumPy**: For numerical operations
- **Matplotlib**: For plotting results
- **SciPy**: For optimization

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Key Dependencies:

- `qiskit`
- `numpy`
- `matplotlib`
- `scipy`

## Running the Code

To execute the script and perform the VQLS optimization, run the following command:

```bash
python vqls.py
```

The script will optimize the variational parameters using the **COBYLA** optimizer and output the results of the quantum and classical solutions.

## Output

The program will display:

1. **Optimized Parameters**: The variational parameters after the optimization procedure.
2. **Classical Solution**: The solution to the linear system obtained by solving \( Ax = b \) using classical methods (using `numpy.linalg.solve`).
3. **Quantum Solution**: The quantum solution, derived from the variational quantum algorithm.
4. **Fidelity**: The fidelity between the classical and quantum solutions.
5. **Cost Convergence**: A plot of the cost function during the optimization process, showing how the optimization converges to a solution.
6. **Comparison Plot**: A comparison of classical and quantum solution probabilities across all basis states.

### Example Output:

- **Optimized parameters**: `[0.1234, 0.5678, 0.9101]`
- **Classical solution**: `[0.123, 0.567, 0.910]`
- **Quantum solution**: `[0.124, 0.568, 0.911]`
- **Fidelity**: `0.9998`

## File Structure

The repository contains the following files:

- `vqls.py`: The main Python script implementing the VQLS algorithm.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file, providing an overview of the project.

## Quantum Circuit Description

The VQLS algorithm involves several key steps:

1. **Hadamard Test**: The circuit measures the expectation value of a Pauli operator on the ancilla qubit.
2. **Controlled Pauli Operations**: Controlled operations such as controlled-X (CX), controlled-Z (CZ), and others are used to implement the required decomposition of the operator \( A \).
3. **Variational Ansatz**: A variational quantum circuit \( V(lpha) \) is used to encode the solution, and the parameters are optimized using a classical optimizer (COBYLA in this case).
4. **State Measurement**: The quantum state is measured, and the real and imaginary parts of the expectation values are computed to update the parameters.

## Quantum vs Classical Solution

The quantum solution is compared against the classical solution obtained by solving the linear system directly. The comparison is made by calculating the absolute and relative error, along with a fidelity score that measures the closeness of the quantum and classical solutions.

## Paper Reference

For more details on the theoretical foundation and implementation of VQLS, refer to the original research paper on **Variational Quantum Linear Solver**.

## Future Improvements

- **Error Mitigation**: Implementing error mitigation techniques to handle noisy quantum circuits.
- **Optimized Variational Ansätze**: Experimenting with different ansatz structures to improve performance.
- **Scalability**: Extending the code to work with larger systems and more qubits.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
