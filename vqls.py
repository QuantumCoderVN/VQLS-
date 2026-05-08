"""
VQLS on Qiskit - Quantum Variational Linear Solver.

This implementation solves the linear system Ax = b using a variational quantum approach.

The quantum circuit applies a sequence of quantum operations, including variational ansatz, controlled operations, and a Hadamard test. The objective is to recover the solution of a linear system by minimizing the cost function.

The problem is formulated as:
    min ||Ax - b||, where A is a Pauli decomposition matrix, x is the quantum state, and b is the target vector.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import matplotlib.pyplot as plt


# ============================================================
# Hyperparameters
# ============================================================
N_QUBITS = 3
TOT_QUBITS = N_QUBITS + 1
ANCILLA_IDX = N_QUBITS  # Ancilla qubit is the last qubit
N_SHOTS = 10**6
STEPS = 30
ETA = 0.8
Q_DELTA = 0.001
RNG_SEED = 0

np.random.seed(RNG_SEED)


# ============================================================
# MODULE 1: Pauli decomposition and circuit components
# ============================================================
C = np.array([1.0, 0.2, 0.2])  # Coefficients for Pauli matrices

# Pauli operators in PennyLane convention:
# A_0 = I⊗I⊗I, A_1 = X⊗Z⊗I, A_2 = X⊗I⊗I
# In Qiskit, the qubit ordering is reversed: "III", "IZX", "IIX"
A_PAULI = SparsePauliOp.from_list([
    ("III", C[0]),
    ("IZX", C[1]),
    ("IIX", C[2]),
])
NUM_PAULI = len(C)


def apply_U_b(qc, qubits):
    """Apply U_b = H⊗H⊗H to the qubits."""
    for q in qubits:
        qc.h(q)


def apply_CA(qc, l, qubits, ancilla):
    """
    Apply the controlled-A_l operator to the qubits with the ancilla qubit as control.

    Pauli matrices:
        A_0 = I⊗I⊗I, A_1 = X⊗Z⊗I, A_2 = X⊗I⊗I
    """
    if l == 0:
        pass
    elif l == 1:
        qc.cx(ancilla, qubits[0])  # X on qubit 0
        qc.cz(ancilla, qubits[1])  # Z on qubit 1
    elif l == 2:
        qc.cx(ancilla, qubits[0])  # X on qubit 0


# ============================================================
# MODULE 2: Variational ansatz
# ============================================================
def apply_variational(qc, params, qubits):
    """Apply the variational ansatz V(α) = (R_Y(α_i)) · H to each qubit."""
    for i, q in enumerate(qubits):
        qc.h(q)  # Apply Hadamard gate on each qubit
    for i, q in enumerate(qubits):
        qc.ry(params[i], q)  # Apply R_Y rotation with parameter α_i


# ============================================================
# MODULE 3: Hadamard test
# ============================================================
def hadamard_test(weights, l, lp, j, part):
    """
    Measure Re/Im of <0|V† A_l† U_b Z_j U_b† A_lp V|0>

    Quantum circuit:
        ancilla: |0> -H- [S†] -•- ... -•- ... -•- -H- <Z>
                              |        |        |
        qubits:  |0> -V- ----A_l-- -U_b†- -CZ_j- -U_b- -A_lp-

    Parameters:
        - weights: variational parameters
        - l, lp: Pauli decomposition indices for the operators A_l and A_lp
        - j: Index for controlled-Z operation
        - part: Either "Re" or "Im" for real or imaginary part of the expectation value
    """
    qc = QuantumCircuit(TOT_QUBITS)
    main_qubits = list(range(N_QUBITS))

    # Apply Hadamard on the ancilla qubit
    qc.h(ANCILLA_IDX)

    # Apply phase shift for imaginary part (if part == "Im")
    if part == "Im":
        qc.p(-np.pi / 2, ANCILLA_IDX)

    # Apply the variational ansatz
    apply_variational(qc, weights, main_qubits)

    # Apply controlled-A_l
    apply_CA(qc, l, main_qubits, ANCILLA_IDX)

    # Apply U_b† (Hermitian of U_b)
    apply_U_b(qc, main_qubits)

    # Apply controlled-Z_j
    if j != -1:
        qc.cz(ANCILLA_IDX, main_qubits[j])

    # Apply U_b
    apply_U_b(qc, main_qubits)

    # Apply controlled-A_lp
    apply_CA(qc, lp, main_qubits, ANCILLA_IDX)

    # Final Hadamard on ancilla
    qc.h(ANCILLA_IDX)

    return qc


def measure_z_ancilla(weights, l, lp, j, part):
    """Measure the expectation value of Z on the ancilla qubit."""
    qc = hadamard_test(weights, l, lp, j, part)
    state = Statevector.from_instruction(qc)

    # Z on the ancilla qubit
    pauli_str = "Z" + "I" * N_QUBITS
    Z_obs = SparsePauliOp.from_list([(pauli_str, 1.0)])

    return state.expectation_value(Z_obs).real


def mu(weights, l, lp, j):
    """Compute μ = Re + i·Im."""
    re = measure_z_ancilla(weights, l, lp, j, "Re")
    im = measure_z_ancilla(weights, l, lp, j, "Im")
    return re + 1j * im


# ============================================================
# MODULE 4: Cost function
# ============================================================
def psi_norm(weights):
    """Compute <ψ|ψ> = <x|A†A|x> (norm of the quantum state)."""
    norm = 0.0 + 0.0j
    for l in range(NUM_PAULI):
        for lp in range(NUM_PAULI):
            norm += np.conj(C[l]) * C[lp] * mu(weights, l, lp, -1)
    return norm.real


def cost_local(weights):
    """Compute the local cost function C_L."""
    mu_sum = 0.0 + 0.0j
    for l in range(NUM_PAULI):
        for lp in range(NUM_PAULI):
            for j in range(N_QUBITS):
                mu_sum += np.conj(C[l]) * C[lp] * mu(weights, l, lp, j)

    return 0.5 - 0.5 * mu_sum.real / (N_QUBITS * psi_norm(weights))


# ============================================================
# MODULE 5: Optimization
# ============================================================
def parameter_shift_gradient(w, cost_fn):
    """Compute the gradient using the parameter-shift rule."""
    grad = np.zeros_like(w)
    shift = np.pi / 2
    for i in range(len(w)):
        wp = w.copy()
        wp[i] += shift
        wm = w.copy()
        wm[i] -= shift
        grad[i] = 0.5 * (cost_fn(wp) - cost_fn(wm))
    return grad


def main():
    """Main optimization loop."""
    w_init = Q_DELTA * np.random.randn(N_QUBITS)
    print(f"Initial parameters: {w_init}")
    print(f"\nPauli decomposition of A:\n{A_PAULI}\n")

    cost_history = []

    def cost_with_log(w):
        c = cost_local(w)
        cost_history.append(c)
        return c

    print("=" * 60)
    print("Optimization using SciPy COBYLA (gradient-free)")
    print("=" * 60)

    # Minimize the cost function using COBYLA optimizer
    from scipy.optimize import minimize
    res = minimize(
        cost_with_log,
        w_init,
        method="COBYLA",
        options={"rhobeg": 0.5, "maxiter": STEPS, "catol": 1e-8},
    )

    w = res.x
    print(f"\n[SciPy] Iterations: {len(cost_history)}, Final cost: {res.fun:.2e}")

    log_indices = [0, 5, 10, 20, len(cost_history) // 2, len(cost_history) - 1]
    for i in log_indices:
        if i < len(cost_history):
            print(f"  Step {i:3d}: Cost = {cost_history[i]:.7f}")

    print(f"\nOptimized parameters: {w}")
    print(f"Final cost: {res.fun:.2e}")

    # === Classical comparison ===
    print("\n" + "=" * 60)
    print("Comparison with classical solution")
    print("=" * 60)

    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    A_matrix = (
        C[0] * np.kron(np.kron(I, I), I)
        + C[1] * np.kron(np.kron(X, Z), I)
        + C[2] * np.kron(np.kron(X, I), I)
    )

    b_vector = np.ones(2**N_QUBITS) / np.sqrt(2**N_QUBITS)
    x_classical = np.linalg.solve(A_matrix, b_vector)
    x_norm = x_classical / np.linalg.norm(x_classical)
    c_probs = np.abs(x_norm) ** 2

    qc_state = QuantumCircuit(N_QUBITS)
    apply_variational(qc_state, w, list(range(N_QUBITS)))
    state = Statevector.from_instruction(qc_state)

    # Reverse qubit order to match PennyLane
    state_data = state.data
    perm = np.array(
        [int(format(i, f"0{N_QUBITS}b")[::-1], 2) for i in range(2**N_QUBITS)]
    )
    state_pennylane = state_data[perm]

    # ============================================================
    # PART B: Solution Recovery via Scaling Factor k
    # ============================================================
    x_prime_normalized = state_pennylane
    b_prime = A_matrix @ x_prime_normalized
    k_coeff = np.vdot(b_prime, b_vector) / np.vdot(b_prime, b_prime)
    x_vqls_recovered = k_coeff * x_prime_normalized

    print(f"\nb' = A · x'_VQLS     = {np.real_if_close(b_prime)}")
    print(f"b  (input)           = {np.real_if_close(b_vector)}")

    print(f"\nRecovery coefficient k = {k_coeff}")
    print(f"  |k|    = {np.abs(k_coeff):.8f}")
    print(f"  arg(k) = {np.angle(k_coeff):.8f} rad")

    print(f"\nRecovered VQLS solution x_VQLS = k·x' =")
    print(np.real_if_close(x_vqls_recovered))

    print(f"\nClassical solution x_classical =")
    print(np.real_if_close(x_classical))

    abs_err = np.linalg.norm(x_vqls_recovered - x_classical)
    rel_err = abs_err / np.linalg.norm(x_classical)

    print(f"\nAbsolute error ||x_VQLS - x_classical|| = {abs_err:.6e}")
    print(f"Relative error                         = {rel_err:.6e}")

    residual = np.linalg.norm(A_matrix @ x_vqls_recovered - b_vector)

    print(f"\nResidual check ||A·x_VQLS - b|| = {residual:.6e}")
    print(f"  A·x_VQLS = {np.real_if_close(A_matrix @ x_vqls_recovered)}")
    print(f"  b        = {np.real_if_close(b_vector)}")

    print("\n" + "=" * 70)
    print("SOLUTION RECOVERY COMPLETE")
    print("=" * 70)

    q_probs = np.abs(state_pennylane) ** 2

    print(f"\n{'Index':<8}{'Classical':<15}{'Quantum':<15}{'|Diff|':<12}")
    for i in range(2**N_QUBITS):
        print(
            f"{i:<8}{c_probs[i]:<15.6f}{q_probs[i]:<15.6f}{abs(c_probs[i] - q_probs[i]):<12.6f}"
        )

    fidelity = np.sum(np.sqrt(c_probs * q_probs)) ** 2
    print(f"\nFidelity: {fidelity:.6f}")

    # === Plotting ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(cost_history, "g-o", linewidth=2, markersize=4)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cost C_L")
    axes[0].set_yscale("log")
    axes[0].set_title("Cost convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(2**N_QUBITS), c_probs, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Basis state")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Classical solution")

    axes[2].bar(range(2**N_QUBITS), q_probs, color="seagreen", edgecolor="black")
    axes[2].set_xlabel("Basis state")
    axes[2].set_ylabel("Probability")
    axes[2].set_title("VQLS solution")

    plt.tight_layout()
    plt.savefig("vqls_results.png", dpi=120, bbox_inches="tight")
    print("\nPlot saved to: vqls_results.png")

    return w, cost_history, c_probs, q_probs


if __name__ == "__main__":
    main()