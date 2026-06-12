"""
Qiskit-based Grover search core.

Conventions
-----------
The user-facing target string is written in circuit order from top to bottom:
    target='101' means q_0=1, q_1=0, q_2=1.

Internally, Qiskit stores statevector amplitudes in little-endian qubit order.
The helper functions in this module convert amplitudes back to the user-facing
(top-to-bottom) basis labels, so plots and printed tables match the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
except Exception as exc:  # pragma: no cover - triggered only when Qiskit is absent
    QuantumCircuit = None  # type: ignore
    Statevector = None  # type: ignore
    transpile = None  # type: ignore
    _QISKIT_IMPORT_ERROR = exc
else:
    _QISKIT_IMPORT_ERROR = None


@dataclass
class GroverStep:
    """One saved stage in the Grover circuit evolution."""

    index: int
    title: str
    kind: str
    iteration: int
    circuit: "QuantumCircuit"
    statevector: "Statevector"


def require_qiskit() -> None:
    """Raise a helpful error if Qiskit is not installed."""
    if _QISKIT_IMPORT_ERROR is not None:
        raise ImportError(
            "Qiskit is required for this refactored project. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from _QISKIT_IMPORT_ERROR


def validate_problem(n: int, target: str, iterations: int) -> None:
    if n not in (2, 3):
        raise ValueError("This teaching demo supports n=2 or n=3 qubits.")
    if len(target) != n or any(bit not in "01" for bit in target):
        raise ValueError(f"target must be a {n}-bit string, for example {'1' * n}.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")


def basis_labels(n: int) -> List[str]:
    """Return basis labels in top-to-bottom circuit order: |q0 q1 ... q(n-1)>.

    For n=3 this returns |000>, |001>, ..., |111> in ordinary binary order.
    """
    return [f"|{i:0{n}b}>" for i in range(2**n)]


def top_bits_to_qiskit_index(bits: str) -> int:
    """Convert top-to-bottom bits q0...q(n-1) to Qiskit's statevector index."""
    return sum(int(bit) << qubit for qubit, bit in enumerate(bits))


def statevector_in_top_order(statevector: "Statevector", n: int) -> np.ndarray:
    """Return amplitudes ordered as |q0 q1 ... q(n-1)> labels."""
    data = np.asarray(statevector.data, dtype=complex)
    ordered = []
    for label in basis_labels(n):
        bits = label.strip("|>")
        ordered.append(data[top_bits_to_qiskit_index(bits)])
    return np.asarray(ordered, dtype=complex)


def probabilities_in_top_order(statevector: "Statevector", n: int) -> np.ndarray:
    amps = statevector_in_top_order(statevector, n)
    return np.abs(amps) ** 2


def target_probability(statevector: "Statevector", target: str) -> float:
    index = top_bits_to_qiskit_index(target)
    return float(abs(statevector.data[index]) ** 2)


def _apply_multi_controlled_z(qc: "QuantumCircuit", qubits: List[int]) -> None:
    """Apply a phase flip to |11...1> on the selected qubits."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return

    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def apply_oracle(qc: "QuantumCircuit", target: str, add_barrier: bool = True) -> None:
    """Apply a phase oracle that maps |target> to -|target>.

    The target string is in top-to-bottom circuit order. For target='101', the
    oracle flips the phase of the basis state q0=1, q1=0, q2=1.
    """
    n = len(target)

    # Convert the selected target state into |11...1>.
    for qubit, bit in enumerate(target):
        if bit == "0":
            qc.x(qubit)

    _apply_multi_controlled_z(qc, list(range(n)))

    # Undo the temporary basis mapping.
    for qubit, bit in enumerate(target):
        if bit == "0":
            qc.x(qubit)

    if add_barrier:
        qc.barrier()


def apply_diffusion(qc: "QuantumCircuit", n: int, add_barrier: bool = True) -> None:
    """Apply Grover diffusion D = 2|s><s| - I.

    The usual H-X-MCZ-X-H gate sequence implements I - 2|s><s|, which differs
    from D only by a global phase. We add a global phase of pi so the saved
    statevectors match the textbook amplitude formula a_i' = 2*mean(a) - a_i.
    """
    for qubit in range(n):
        qc.h(qubit)
    for qubit in range(n):
        qc.x(qubit)

    _apply_multi_controlled_z(qc, list(range(n)))

    for qubit in range(n):
        qc.x(qubit)
    for qubit in range(n):
        qc.h(qubit)

    qc.global_phase += np.pi

    if add_barrier:
        qc.barrier()


def build_grover_circuit(
    n: int,
    target: str,
    iterations: int,
    measure: bool = False,
    add_barriers: bool = True,
) -> "QuantumCircuit":
    """Build a complete Grover circuit."""
    require_qiskit()
    validate_problem(n, target, iterations)

    if measure:
        qc = QuantumCircuit(n, n)
    else:
        qc = QuantumCircuit(n)

    qc.h(range(n))
    if add_barriers:
        qc.barrier()

    for _ in range(iterations):
        apply_oracle(qc, target, add_barrier=add_barriers)
        apply_diffusion(qc, n, add_barrier=add_barriers)

    if measure:
        # Qiskit displays count bitstrings as c[n-1]...c[0]. Measuring q_i into
        # c[n-1-i] makes the printed key match q0...q(n-1), i.e. target order.
        for qubit in range(n):
            qc.measure(qubit, n - 1 - qubit)

    return qc


def build_grover_history(n: int, target: str, iterations: int) -> List[GroverStep]:
    """Build the circuit step by step and save statevectors after each stage."""
    require_qiskit()
    validate_problem(n, target, iterations)

    qc = QuantumCircuit(n)
    history: List[GroverStep] = []

    qc.h(range(n))
    qc.barrier()
    history.append(
        GroverStep(
            index=0,
            title="Step 0: Uniform superposition after Hadamard gates",
            kind="initial",
            iteration=0,
            circuit=qc.copy(),
            statevector=Statevector.from_instruction(qc),
        )
    )

    step_index = 1
    for iteration in range(1, iterations + 1):
        apply_oracle(qc, target, add_barrier=True)
        history.append(
            GroverStep(
                index=step_index,
                title=f"Step {step_index}: Iteration {iteration} - Oracle phase flip",
                kind="oracle",
                iteration=iteration,
                circuit=qc.copy(),
                statevector=Statevector.from_instruction(qc),
            )
        )
        step_index += 1

        apply_diffusion(qc, n, add_barrier=True)
        history.append(
            GroverStep(
                index=step_index,
                title=f"Step {step_index}: Iteration {iteration} - Diffusion inversion about average",
                kind="diffusion",
                iteration=iteration,
                circuit=qc.copy(),
                statevector=Statevector.from_instruction(qc),
            )
        )
        step_index += 1

    return history


def target_probability_by_iteration(history: List[GroverStep], target: str) -> List[Tuple[int, float]]:
    """Return target probabilities after the initial state and each full iteration."""
    points: List[Tuple[int, float]] = []
    for step in history:
        if step.kind == "initial" or step.kind == "diffusion":
            points.append((step.iteration, target_probability(step.statevector, target)))
    return points


def simulate_measurements(
    n: int,
    target: str,
    iterations: int,
    shots: int = 2048,
) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """Run the measured Grover circuit on Qiskit Aer, if available.

    Returns (counts, error_message). If Aer is unavailable, counts is None and
    error_message describes the issue. This keeps the project usable for circuit
    drawing and statevector analysis even before Aer is installed.
    """
    require_qiskit()
    qc = build_grover_circuit(n, target, iterations, measure=True, add_barriers=False)

    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:  # pragma: no cover
        return None, f"qiskit-aer is not available: {exc}"

    try:
        simulator = AerSimulator()
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=shots).result()
        return dict(result.get_counts()), None
    except Exception as exc:  # pragma: no cover
        return None, f"Aer simulation failed: {exc}"
