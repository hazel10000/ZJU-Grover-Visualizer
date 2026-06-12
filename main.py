"""Command-line entry point for the Qiskit Grover visualizer."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from qiskit_grover_core import (
    basis_labels,
    build_grover_circuit,
    build_grover_history,
    probabilities_in_top_order,
    simulate_measurements,
    statevector_in_top_order,
    target_probability,
    target_probability_by_iteration,
)
from visualization import (
    save_plotly_3d_probability_history,
    plot_amplitudes,
    plot_counts,
    plot_probabilities,
    plot_target_probability_curve,
)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def save_step_csv(history, n: int, target: str, output_dir: Path) -> Path:
    labels = basis_labels(n)
    path = output_dir / "state_history.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "iteration", "kind", "basis", "real_amplitude", "imag_amplitude", "probability", "is_target"])
        for step in history:
            amps = statevector_in_top_order(step.statevector, n)
            probs = probabilities_in_top_order(step.statevector, n)
            for label, amp, prob in zip(labels, amps, probs):
                writer.writerow([
                    step.index,
                    step.iteration,
                    step.kind,
                    label,
                    f"{amp.real:.10f}",
                    f"{amp.imag:.10f}",
                    f"{prob:.10f}",
                    int(label == f"|{target}>"),
                ])
    return path


def save_probability_summary(points, output_dir: Path) -> Path:
    path = output_dir / "target_probability_by_iteration.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["completed_iterations", "target_probability"])
        for iteration, prob in points:
            writer.writerow([iteration, f"{prob:.10f}"])
    return path


def draw_circuit_png(n: int, target: str, iterations: int, output_dir: Path) -> Path | None:
    qc = build_grover_circuit(n, target, iterations, measure=True, add_barriers=True)
    path = output_dir / f"grover_n{n}_target{target}_r{iterations}_circuit.png"
    try:
        fig = qc.draw(output="mpl", fold=-1, idle_wires=False)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        return path
    except Exception as exc:
        txt_path = output_dir / f"grover_n{n}_target{target}_r{iterations}_circuit.txt"
        txt_path.write_text(str(qc.draw(output="text", fold=-1)) + f"\n\n[PNG drawing failed: {exc}]\n", encoding="utf-8")
        print(f"[warning] Circuit PNG drawing failed. Text circuit saved to {txt_path}")
        return None


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = build_grover_history(args.n, args.target, args.iterations)
    points = target_probability_by_iteration(history, args.target)

    print("=== Grover Qiskit Visualizer ===")
    print(f"n={args.n}, target=|{args.target}>, iterations={args.iterations}")
    print("\nTarget probability after each completed iteration:")
    for iteration, prob in points:
        print(f"  iteration {iteration}: {prob:.6f}")

    for step in history:
        stem = f"{step.index:02d}_{slugify(step.kind)}_iter{step.iteration}"
        amp_path = figures_dir / f"{stem}_amplitude.png"
        prob_path = figures_dir / f"{stem}_probability.png"
        plot_amplitudes(step, args.n, args.target, amp_path)
        plot_probabilities(step, args.n, args.target, prob_path)

        amps = statevector_in_top_order(step.statevector, args.n)
        probs = probabilities_in_top_order(step.statevector, args.n)
        print(f"\n{step.title}")
        for label, amp, prob in zip(basis_labels(args.n), amps, probs):
            marker = " <-- target" if label == f"|{args.target}>" else ""
            print(f"  {label}: amp={amp.real:+.6f}{amp.imag:+.6f}j, P={prob:.6f}{marker}")

    plot_target_probability_curve(points, figures_dir / "target_probability_curve.png")
    save_plotly_3d_probability_history(history, args.n, args.target, figures_dir / "probability_history_3d.html")
    save_step_csv(history, args.n, args.target, output_dir)
    save_probability_summary(points, output_dir)

    circuit_path = draw_circuit_png(args.n, args.target, args.iterations, figures_dir)
    if circuit_path is not None:
        print(f"\nCircuit figure saved to: {circuit_path}")

    if not args.no_sim:
        counts, error = simulate_measurements(args.n, args.target, args.iterations, shots=args.shots)
        if counts is None:
            print(f"\n[warning] Measurement simulation skipped: {error}")
        else:
            print(f"\nMeasurement counts with {args.shots} shots:")
            for key, value in sorted(counts.items()):
                print(f"  {key}: {value}")
            plot_counts(counts, args.target, figures_dir / "measurement_counts.png")

    print(f"\nAll outputs saved under: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qiskit-based Grover amplitude amplification visualizer")
    parser.add_argument("--n", type=int, default=3, choices=[2, 3], help="number of qubits")
    parser.add_argument("--target", type=str, default="101", help="target bitstring in circuit top-to-bottom order")
    parser.add_argument("--iterations", type=int, default=2, help="number of Grover iterations")
    parser.add_argument("--shots", type=int, default=2048, help="shots for Aer measurement simulation")
    parser.add_argument("--output", type=str, default="outputs", help="output directory")
    parser.add_argument("--no-sim", action="store_true", help="skip qiskit-aer measurement simulation")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
