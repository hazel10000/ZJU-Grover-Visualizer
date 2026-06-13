"""Visualization helpers for the Qiskit Grover visualizer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qiskit_grover_core import (
    GroverStep,
    basis_labels,
    probabilities_in_top_order,
    statevector_in_top_order,
)


def _target_index(labels: Sequence[str], target: str) -> int:
    return labels.index(f"|{target}>")


def _save_or_return(fig, output_path: str | Path | None):
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig


def plot_amplitudes(step: GroverStep, n: int, target: str, output_path: str | Path | None = None):
    labels = basis_labels(n)
    amps = statevector_in_top_order(step.statevector, n)
    real_amps = np.real_if_close(amps, tol=1000).real
    target_idx = _target_index(labels, target)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, real_amps)
    bars[target_idx].set_hatch("//")
    bars[target_idx].set_linewidth(1.6)

    ax.axhline(0, linewidth=0.8)
    ax.set_ylim(min(-1.05, float(np.min(real_amps)) - 0.1), max(1.05, float(np.max(real_amps)) + 0.1))
    ax.set_xlabel("Basis state")
    ax.set_ylabel("Real amplitude")
    ax.set_title(step.title + f"\nTarget state: |{target}>")
    return _save_or_return(fig, output_path)


def plot_probabilities(step: GroverStep, n: int, target: str, output_path: str | Path | None = None):
    labels = basis_labels(n)
    probs = probabilities_in_top_order(step.statevector, n)
    target_idx = _target_index(labels, target)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, probs)
    bars[target_idx].set_hatch("//")
    bars[target_idx].set_linewidth(1.6)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Basis state")
    ax.set_ylabel("Probability")
    ax.set_title(step.title + f"\nTarget state: |{target}>")
    return _save_or_return(fig, output_path)


def plot_target_probability_curve(points: Sequence[Tuple[int, float]], output_path: str | Path | None = None):
    iterations = [p[0] for p in points]
    probabilities = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, probabilities, marker="o")
    ax.set_xticks(iterations)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Completed Grover iterations")
    ax.set_ylabel("Target-state probability")
    ax.set_title("Target probability after each full Grover iteration")
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, output_path)


def plot_counts(counts: Dict[str, int], target: str, output_path: str | Path | None = None):
    labels = sorted(counts.keys())
    values = [counts[label] for label in labels]
    target_idx = labels.index(target) if target in labels else None

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values)
    if target_idx is not None:
        bars[target_idx].set_hatch("//")
        bars[target_idx].set_linewidth(1.6)

    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Counts")
    ax.set_title(f"Measurement counts, target={target}")
    return _save_or_return(fig, output_path)


def plotly_3d_probability_history(history: Sequence[GroverStep], n: int, target: str) -> "go.Figure":
    """Create an interactive Plotly 3D step-by-step probability history.

    Plotly does not provide a native 3D bar chart primitive, so this function
    draws vertical probability columns with Scatter3d line segments and markers.
    The result is lighter and more interactive than the previous Matplotlib 3D
    chart: users can rotate, zoom, and inspect hover labels in Streamlit.
    """
    labels = basis_labels(n)
    target_label = f"|{target}>"
    target_idx = labels.index(target_label)
    step_labels = [f"S{step.index}: {step.kind}, iter {step.iteration}" for step in history]

    non_target_x: List[float | None] = []
    non_target_y: List[float | None] = []
    non_target_z: List[float | None] = []
    non_target_hover: List[str | None] = []

    target_x: List[float | None] = []
    target_y: List[float | None] = []
    target_z: List[float | None] = []
    target_hover: List[str | None] = []

    marker_x: List[float] = []
    marker_y: List[float] = []
    marker_z: List[float] = []
    marker_hover: List[str] = []

    target_marker_x: List[float] = []
    target_marker_y: List[float] = []
    target_marker_z: List[float] = []
    target_marker_hover: List[str] = []

    for y, step in enumerate(history):
        probs = probabilities_in_top_order(step.statevector, n)
        for x, (label, prob) in enumerate(zip(labels, probs)):
            hover = (
                f"Step: {step_labels[y]}<br>"
                f"Basis: {label}<br>"
                f"Probability: {float(prob):.6f}"
            )
            if x == target_idx:
                target_x.extend([x, x, None])
                target_y.extend([y, y, None])
                target_z.extend([0.0, float(prob), None])
                target_hover.extend([hover, hover, None])
                target_marker_x.append(x)
                target_marker_y.append(y)
                target_marker_z.append(float(prob))
                target_marker_hover.append(hover)
            else:
                non_target_x.extend([x, x, None])
                non_target_y.extend([y, y, None])
                non_target_z.extend([0.0, float(prob), None])
                non_target_hover.extend([hover, hover, None])
                marker_x.append(x)
                marker_y.append(y)
                marker_z.append(float(prob))
                marker_hover.append(hover)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=non_target_x,
            y=non_target_y,
            z=non_target_z,
            mode="lines",
            name="Non-target states",
            line={"width": 7, "color": "#4c78a8"},
            text=non_target_hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=target_x,
            y=target_y,
            z=target_z,
            mode="lines",
            name=f"Target {target_label}",
            line={"width": 11, "color": "#f58518"},
            text=target_hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=marker_x,
            y=marker_y,
            z=marker_z,
            mode="markers",
            name="Probability values",
            marker={"size": 4, "color": "#4c78a8"},
            text=marker_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=target_marker_x,
            y=target_marker_y,
            z=target_marker_z,
            mode="markers",
            name="Target probability values",
            marker={"size": 6, "symbol": "diamond", "color": "#f58518"},
            text=target_marker_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": f"Interactive 3D probability history, target={target_label}",
            "x": 0.02,
            "xanchor": "left",
            "y": 0.965,
            "yanchor": "top",
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 28},
        height=720,
        autosize=True,
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.02, 0.96]},
            "aspectmode": "cube",
            "xaxis": {
                "title": "Basis state",
                "tickmode": "array",
                "tickvals": list(range(len(labels))),
                "ticktext": labels,
            },
            "yaxis": {
                "title": "Evolution step",
                "tickmode": "array",
                "tickvals": list(range(len(history))),
                "ticktext": [f"S{step.index}" for step in history],
            },
            "zaxis": {"title": "Probability", "range": [0, 1.0]},
            "camera": {"eye": {"x": 1.55, "y": 1.75, "z": 1.15}},
        },
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.93,
            "xanchor": "left",
            "x": 0.02,
            "bgcolor": "rgba(255,255,255,0.9)",
            "borderwidth": 1,
        },
    )
    return fig


def save_plotly_3d_probability_history(
    history: Sequence[GroverStep],
    n: int,
    target: str,
    output_path: str | Path,
) -> Path:
    """Save the interactive Plotly 3D history as a standalone HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plotly_3d_probability_history(history, n, target)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def _highlight_target_marker(labels: Sequence[str], target: str) -> List[str]:
    target_label = f"|{target}>"
    return ["Target state" if label == target_label else "Other state" for label in labels]


def plotly_amplitudes(step: GroverStep, n: int, target: str) -> "go.Figure":
    """Interactive Plotly bar chart for real probability amplitudes."""
    labels = basis_labels(n)
    amps = statevector_in_top_order(step.statevector, n)
    real_amps = np.real_if_close(amps, tol=1000).real.astype(float)
    probs = np.abs(amps) ** 2
    target_label = f"|{target}>"
    categories = _highlight_target_marker(labels, target)
    colors = ["#f58518" if category == "Target state" else "#4c78a8" for category in categories]
    hover = [
        f"Basis: {label}<br>Real amplitude: {amp:.6f}<br>Imag amplitude: {complex_amp.imag:.6f}<br>Probability: {prob:.6f}"
        for label, amp, complex_amp, prob in zip(labels, real_amps, amps, probs)
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=real_amps,
            name="Basis states",
            text=[f"{y:.3f}" for y in real_amps],
            textposition="outside",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker={"color": colors},
        )
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Other state", marker={"size": 10, "color": "#4c78a8"}))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Target state", marker={"size": 10, "color": "#f58518"}))
    ymin = min(-1.05, float(np.min(real_amps)) - 0.1)
    ymax = max(1.05, float(np.max(real_amps)) + 0.1)
    fig.update_layout(
        title=f"Real amplitudes - {step.title}, target={target_label}",
        xaxis_title="Basis state",
        yaxis_title="Real amplitude",
        yaxis_range=[ymin, ymax],
        height=440,
        bargap=0.25,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    fig.add_hline(y=0, line_width=1)
    return fig


def plotly_probabilities(step: GroverStep, n: int, target: str) -> "go.Figure":
    """Interactive Plotly bar chart for measurement probabilities."""
    labels = basis_labels(n)
    probs = probabilities_in_top_order(step.statevector, n).astype(float)
    amps = statevector_in_top_order(step.statevector, n)
    target_label = f"|{target}>"
    categories = _highlight_target_marker(labels, target)
    colors = ["#f58518" if category == "Target state" else "#4c78a8" for category in categories]
    hover = [
        f"Basis: {label}<br>Probability: {prob:.6f}<br>Real amplitude: {amp.real:.6f}<br>Imag amplitude: {amp.imag:.6f}"
        for label, prob, amp in zip(labels, probs, amps)
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=probs,
            name="Basis states",
            text=[f"{100*y:.1f}%" for y in probs],
            textposition="outside",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker={"color": colors},
        )
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Other state", marker={"size": 10, "color": "#4c78a8"}))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Target state", marker={"size": 10, "color": "#f58518"}))
    fig.update_layout(
        title=f"Measurement probabilities - {step.title}, target={target_label}",
        xaxis_title="Basis state",
        yaxis_title="Probability",
        yaxis_range=[0, 1.05],
        height=440,
        bargap=0.25,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    return fig


def plotly_step_animation(
    history: Sequence[GroverStep],
    n: int,
    target: str,
    frame_duration_ms: int = 700,
) -> "go.Figure":
    """Animated side-by-side amplitude/probability view driven by Plotly frames."""
    labels = basis_labels(n)
    target_label = f"|{target}>"
    colors = ["#f58518" if label == target_label else "#4c78a8" for label in labels]

    def frame_values(step: GroverStep):
        amps = statevector_in_top_order(step.statevector, n)
        real_amps = np.real_if_close(amps, tol=1000).real.astype(float)
        probs = probabilities_in_top_order(step.statevector, n).astype(float)
        amp_hover = [
            f"Step: S{step.index} ({step.kind})<br>Basis: {label}<br>Real amplitude: {amp:.6f}<br>Probability: {prob:.6f}"
            for label, amp, prob in zip(labels, real_amps, probs)
        ]
        prob_hover = [
            f"Step: S{step.index} ({step.kind})<br>Basis: {label}<br>Probability: {prob:.6f}<br>Real amplitude: {amp:.6f}"
            for label, prob, amp in zip(labels, probs, real_amps)
        ]
        return real_amps, probs, amp_hover, prob_hover

    first_amps, first_probs, first_amp_hover, first_prob_hover = frame_values(history[0])
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("实数概率幅", "测量概率"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=first_amps,
            marker={"color": colors},
            text=[f"{value:.3f}" for value in first_amps],
            textposition="outside",
            hovertext=first_amp_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=first_probs,
            marker={"color": colors},
            text=[f"{100 * value:.1f}%" for value in first_probs],
            textposition="outside",
            hovertext=first_prob_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    frames = []
    for step in history:
        amps, probs, amp_hover, prob_hover = frame_values(step)
        frames.append(
            go.Frame(
                name=str(step.index),
                data=[
                    go.Bar(
                        y=amps,
                        text=[f"{value:.3f}" for value in amps],
                        hovertext=amp_hover,
                        marker={"color": colors},
                    ),
                    go.Bar(
                        y=probs,
                        text=[f"{100 * value:.1f}%" for value in probs],
                        hovertext=prob_hover,
                        marker={"color": colors},
                    ),
                ],
                layout=go.Layout(
                    title_text=f"Grover 步进动画 - S{step.index}: {step.kind}, iteration {step.iteration}, target={target_label}"
                ),
            )
        )
    fig.frames = frames

    slider_steps = [
        {
            "args": [
                [str(step.index)],
                {
                    "frame": {"duration": frame_duration_ms, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 180},
                },
            ],
            "label": f"S{step.index}",
            "method": "animate",
        }
        for step in history
    ]
    play_args = {
        "frame": {"duration": frame_duration_ms, "redraw": True},
        "fromcurrent": True,
        "mode": "immediate",
        "transition": {"duration": 180},
    }
    pause_args = {
        "frame": {"duration": 0, "redraw": False},
        "mode": "immediate",
        "transition": {"duration": 0},
    }

    fig.update_layout(
        title=f"Grover 步进动画 - S0: initial, iteration 0, target={target_label}",
        height=540,
        margin={"l": 24, "r": 24, "t": 76, "b": 118},
        bargap=0.25,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": -0.16,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"r": 10, "t": 0, "b": 0},
                "showactive": False,
                "buttons": [
                    {"label": "播放", "method": "animate", "args": [None, play_args]},
                    {"label": "暂停", "method": "animate", "args": [[None], pause_args]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "步骤："},
                "x": 0.20,
                "len": 0.74,
                "y": -0.10,
                "pad": {"t": 0, "b": 12},
                "steps": slider_steps,
            }
        ],
    )
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    fig.update_yaxes(title_text="实数概率幅", range=[-1.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="测量概率", range=[0, 1.05], row=1, col=2)
    return fig


def plotly_target_probability_curve(points: Sequence[Tuple[int, float]]) -> "go.Figure":
    """Interactive line chart for target probability after each full iteration."""
    iterations = [int(p[0]) for p in points]
    probabilities = [float(p[1]) for p in points]
    fig = go.Figure(
        go.Scatter(
            x=iterations,
            y=probabilities,
            mode="lines+markers+text",
            text=[f"{100*p:.1f}%" for p in probabilities],
            textposition="top center",
            hovertemplate="Completed iterations: %{x}<br>Target probability: %{y:.6f}<extra></extra>",
            name="Target probability",
        )
    )
    fig.update_layout(
        title="Target-state probability after each completed Grover iteration",
        xaxis_title="Completed Grover iterations",
        yaxis_title="Target-state probability",
        yaxis_range=[0, 1.05],
        height=430,
    )
    fig.update_xaxes(tickmode="array", tickvals=iterations)
    return fig


def plotly_noise_degradation_scan(rows: Sequence[Dict[str, object]]) -> "go.Figure":
    """Plot ideal probability, noisy target frequency, and circuit depth."""
    iterations = [int(row["iterations"]) for row in rows]
    ideal = [float(row["ideal_target_probability"]) for row in rows]
    noisy = [float(row["noisy_target_frequency"]) for row in rows]
    depths = [int(row["depth"]) for row in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=ideal,
            mode="lines+markers+text",
            text=[f"{100*y:.1f}%" for y in ideal],
            textposition="top center",
            name="Ideal target probability",
            hovertemplate="Iterations: %{x}<br>Ideal probability: %{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=noisy,
            mode="lines+markers+text",
            text=[f"{100*y:.1f}%" for y in noisy],
            textposition="bottom center",
            name="Noisy measured frequency",
            hovertemplate="Iterations: %{x}<br>Noisy frequency: %{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=iterations,
            y=depths,
            name="Transpiled circuit depth",
            yaxis="y2",
            opacity=0.24,
            marker={"color": "#6f7688"},
            hovertemplate="Iterations: %{x}<br>Depth: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Noise and circuit-depth degradation scan",
        xaxis_title="Completed Grover iterations",
        yaxis={"title": "Target-state probability / frequency", "range": [0, 1.05]},
        yaxis2={"title": "Circuit depth", "overlaying": "y", "side": "right", "showgrid": False},
        height=460,
        bargap=0.35,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(tickmode="array", tickvals=iterations)
    return fig


def plotly_counts(counts: Dict[str, int], target: str, n: int | None = None) -> "go.Figure":
    """Interactive bar chart for measurement counts.

    Qiskit returns raw classical bitstrings such as "101". For consistency with
    the statevector plots, this function displays all basis states in ket form
    |q0q1...>, fills unobserved outcomes with zero counts, and uses ordinary
    binary basis order rather than alphabetical order of only observed keys.
    """
    if n is None:
        n = len(target)
    labels = basis_labels(n)
    raw_keys = [label.strip("|>") for label in labels]
    values = [int(counts.get(key, 0)) for key in raw_keys]
    total = sum(values) if values else 1
    target_label = f"|{target}>"
    categories = ["Target state" if label == target_label else "Other state" for label in labels]
    colors = ["#f58518" if category == "Target state" else "#4c78a8" for category in categories]
    hover = [
        f"Basis state: {label}<br>Raw Qiskit key: {raw}<br>Counts: {value}<br>Frequency: {value / total:.6f}"
        for label, raw, value in zip(labels, raw_keys, values)
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            name="Basis states",
            text=[str(y) if y > 0 else "0" for y in values],
            textposition="outside",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker={"color": colors},
        )
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Other state", marker={"size": 10, "color": "#4c78a8"}))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Target state", marker={"size": 10, "color": "#f58518"}))
    fig.update_layout(
        title=f"Measurement counts from Qiskit Aer, target={target_label}",
        xaxis_title="Measured basis state, displayed as |q0q1...>",
        yaxis_title="Counts",
        height=450,
        bargap=0.25,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    fig.update_yaxes(range=[0, max(values + [1]) * 1.15])
    return fig

def circuit_operation_rows(qc) -> List[Dict[str, object]]:
    """Return a user-readable operation table for a Qiskit circuit."""
    rows: List[Dict[str, object]] = []
    for idx, instruction in enumerate(qc.data):
        # Qiskit 1.x uses CircuitInstruction objects; older versions unpack as tuples.
        try:
            operation = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
        except AttributeError:  # pragma: no cover - for older qiskit only
            operation, qargs, cargs = instruction

        q_indices = [qc.find_bit(q).index for q in qargs]
        c_indices = [qc.find_bit(c).index for c in cargs]
        params = []
        for param in getattr(operation, "params", []):
            try:
                params.append(f"{float(param):.6g}")
            except Exception:
                params.append(str(param))
        rows.append(
            {
                "step": idx,
                "gate": operation.name,
                "qubits": ", ".join(f"q{q}" for q in q_indices) if q_indices else "",
                "clbits": ", ".join(f"c{c}" for c in c_indices) if c_indices else "",
                "params": ", ".join(params),
            }
        )
    return rows


def plotly_circuit_operation_view(qc) -> "go.Figure":
    """Create an interactive, Qiskit-like gate-level circuit timeline.

    This keeps the hoverable Plotly companion view while using visual cues from
    Qiskit's canonical circuit drawing: wires, gate boxes, barriers, controls,
    targets, and measurement links.
    """
    rows = circuit_operation_rows(qc)
    n_qubits = len(qc.qubits)
    n_clbits = len(qc.clbits)
    classical_bus_label = f"c[0..{n_clbits - 1}]" if n_clbits else None
    y_labels = [f"q{i}" for i in range(n_qubits)] + ([classical_bus_label] if classical_bus_label else [])
    y_values = list(range(len(y_labels)))
    y_map = {label: y for label, y in zip(y_labels, y_values)}
    if classical_bus_label is not None:
        for clbit in range(n_clbits):
            y_map[f"c{clbit}"] = y_map[classical_bus_label]

    scheduled_rows = []
    last_column_by_wire = {label: -1 for label in y_labels}
    if classical_bus_label is not None:
        for clbit in range(n_clbits):
            last_column_by_wire[f"c{clbit}"] = -1
    for row in rows:
        gate = str(row["gate"])
        qubits = [int(part.strip().replace("q", "")) for part in str(row["qubits"]).split(",") if part.strip()]
        clbits = [int(part.strip().replace("c", "")) for part in str(row["clbits"]).split(",") if part.strip()]
        touched_wires = [f"q{q}" for q in qubits] + [f"c{c}" for c in clbits]
        if gate == "barrier":
            touched_wires = [f"q{q}" for q in range(n_qubits)]
        if gate == "measure":
            column = max(last_column_by_wire.values(), default=-1) + 1
        else:
            column = max([last_column_by_wire.get(wire, -1) for wire in touched_wires] or [-1]) + 1
        for wire in touched_wires:
            if wire in last_column_by_wire:
                last_column_by_wire[wire] = column
        if gate == "measure" and classical_bus_label is not None:
            last_column_by_wire[classical_bus_label] = column
        scheduled_rows.append({**row, "column": column, "qubit_indices": qubits, "clbit_indices": clbits})

    fig = go.Figure()
    wire_color = "#303642"
    gate_line = "#48506a"
    gate_fill = "#ffffff"
    target_fill = "#ffffff"
    barrier_color = "#8b8f9d"
    annotation_font = {"size": 12, "color": "#202432"}
    gate_half_height = 0.42

    def add_gate_marker(x: float, y: float, label: str) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={
                    "size": 34,
                    "symbol": "square",
                    "color": gate_fill,
                    "line": {"color": gate_line, "width": 1.5},
                },
                text=[label],
                textposition="middle center",
                textfont=annotation_font,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def add_control_marker(x: float, y: float) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker={"size": 9, "symbol": "circle", "color": wire_color},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def add_target_marker(x: float, y: float) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={
                    "size": 25,
                    "symbol": "circle",
                    "color": target_fill,
                    "line": {"color": wire_color, "width": 1.8},
                },
                text=["+"],
                textposition="middle center",
                textfont={"size": 15, "color": wire_color},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Horizontal wires.
    x_spacing = 0.82
    max_x = (max([row["column"] for row in scheduled_rows], default=0) + 1) * x_spacing
    for label in y_labels:
        y = y_map[label]
        is_classical = label.startswith("c")
        offsets = [-0.045, 0.045] if is_classical else [0]
        for offset in offsets:
            fig.add_shape(
                type="line",
                x0=-0.35,
                x1=max_x + 0.35,
                y0=y + offset,
                y1=y + offset,
                line={"color": wire_color, "width": 1.4 if is_classical else 1.8},
                layer="below",
            )
        fig.add_annotation(
            x=-0.68,
            y=y,
            text=label,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font={"size": 12, "color": wire_color},
        )

    # Gate markers and connectors. We use a scaled x coordinate so exported
    # HTML can be displayed on a wide, horizontally scrollable canvas.
    for row in scheduled_rows:
        gate = str(row["gate"])
        x = int(row["column"]) * x_spacing
        qubits = row["qubit_indices"]
        clbits = row["clbit_indices"]
        qys = [y_map[f"q{q}"] for q in qubits if f"q{q}" in y_map]
        cys = [y_map[f"c{c}"] for c in clbits if f"c{c}" in y_map]
        ys = qys + cys
        if gate == "barrier":
            if qys:
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=min(qys) - gate_half_height,
                    y1=max(qys) + gate_half_height,
                    line={"color": barrier_color, "width": 1.5, "dash": "dash"},
                )
            continue

        hover = (
            f"Operation step: {row['step']}<br>Gate: {gate}<br>Qubits: {row['qubits'] or '-'}"
            f"<br>Classical bits: {row['clbits'] or '-'}<br>Parameters: {row['params'] or '-'}"
        )

        if gate in {"cx", "ccx", "mcx"} and len(qys) >= 2:
            controls = qys[:-1]
            target_y = qys[-1]
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=min(qys),
                y1=target_y - 0.22 if target_y == max(qys) else max(qys),
                line={"color": wire_color, "width": 1.8},
            )
            if target_y != max(qys):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=target_y + 0.22,
                    y1=max(qys),
                    line={"color": wire_color, "width": 1.8},
                )
            for y in controls:
                add_control_marker(x, y)
            add_target_marker(x, target_y)
        elif gate == "measure" and qys:
            qy = qys[0]
            cy = cys[0] if cys else qy + 0.55
            add_gate_marker(x, qy, "M")
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=qy + 0.27,
                y1=cy,
                line={"color": wire_color, "width": 1.4},
            )
            fig.add_annotation(
                x=x,
                y=cy - 0.14,
                text=f"c{clbits[0]}" if clbits else "c",
                showarrow=False,
                yanchor="middle",
                xshift=16,
                font={"size": 11, "color": wire_color},
            )
        else:
            for y in qys or [0]:
                label = gate.upper()
                if gate == "id":
                    label = "I"
                add_gate_marker(x, y, label)

        # Transparent hover target over the rendered operation.
        hover_y = float(np.mean(ys)) if ys else 0
        hover_height = max(0.9, (max(ys) - min(ys) + 0.8) if ys else 0.9)
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[hover_y],
                mode="markers",
                marker={"size": max(34, hover_height * 30), "color": "rgba(0,0,0,0)"},
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            )
        )

    # Invert y-axis so q0 appears at the top.
    fig.update_layout(
        title="Interactive Qiskit-style circuit view",
        height=max(360, 90 + 60 * len(y_labels)),
        margin={"l": 106, "r": 24, "t": 34, "b": 12},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[len(y_labels) - 0.5, -0.5],
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[-1.05, max_x + 0.55])
    return fig
