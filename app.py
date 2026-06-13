"""Streamlit interface for the Qiskit-based Grover visualizer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time
from typing import Dict, List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from qiskit_grover_core import (
    analyze_noisy_iteration_scan,
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
    circuit_operation_rows,
    plotly_3d_probability_history,
    plotly_amplitudes,
    plotly_circuit_operation_view,
    plotly_counts,
    plotly_noise_degradation_scan,
    plotly_probabilities,
    plotly_target_probability_curve,
)

PLAYBACK_STATE_VERSION = 1


def figure_to_png_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes for Streamlit downloads."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def plotly_to_html_bytes(fig) -> bytes:
    """Convert a Plotly figure to downloadable HTML bytes."""
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


def render_scrollable_plotly(fig, height: int, *, min_width: int = 1100) -> None:
    """Render a Plotly figure inside a bordered, horizontally scrollable frame."""
    fig.update_layout(width=min_width, height=height, autosize=False)
    html = fig.to_html(include_plotlyjs="cdn", full_html=False, default_width=f"{min_width}px", default_height=f"{height}px", config={"displayModeBar": True})
    components.html(
        f"""
        <div style="border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; padding: 8px; overflow-x: scroll; overflow-y: hidden; background: rgba(250,250,250,0.03);">
            <div style="width: {min_width}px; min-width: {min_width}px;">{html}</div>
        </div>
        """,
        height=height + 38,
        scrolling=False,
    )


def render_scrollable_png(png_bytes: bytes, *, height: int = 520, min_width: int | None = None) -> None:
    """Render a PNG inside a bordered, horizontally scrollable frame."""
    import base64

    b64 = base64.b64encode(png_bytes).decode("ascii")
    frame_height = max(120, height - 2)
    image_height = max(80, height - 26)
    width_style = f"min-width: {min_width}px;" if min_width is not None else "width: max-content;"
    components.html(
        f"""
        <div style="box-sizing: border-box; height: {frame_height}px; border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; padding: 12px; overflow-x: auto; overflow-y: hidden; background: rgba(250,250,250,0.03);">
            <div style="{width_style} height: {image_height}px;">
                <img src="data:image/png;base64,{b64}" style="max-width: none; height: 100%; width: auto; display: block;" />
            </div>
        </div>
        """,
        height=height,
        scrolling=False,
    )


def step_explanation(kind: str, target: str) -> str:
    """Return a short teaching explanation for the selected Grover step."""
    if kind == "initial":
        return (
            "当前步骤为 Hadamard 初始化后的均匀叠加态。"
            f"此时所有计算基态具有相同概率幅，目标态 |{target}> 还没有被突出。"
        )
    if kind == "oracle":
        return (
            f"当前步骤为 Oracle 相位翻转。Oracle 将目标态 |{target}> 的概率幅乘以 -1，"
            "但概率是概率幅模平方，因此仅看概率图时通常不会立刻看到目标概率升高。"
        )
    if kind == "diffusion":
        return (
            "当前步骤为 Diffusion，也就是关于平均概率幅的反演。"
            "它把 Oracle 产生的相位标记转化为振幅放大，使目标态概率上升。"
        )
    return "当前步骤展示 Grover 电路演化中的一个中间状态。"

def make_state_rows(step, n: int, target: str) -> List[Dict[str, object]]:
    labels = basis_labels(n)
    amps = statevector_in_top_order(step.statevector, n)
    probs = probabilities_in_top_order(step.statevector, n)
    rows: List[Dict[str, object]] = []
    for label, amp, prob in zip(labels, amps, probs):
        rows.append(
            {
                "basis": label,
                "real_amp": round(float(amp.real), 6),
                "imag_amp": round(float(amp.imag), 6),
                "probability": round(float(prob), 6),
                "is_target": label == f"|{target}>",
            }
        )
    return rows


def draw_circuit_figure(n: int, target: str, iterations: int):
    qc = build_grover_circuit(n, target, iterations, measure=True, add_barriers=True)
    return qc.draw(output="mpl", fold=-1, idle_wires=False)


st.set_page_config(page_title="Grover Visualizer", layout="wide")

st.title("Grover 搜索算法振幅放大交互式可视化")
st.caption("Qiskit 电路主线 + Statevector 中间态分析 + Plotly 交互式可视化")

with st.sidebar:
    st.header("实验参数")
    n = st.selectbox("量子比特数 n", [2, 3], index=1, help="课程题目要求 2 到 3 个量子比特。")
    raw_labels = [label.strip("|>") for label in basis_labels(n)]
    default_target = "101" if n == 3 else "10"
    target = st.selectbox("目标态", raw_labels, index=raw_labels.index(default_target))
    max_iterations = st.slider("最大 Grover 迭代次数", min_value=0, max_value=6, value=2 if n == 3 else 1)
    shots = st.slider("测量 shots", min_value=128, max_value=8192, value=2048, step=128)
    run_measurement = st.checkbox("运行 Aer 测量统计", value=True)
    noisy_measurement = st.checkbox("测量统计使用同一噪声模型", value=False)
    show_noise_analysis = st.checkbox("显示噪声/深度损耗分析", value=True)
    noise_shots = st.slider("噪声扫描 shots", min_value=256, max_value=8192, value=2048, step=256)
    one_qubit_error = st.slider("单比特门错误率", min_value=0.0, max_value=0.02, value=0.001, step=0.001, format="%.3f")
    two_qubit_error = st.slider("双比特门错误率", min_value=0.0, max_value=0.08, value=0.01, step=0.002, format="%.3f")
    readout_error = st.slider("读出错误率", min_value=0.0, max_value=0.10, value=0.02, step=0.005, format="%.3f")

    st.divider()
    st.markdown("**显示选项**")
    show_state_table = st.checkbox("显示 statevector 表格", value=True)
    show_3d = st.checkbox("显示 Plotly 交互式 3D 步进概率图", value=True)
    show_circuit = st.checkbox("显示 Qiskit 电路", value=True)
try:
    history = build_grover_history(n, target, max_iterations)
    probability_points = target_probability_by_iteration(history, target)
except Exception as exc:
    st.error("工程运行失败。请确认已经安装 requirements.txt 中的依赖，尤其是 qiskit。")
    st.exception(exc)
    st.stop()

current_prob_default_step = min(2, len(history) - 1)
step_names = [f"S{step.index} | {step.kind} | iter {step.iteration}" for step in history]
best_iteration, best_prob = max(probability_points, key=lambda item: item[1])
expected_best = int(np.floor(np.pi / 4 * np.sqrt(2**n)))

metric_cols = st.columns(4)
metric_cols[0].metric("目标态", f"|{target}>")
metric_cols[1].metric("总演化步骤", f"{len(history)}")
metric_cols[2].metric("最高目标态概率", f"{best_prob * 100:.2f}%", f"iter {best_iteration}")
metric_cols[3].metric("经验最佳迭代", f"r ≈ {expected_best}")

st.subheader("一、当前展示")
if "step_index" not in st.session_state or st.session_state.step_index >= len(history):
    st.session_state.step_index = current_prob_default_step
if st.session_state.get("playback_state_version") != PLAYBACK_STATE_VERSION:
    st.session_state.is_playing_steps = False
    st.session_state.playback_state_version = PLAYBACK_STATE_VERSION
elif "is_playing_steps" not in st.session_state:
    st.session_state.is_playing_steps = False
if "step_slider_nonce" not in st.session_state:
    st.session_state.step_slider_nonce = 0

control_cols = st.columns([1, 1, 1, 1, 2])
with control_cols[0]:
    if st.button("上一帧", use_container_width=True):
        st.session_state.step_index = max(0, st.session_state.step_index - 1)
        st.session_state.is_playing_steps = False
        st.session_state.step_slider_nonce += 1
with control_cols[1]:
    play_button_label = "暂停" if st.session_state.is_playing_steps else "播放"
    if st.button(play_button_label, use_container_width=True):
        st.session_state.is_playing_steps = not st.session_state.is_playing_steps
with control_cols[2]:
    if st.button("下一帧", use_container_width=True):
        st.session_state.step_index = min(len(history) - 1, st.session_state.step_index + 1)
        st.session_state.is_playing_steps = False
        st.session_state.step_slider_nonce += 1
with control_cols[3]:
    if st.button("重置", use_container_width=True):
        st.session_state.step_index = 0
        st.session_state.is_playing_steps = False
        st.session_state.step_slider_nonce += 1
with control_cols[4]:
    playback_delay = st.slider("播放间隔（秒）", 0.2, 2.0, 0.8, 0.1)
st.caption(f"播放状态：{'播放中' if st.session_state.is_playing_steps else '已暂停'}")

step_index = st.slider(
    "选择演化步骤",
    min_value=0,
    max_value=len(history) - 1,
    value=int(st.session_state.step_index),
    format="S%d",
    help="拖动滑块观察每一步后的概率幅和概率变化。",
    key=f"step_slider_{st.session_state.step_slider_nonce}",
)
if step_index != st.session_state.step_index:
    st.session_state.step_index = step_index
    st.session_state.is_playing_steps = False
selected_step = history[step_index]
current_prob = target_probability(selected_step.statevector, target)

metric_cols1 = st.columns(2)
metric_cols1[0].metric("当前步骤类型", f"{selected_step.kind}")
metric_cols1[1].metric("当前目标态概率", f"{current_prob * 100:.2f}%")
amp_fig = plotly_amplitudes(selected_step, n, target)
prob_fig = plotly_probabilities(selected_step, n, target)

left, right = st.columns(2)
with left:
    st.plotly_chart(amp_fig, use_container_width=True, key=f"amp_fig_{selected_step.index}_{n}_{target}")
    st.download_button(
        "下载当前概率幅交互图 HTML",
        data=plotly_to_html_bytes(amp_fig),
        file_name=f"step_{selected_step.index:02d}_amplitude.html",
        mime="text/html",
    )
with right:
    st.plotly_chart(prob_fig, use_container_width=True, key=f"prob_fig_{selected_step.index}_{n}_{target}")
    st.download_button(
        "下载当前概率交互图 HTML",
        data=plotly_to_html_bytes(prob_fig),
        file_name=f"step_{selected_step.index:02d}_probability.html",
        mime="text/html",
    )

if show_state_table:
    with st.expander("查看当前 statevector 数值表", expanded=False):
        st.dataframe(make_state_rows(selected_step, n, target), hide_index=True, use_container_width=True)

st.info(step_explanation(selected_step.kind, target))

st.subheader("二、目标态概率随迭代次数变化")
curve_fig = plotly_target_probability_curve(probability_points)
st.plotly_chart(curve_fig, use_container_width=True, key=f"target_curve_{n}_{target}_{max_iterations}")
st.write(
    f"在当前参数下，完整 Grover 迭代后的最高目标态概率出现在 **第 {best_iteration} 次迭代**，"
    f"目标态概率约为 **{best_prob * 100:.2f}%**。"
)
st.download_button(
    "下载目标态概率曲线 HTML",
    data=plotly_to_html_bytes(curve_fig),
    file_name="target_probability_curve.html",
    mime="text/html",
)

with st.expander("错误迭代次数演示：迭代过多会回落", expanded=True):
    over_iteration_rows = []
    previous_prob = None
    for iteration, probability in probability_points:
        if previous_prob is None:
            trend = "initial"
            delta = None
        else:
            delta = probability - previous_prob
            trend = "increase" if delta > 1e-9 else "decrease" if delta < -1e-9 else "flat"
        over_iteration_rows.append(
            {
                "iteration r": int(iteration),
                "target probability": round(float(probability), 6),
                "change from previous": round(float(delta), 6) if delta is not None else None,
                "trend": trend,
                "is best": int(iteration) == int(best_iteration),
            }
        )
        previous_prob = probability

    final_iteration, final_probability = probability_points[-1]
    if best_iteration < final_iteration:
        st.warning(
            f"当前扫描已经出现过迭代现象：最佳迭代是 r={best_iteration}，"
            f"目标概率约 {best_prob * 100:.2f}%；继续到 r={final_iteration} 后，"
            f"目标概率变为 {final_probability * 100:.2f}%。"
        )
    else:
        st.info(
            "当前最大迭代次数还没有明显展示过迭代回落。"
            "可以把左侧“最大 Grover 迭代次数”调大，观察目标概率超过最佳点后下降。"
        )
    st.dataframe(over_iteration_rows, hide_index=True, use_container_width=True)

with st.expander("理论公式对照：Grover 振幅放大", expanded=True):
    search_space_size = 2**n
    theta = float(np.arcsin(1 / np.sqrt(search_space_size)))
    theory_rows = []
    simulated_by_iter = {int(iteration): float(prob) for iteration, prob in probability_points}
    for iteration in range(max_iterations + 1):
        theoretical_prob = float(np.sin((2 * iteration + 1) * theta) ** 2)
        simulated_prob = simulated_by_iter.get(iteration)
        theory_rows.append(
            {
                "iteration r": iteration,
                "theory P_r": round(theoretical_prob, 6),
                "statevector P_r": round(simulated_prob, 6) if simulated_prob is not None else None,
                "absolute error": round(abs(theoretical_prob - simulated_prob), 6) if simulated_prob is not None else None,
            }
        )

    st.markdown(
        rf"""
对于单个目标态的 Grover 搜索，令搜索空间大小为 \(N=2^n={search_space_size}\)，并定义

$$
\sin(\theta)=\frac{{1}}{{\sqrt{{N}}}}
$$

完成 \(r\) 次 Grover 迭代后，目标态的理论概率为

$$
P_r=\sin^2((2r+1)\theta)
$$

当前参数下 \(\theta \approx {theta:.6f}\)，经验最优迭代次数约为

$$
r_\text{{opt}}\approx \left\lfloor \frac{{\pi}}{{4}}\sqrt{{N}} \right\rfloor = {expected_best}.
$$
"""
    )
    st.dataframe(theory_rows, hide_index=True, use_container_width=True)

if show_3d:
    st.subheader("三、Plotly 交互式 3D 步进概率历史图")
    st.caption("x 轴为计算基态，y 轴为演化步骤，z 轴为对应测量概率。可以旋转、缩放并悬停查看数值。")
    fig3d = plotly_3d_probability_history(history, n, target)
    with st.container(border=True):
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": True, "responsive": True}, key=f"probability_history_3d_{n}_{target}_{max_iterations}")
    st.download_button(
        "下载 3D 交互图 HTML",
        data=plotly_to_html_bytes(fig3d),
        file_name="probability_history_3d.html",
        mime="text/html",
    )

if show_circuit:
    st.subheader("四、Qiskit 量子电路")
    qc = build_grover_circuit(n, target, max_iterations, measure=True, add_barriers=True)

    circuit_tabs = st.tabs(["交互式操作视图", "操作表", "Qiskit 标准电路图"])
    with circuit_tabs[0]:
        circuit_view = plotly_circuit_operation_view(qc)
        circuit_width = max(760, 42 * (len(qc.data) + 2))
        render_scrollable_plotly(circuit_view, height=264, min_width=circuit_width)
        st.download_button(
            "下载交互式电路操作视图 HTML",
            data=plotly_to_html_bytes(circuit_view),
            file_name=f"grover_n{n}_target{target}_r{max_iterations}_circuit_view.html",
            mime="text/html",
        )
    with circuit_tabs[1]:
        st.dataframe(circuit_operation_rows(qc), hide_index=True, use_container_width=True)
    with circuit_tabs[2]:
        try:
            circuit_fig = draw_circuit_figure(n, target, max_iterations)
            circuit_png = figure_to_png_bytes(circuit_fig)
            render_scrollable_png(circuit_png, height=280)
            st.download_button(
                "下载 Qiskit 标准电路图 PNG",
                data=circuit_png,
                file_name=f"grover_n{n}_target{target}_r{max_iterations}_circuit.png",
                mime="image/png",
            )
        except Exception as exc:
            st.warning("Matplotlib 电路图绘制失败，下面显示文本电路。通常安装 pylatexenc 后即可绘制 PNG 电路图。")
            st.code(str(qc.draw(output="text", fold=-1)), language="text")
            st.caption(f"绘图错误信息：{exc}")

if show_noise_analysis:
    st.subheader("五、噪声与电路深度损耗分析")
    st.caption(
        "该模块使用一个简化的 depolarizing + readout noise 模型，扫描 0 到当前最大 Grover 迭代次数，"
        "对比理想目标态概率、带噪声测量频率，以及 transpile 后的电路深度。"
    )
    with st.spinner("Running noisy iteration scan..."):
        noise_rows, noise_error = analyze_noisy_iteration_scan(
            n,
            target,
            max_iterations,
            shots=noise_shots,
            one_qubit_error=one_qubit_error,
            two_qubit_error=two_qubit_error,
            readout_error=readout_error,
        )
    if noise_rows is None:
        st.warning(f"Noise analysis is unavailable: {noise_error}")
    else:
        noise_fig = plotly_noise_degradation_scan(noise_rows)
        st.plotly_chart(
            noise_fig,
            use_container_width=True,
            config={"displayModeBar": True},
            key=f"noise_scan_{n}_{target}_{max_iterations}_{noise_shots}_{one_qubit_error}_{two_qubit_error}_{readout_error}",
        )
        st.dataframe(noise_rows, hide_index=True, use_container_width=True)
        st.info(
            "解释：理想的格罗弗曲线在封闭的数学模型中表现出了振幅放大现象。"
            "噪声曲线估算了随着电路层数增加，栅极误差和读出误差如何导致观测到的目标频率降低。"
        )
st.subheader("六、测量统计验证")
if run_measurement:
    measurement_mode = "noisy" if noisy_measurement else "ideal"
    with st.spinner(f"正在运行 Qiskit Aer {measurement_mode} 测量模拟……"):
        counts, error = simulate_measurements(
            n,
            target,
            max_iterations,
            shots=shots,
            one_qubit_error=one_qubit_error if noisy_measurement else 0.0,
            two_qubit_error=two_qubit_error if noisy_measurement else 0.0,
            readout_error=readout_error if noisy_measurement else 0.0,
        )
    if counts is None:
        st.warning(f"测量统计暂不可用：{error}")
        st.write("这通常是因为没有安装 qiskit-aer。你仍然可以使用 statevector 中间态图完成主要展示。")
    else:
        st.caption(
            "当前测量统计模式："
            + (
                f"带噪声模拟（1q={one_qubit_error:.3f}, 2q={two_qubit_error:.3f}, readout={readout_error:.3f}）。"
                if noisy_measurement
                else "理想 Aer shots 抽样，不包含 gate/readout noise。"
            )
        )
        counts_fig = plotly_counts(counts, target, n=n)
        st.plotly_chart(
            counts_fig,
            use_container_width=True,
            config={"displayModeBar": True},
            key=f"measurement_counts_{n}_{target}_{max_iterations}_{shots}_{noisy_measurement}_{one_qubit_error}_{two_qubit_error}_{readout_error}",
        )
        target_counts = counts.get(target, 0)
        st.write(f"目标态 `{target}` 出现次数：**{target_counts} / {shots}**，频率约为 **{target_counts / shots * 100:.2f}%**。")
        st.download_button(
            "下载测量统计交互图 HTML",
            data=plotly_to_html_bytes(counts_fig),
            file_name="measurement_counts.html",
            mime="text/html",
        )
        with st.expander("查看原始 counts 字典", expanded=False):
            st.json(counts)
else:
    st.write("已关闭 Aer 测量统计。打开左侧选项可运行 shots 统计。")

with st.expander("原理提示：为什么 Oracle 后概率不一定变化？", expanded=False):
    st.markdown(
        r"""
Oracle 的作用是将目标态概率幅乘以 \(-1\)，即完成相位标记。由于测量概率等于概率幅模平方，
所以仅执行 Oracle 后，目标态概率通常不会升高。随后 Diffusion 执行关于平均概率幅的反演：

$$
a_i' = 2\bar{a} - a_i
$$

这样目标态在相位翻转后会被反演到更大的正振幅位置，从而实现振幅放大。
        """
    )

if st.session_state.get("is_playing_steps", False):
    time.sleep(playback_delay)
    st.session_state.step_index = (st.session_state.step_index + 1) % len(history)
    st.session_state.step_slider_nonce += 1
    st.rerun()
