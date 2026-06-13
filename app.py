"""Streamlit interface for the Qiskit-based Grover visualizer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from qiskit_grover_core import (
    analyze_distribution_comparison,
    basis_labels,
    build_grover_circuit,
    build_grover_history,
    probabilities_in_top_order,
    statevector_in_top_order,
    target_probability,
    target_probability_by_iteration,
)
from visualization import (
    circuit_operation_rows,
    plotly_amplitudes,
    plotly_circuit_operation_view,
    plotly_distribution_comparison_3d,
    plotly_noise_degradation_scan,
    plotly_probabilities,
    plotly_step_animation,
    plotly_target_probability_curve,
)


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

    st.divider()
    st.markdown("**显示选项**")
    animation_duration_ms = st.slider("步进动画间隔（毫秒）", 250, 1800, 700, 50)
    show_state_table = st.checkbox("显示 statevector 表格", value=True)
    show_circuit = st.checkbox("显示 Qiskit 电路", value=True)
    show_experiment_analysis = st.checkbox("显示理想/带噪声实验对照", value=True)

    with st.expander("高级：噪声模型参数", expanded=True):
        noise_shots = st.slider("噪声对照 shots", min_value=256, max_value=8192, value=2048, step=256)
        one_qubit_error = st.slider("单比特门错误率", min_value=0.0, max_value=0.02, value=0.001, step=0.001, format="%.3f")
        two_qubit_error = st.slider("双比特门错误率", min_value=0.0, max_value=0.08, value=0.01, step=0.002, format="%.3f")
        readout_error = st.slider("读出错误率", min_value=0.0, max_value=0.10, value=0.02, step=0.005, format="%.3f")
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
if "step_slider_nonce" not in st.session_state:
    st.session_state.step_slider_nonce = 0

animation_fig = plotly_step_animation(history, n, target, frame_duration_ms=animation_duration_ms)
st.plotly_chart(
    animation_fig,
    use_container_width=True,
    config={"displayModeBar": True, "responsive": True},
    key=f"step_animation_{n}_{target}_{max_iterations}_{animation_duration_ms}",
)

# with st.expander("补充检查：手动选帧、下载与 statevector", expanded=False):
with st.expander("补充检查：手动选帧与 statevector", expanded=False):
    control_cols = st.columns([1, 1, 1])
    with control_cols[0]:
        if st.button("上一帧", use_container_width=True):
            st.session_state.step_index = max(0, st.session_state.step_index - 1)
            st.session_state.step_slider_nonce += 1
    with control_cols[1]:
        if st.button("下一帧", use_container_width=True):
            st.session_state.step_index = min(len(history) - 1, st.session_state.step_index + 1)
            st.session_state.step_slider_nonce += 1
    with control_cols[2]:
        if st.button("重置", use_container_width=True):
            st.session_state.step_index = 0
            st.session_state.step_slider_nonce += 1

    step_index = st.slider(
        "选择用于详细检查的演化步骤",
        min_value=0,
        max_value=len(history) - 1,
        value=int(st.session_state.step_index),
        format="S%d",
        help="该滑块只控制下方单步检查区；主动画请使用动画图内部的播放按钮和进度条。",
        key=f"step_slider_{st.session_state.step_slider_nonce}",
    )
    if step_index != st.session_state.step_index:
        st.session_state.step_index = step_index

    selected_step = history[step_index]
    current_prob = target_probability(selected_step.statevector, target)
    metric_cols1 = st.columns(2)
    metric_cols1[0].metric("检查步骤类型", f"{selected_step.kind}")
    metric_cols1[1].metric("检查目标态概率", f"{current_prob * 100:.2f}%")
    st.info(step_explanation(selected_step.kind, target))

    amp_fig = plotly_amplitudes(selected_step, n, target)
    prob_fig = plotly_probabilities(selected_step, n, target)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(amp_fig, use_container_width=True, key=f"amp_fig_{selected_step.index}_{n}_{target}")
        # st.download_button(
        #     "下载当前概率幅交互图 HTML",
        #     data=plotly_to_html_bytes(amp_fig),
        #     file_name=f"step_{selected_step.index:02d}_amplitude.html",
        #     mime="text/html",
        # )
    with right:
        st.plotly_chart(prob_fig, use_container_width=True, key=f"prob_fig_{selected_step.index}_{n}_{target}")
        # st.download_button(
        #     "下载当前概率交互图 HTML",
        #     data=plotly_to_html_bytes(prob_fig),
        #     file_name=f"step_{selected_step.index:02d}_probability.html",
        #     mime="text/html",
        # )

    if show_state_table:
        st.dataframe(make_state_rows(selected_step, n, target), hide_index=True, use_container_width=True)

st.subheader("二、目标态概率随迭代次数变化")
curve_fig = plotly_target_probability_curve(probability_points)
st.plotly_chart(curve_fig, use_container_width=True, key=f"target_curve_{n}_{target}_{max_iterations}")
st.write(
    f"在当前参数下，完整 Grover 迭代后的最高目标态概率出现在 **第 {best_iteration} 次迭代**，"
    f"目标态概率约为 **{best_prob * 100:.2f}%**。"
)
if show_circuit:
    st.subheader("三、Qiskit 量子电路结构")
    qc = build_grover_circuit(n, target, max_iterations, measure=True, add_barriers=True)

    circuit_tabs = st.tabs(["交互式操作视图", "操作表", "Qiskit 标准电路图"])
    with circuit_tabs[0]:
        circuit_view = plotly_circuit_operation_view(qc)
        circuit_width = max(760, 42 * (len(qc.data) + 2))
        render_scrollable_plotly(circuit_view, height=264, min_width=circuit_width)
        # st.download_button(
        #     "下载交互式电路操作视图 HTML",
        #     data=plotly_to_html_bytes(circuit_view),
        #     file_name=f"grover_n{n}_target{target}_r{max_iterations}_circuit_view.html",
        #     mime="text/html",
        # )
    with circuit_tabs[1]:
        st.dataframe(circuit_operation_rows(qc), hide_index=True, use_container_width=True)
    with circuit_tabs[2]:
        try:
            circuit_fig = draw_circuit_figure(n, target, max_iterations)
            circuit_png = figure_to_png_bytes(circuit_fig)
            render_scrollable_png(circuit_png, height=280)
            # st.download_button(
            #     "下载 Qiskit 标准电路图 PNG",
            #     data=circuit_png,
            #     file_name=f"grover_n{n}_target{target}_r{max_iterations}_circuit.png",
            #     mime="image/png",
            # )
        except Exception as exc:
            st.warning("Matplotlib 电路图绘制失败，下面显示文本电路。通常安装 pylatexenc 后即可绘制 PNG 电路图。")
            st.code(str(qc.draw(output="text", fold=-1)), language="text")
            st.caption(f"绘图错误信息：{exc}")

if show_experiment_analysis:
    st.subheader("四、理想与带噪声实验对照")
    st.caption(
        "该模块扫描 0 到当前最大 Grover 迭代次数，用同一组噪声参数对比理想 statevector 概率、"
        "带噪声 Aer shots 频率，以及二者的差值。"
    )
    with st.spinner("正在运行理想/带噪声分布对照扫描……"):
        distribution_rows, experiment_rows, experiment_error = analyze_distribution_comparison(
            n,
            target,
            max_iterations,
            shots=noise_shots,
            one_qubit_error=one_qubit_error,
            two_qubit_error=two_qubit_error,
            readout_error=readout_error,
        )
    if distribution_rows is None or experiment_rows is None:
        st.warning(f"实验对照暂不可用：{experiment_error}")
    else:
        final_row = experiment_rows[-1]
        comparison_options = ["理想概率分布", "带噪声测量频率", "理想 - 带噪声差值"]
        if "distribution_comparison_mode" not in st.session_state:
            st.session_state.distribution_comparison_mode = comparison_options[0]
        comparison_mode_label = st.session_state.distribution_comparison_mode
        plot_col, summary_col = st.columns([0.7, 0.3], gap="large")
        with plot_col:
            comparison_mode = {
                "理想概率分布": "ideal_probability",
                "带噪声测量频率": "noisy_frequency",
                "理想 - 带噪声差值": "loss",
            }[comparison_mode_label]
            # st.radio(
            #     "3D 图显示模式",
            #     comparison_options,
            #     key="distribution_comparison_mode",
            #     horizontal=True,
            # )
            comparison_fig = plotly_distribution_comparison_3d(distribution_rows, n, target, mode=comparison_mode)
            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
                config={"displayModeBar": True},
                key=f"distribution_comparison_{n}_{target}_{max_iterations}_{noise_shots}_{one_qubit_error}_{two_qubit_error}_{readout_error}_{comparison_mode}",
            )
            st.radio(
                "3D 图显示模式",
                comparison_options,
                key="distribution_comparison_mode",
                horizontal=True,
            )

        with summary_col:
            st.markdown("**最终迭代摘要**")
            st.metric("理想目标概率", f"{float(final_row['ideal_target_probability']) * 100:.2f}%")
            st.metric("带噪声频率", f"{float(final_row['noisy_target_frequency']) * 100:.2f}%")
            st.metric("目标态损耗", f"{float(final_row['target_loss']) * 100:.2f}%")
            st.metric("电路深度", f"{int(final_row['depth'])}")
            st.metric("CX 门数量", f"{int(final_row['cx_count'])}")

        st.info(
            "解释：理想的 Grover 曲线在封闭的数学模型中表现出了振幅放大现象。"
            "带噪声频率来自简化的 depolarizing + readout noise 模型，展示了更深电路中栅极误差和读出误差如何累积。"
        )

        detail_tabs = st.tabs(["目标态损耗曲线", "迭代摘要表", "完整分布数据"])
        with detail_tabs[0]:
            noise_fig = plotly_noise_degradation_scan(experiment_rows)
            st.plotly_chart(
                noise_fig,
                use_container_width=True,
                config={"displayModeBar": True},
                key=f"noise_scan_{n}_{target}_{max_iterations}_{noise_shots}_{one_qubit_error}_{two_qubit_error}_{readout_error}",
            )
        with detail_tabs[1]:
            st.dataframe(experiment_rows, hide_index=True, use_container_width=True)
        with detail_tabs[2]:
            st.dataframe(distribution_rows, hide_index=True, use_container_width=True)

st.subheader("五、理论解释与错误迭代")
theory_tabs = st.tabs(["理论公式对照", "错误迭代演示", "Oracle 与 Diffusion 原理"])
with theory_tabs[0]:
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

with theory_tabs[1]:
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

with theory_tabs[2]:
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
