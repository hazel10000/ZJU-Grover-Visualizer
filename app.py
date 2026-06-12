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
    plotly_probabilities,
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
        <div style="border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; padding: 12px; overflow-x: auto; background: rgba(250,250,250,0.03);">
            <div style="min-width: {min_width}px;">{html}</div>
        </div>
        """,
        height=height + 42,
        scrolling=False,
    )


def render_scrollable_png(png_bytes: bytes, *, height: int = 520, min_width: int = 1200) -> None:
    """Render a PNG inside a bordered, horizontally scrollable frame."""
    import base64

    b64 = base64.b64encode(png_bytes).decode("ascii")
    frame_height = max(120, height - 2)
    image_height = max(80, height - 26)
    components.html(
        f"""
        <div style="box-sizing: border-box; height: {frame_height}px; border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; padding: 12px; overflow-x: auto; overflow-y: hidden; background: rgba(250,250,250,0.03);">
            <div style="min-width: {min_width}px; height: {image_height}px;">
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
            "当前步骤为 Hadamard 初始化后的均匀叠加态。此时所有计算基态具有相同的概率幅，"
            f"目标态 |{target}> 还没有被突出。"
        )
    if kind == "oracle":
        return (
            f"当前步骤为 Oracle 相位翻转。Oracle 将目标态 |{target}> 的概率幅乘以 -1，"
            "但概率是概率幅模平方，因此仅看概率图时通常不会立刻看到目标概率升高。"
        )
    if kind == "diffusion":
        return (
            "当前步骤为 Diffusion，也就是关于平均概率幅的反演。它把 Oracle 产生的相位标记"
            "转化为振幅放大，使目标态概率上升。"
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
metric_cols[2].metric("当前参数下最高目标态概率", f"{best_prob * 100:.2f}%", f"iter {best_iteration}")
metric_cols[3].metric("经验最佳迭代", f"r ≈ {expected_best}")

st.subheader("一、当前展示")
step_index = st.slider(
    "选择演化步骤",
    min_value=0,
    max_value=len(history) - 1,
    value=current_prob_default_step,
    format="S%d",
    help="这里放在当前展示区域，便于一边拖动一边观察概率幅和概率图的变化。",
)
selected_step = history[step_index]
current_prob = target_probability(selected_step.statevector, target)

st.write(f"当前展示：**{selected_step.title}**")
st.info(step_explanation(selected_step.kind, target))
st.metric("当前步骤的目标态概率", f"{current_prob * 100:.2f}%")

amp_fig = plotly_amplitudes(selected_step, n, target)
prob_fig = plotly_probabilities(selected_step, n, target)

left, right = st.columns(2)
with left:
    st.plotly_chart(amp_fig, use_container_width=True)
    st.download_button(
        "下载当前概率幅交互图 HTML",
        data=plotly_to_html_bytes(amp_fig),
        file_name=f"step_{selected_step.index:02d}_amplitude.html",
        mime="text/html",
    )
with right:
    st.plotly_chart(prob_fig, use_container_width=True)
    st.download_button(
        "下载当前概率交互图 HTML",
        data=plotly_to_html_bytes(prob_fig),
        file_name=f"step_{selected_step.index:02d}_probability.html",
        mime="text/html",
    )

if show_state_table:
    with st.expander("查看当前 statevector 数值表", expanded=False):
        st.dataframe(make_state_rows(selected_step, n, target), hide_index=True, use_container_width=True)

st.subheader("二、目标态概率随迭代次数变化")
curve_fig = plotly_target_probability_curve(probability_points)
st.plotly_chart(curve_fig, use_container_width=True)
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

if show_3d:
    st.subheader("三、Plotly 交互式 3D 步进概率历史图")
    st.caption("x轴为计算基态，y轴为演化步骤，z轴为对应测量概率。可以用鼠标旋转、缩放并悬停查看数值。")
    fig3d = plotly_3d_probability_history(history, n, target)
    with st.container(border=True):
        st.caption("该区域为独立交互图框：可旋转、缩放、拖动，并可悬停查看具体概率。")
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": True})
    st.download_button(
        "下载 3D 交互图 HTML",
        data=plotly_to_html_bytes(fig3d),
        file_name="probability_history_3d.html",
        mime="text/html",
    )

if show_circuit:
    st.subheader("四、Qiskit 量子电路")
    st.caption("Qiskit 的标准电路图本质上仍是静态图；因此这里额外提供一个 Plotly 交互式操作视图和操作表，用于悬停查看每个门的细节。")
    qc = build_grover_circuit(n, target, max_iterations, measure=True, add_barriers=True)

    circuit_tabs = st.tabs(["交互式操作视图", "操作表", "Qiskit 标准电路图"])
    with circuit_tabs[0]:
        st.caption("这个视图使用宽画布绘制；如果电路较长，可以在图框内横向滚动，避免门元件挤在一起。")
        circuit_view = plotly_circuit_operation_view(qc)
        circuit_width = max(1200, 85 * (len(qc.data) + 2))
        render_scrollable_plotly(circuit_view, height=260, min_width=circuit_width)
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
            render_scrollable_png(circuit_png, height=280, min_width=max(1200, 90 * (len(qc.data) + 2)))
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

st.subheader("五、测量统计验证")
if run_measurement:
    with st.spinner("正在运行 Qiskit Aer 测量模拟……"):
        counts, error = simulate_measurements(n, target, max_iterations, shots=shots)
    if counts is None:
        st.warning(f"测量统计暂不可用：{error}")
        st.write("这通常是因为没有安装 qiskit-aer。你仍然可以使用 statevector 中间态图完成主要展示。")
    else:
        st.caption("横坐标现在按计算基态顺序显示为 |q0q1...>。Qiskit 原始 counts 是普通经典比特串；这里已转换成和前文一致的 ket 标记，并补全未出现的零计数基态。")
        counts_fig = plotly_counts(counts, target, n=n)
        st.plotly_chart(counts_fig, use_container_width=True, config={"displayModeBar": True})
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
