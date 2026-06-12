# Grover Qiskit Streamlit Visualizer

这是一个面向课程选题五的交互式演示工程：**Grover 搜索算法中“振幅放大”过程的步进式可视化展示**。

工程采用：

- **Qiskit**：构造 Grover 量子电路；
- **Statevector**：提取 Oracle 与 Diffusion 后的中间量子态；
- **Matplotlib**：绘制/导出 Qiskit 标准电路图等静态图；
- **Plotly**：绘制概率幅图、概率图、目标态概率曲线、测量统计图、交互式 3D 步进概率图和宽画布电路操作视图；
- **Streamlit**：构建可视化交互界面。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果 `qiskit-aer` 安装失败，仍然可以使用 statevector 中间态展示和电路图；只是测量 `counts` 统计功能不可用。

## 2. 运行交互式界面

```bash
streamlit run app.py
```

打开网页后，可以在侧边栏选择：

- 量子比特数：2 或 3；
- 目标态：例如 `101`；
- Grover 迭代次数；
- 测量 shots；
- 是否显示 statevector 表格、3D 图、电路图、测量统计。

## 3. 命令行生成图片

也可以使用命令行一次性生成所有图片和 CSV：

```bash
python main.py --n 3 --target 101 --iterations 2
```

跳过 Aer 测量统计：

```bash
python main.py --n 3 --target 101 --iterations 2 --no-sim
```

输出文件保存在 `outputs/` 目录下。

## 4. 项目结构

```text
grover_qiskit_streamlit/
├── app.py                    # Streamlit 交互界面
├── main.py                   # 命令行入口
├── qiskit_grover_core.py     # Qiskit 电路构造与 statevector 分析
├── visualization.py          # Matplotlib 2D 绘图 + Plotly 3D 交互图
├── requirements.txt          # 依赖
└── README.md                 # 使用说明
```

## 5. 功能说明

本工程以 Qiskit 电路为主线，实现以下功能：

1. 构造 2 到 3 比特 Grover 搜索电路；
2. 支持任意目标态，例如 `10`、`101`；
3. 分别展示 Hadamard 初始化、Oracle 相位翻转、Diffusion 均值倒置后的 statevector；
4. 绘制概率幅柱状图，显示 Oracle 对目标态的相位翻转；
5. 绘制概率柱状图，显示目标态概率在 Diffusion 后被放大；
6. 绘制目标态概率随完整 Grover 迭代次数变化的曲线；
7. 绘制带边框容器的 Plotly 交互式 3D 步进概率历史图；
8. 显示 Qiskit 标准量子电路图，并提供可横向滚动的宽画布交互式操作视图；
9. 可选使用 Qiskit Aer 进行 shots 测量统计，并将横坐标统一显示为 ket 形式的计算基态。

## 6. 报告表述建议

可以在报告中写：

> 本项目采用 Streamlit 构建交互式 Web 界面，用户可以选择量子比特数、目标态和 Grover 迭代次数。程序基于 Qiskit 构造对应量子电路，并在每次 Oracle 与 Diffusion 操作后提取 statevector。界面通过滑动条实现步进式展示，分别绘制概率幅分布、测量概率分布、目标态概率变化曲线和 Plotly 交互式 3D 概率历史图，从而直观呈现 Grover 算法中的振幅放大过程。


## 交互界面改进说明

本版本针对课堂展示界面做了进一步调整：

1. “选择演化步骤”滑块已移动到主页面的“当前展示”区域，便于拖动时同步观察概率幅和概率变化。
2. 概率幅柱状图、概率柱状图、目标态概率曲线、测量 counts 统计均改为 Plotly 交互式图表，支持悬停查看精确数值，并可下载为 HTML。
3. Qiskit 标准电路图仍保留为官方静态电路图；同时新增 Plotly 交互式电路操作视图和操作表，可悬停查看每一步量子门、作用量子比特、经典比特和参数。

运行方式保持不变：

```bash
streamlit run app.py
```


## 7. 本版本界面修正

针对最近一版界面反馈，本版本做了以下修正：

1. **3D 步进概率历史图增加独立图框**：在 Streamlit 中用带边框的容器包围 3D 图，避免交互区域和普通页面区域混在一起。
2. **修正 3D 图标题与图例重叠**：增加顶部边距，将图例放到标题右侧并添加浅色背景框。
3. **交互式电路操作视图改为宽画布 + 横向滚动**：长电路不会再被强行压缩到页面宽度内，相邻量子门不容易挤在一起。
4. **Qiskit 标准电路图也放入横向滚动图框**：保留官方电路图样式，同时避免长图被页面缩小到难以阅读。
5. **测量统计图横坐标统一为 `|q0q1...>` 形式**：Qiskit 原始输出是普通经典 bitstring，例如 `101`；本版本将其转换为与 statevector 图一致的 ket 标记，并补全零计数基态。
