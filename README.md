# Grover Qiskit Streamlit Visualizer

这是一个基于 Qiskit 和 Streamlit 的 Grover 搜索算法可视化程序。项目支持 2 到 3 个量子比特的单目标搜索，并逐步展示 Hadamard 初始化、Oracle 相位翻转和 Diffusion 均值倒置后的量子态变化。

核心功能包括：

- 使用 Qiskit 构造 Grover 量子电路；
- 使用 `Statevector` 提取每一步后的精确量子态；
- 使用 Plotly 绘制概率幅、测量概率、目标态概率曲线和 3D 分布图；
- 使用 Streamlit 提供交互式参数选择和结果展示；
- 可选使用 Qiskit Aer 进行 shots 测量统计和简化噪声对照。

## 在线运行

项目可以通过 Streamlit Community Cloud 直接访问：

https://zju-grover-visualizer-7qrbuk87raprvngeew75ml.streamlit.app/

## 安装依赖

```bash
pip install -r requirements.txt
```

如果 `qiskit-aer` 安装失败，仍然可以使用 statevector 中间态展示、电路图和主要可视化功能；只有 Aer 测量统计和噪声对照功能不可用。

## 启动交互式界面

```bash
streamlit run app.py
```

界面侧边栏支持设置：

- 量子比特数：2 或 3；
- 目标态：例如 `10` 或 `101`；
- 最大 Grover 迭代次数；
- 是否显示 statevector 表格；
- 是否显示 Qiskit 电路结构；
- 是否显示理想与带噪声实验对照；
- Aer shots 数量和简化噪声参数。

## 命令行导出结果

也可以使用命令行生成静态图片和 CSV 数据：

```bash
python main.py --n 3 --target 101 --iterations 2
```

跳过 Aer 测量统计：

```bash
python main.py --n 3 --target 101 --iterations 2 --no-sim
```

输出文件保存在 `outputs/` 目录下。

## 项目结构

```text
grover_qiskit_ui_refined/
├── app.py                    # Streamlit 交互界面
├── main.py                   # 命令行导出入口
├── qiskit_grover_core.py     # Grover 电路构造与 statevector 分析
├── visualization.py          # Plotly/Matplotlib 可视化函数
├── requirements.txt          # Python 依赖
├── outputs/                  # 运行后生成的图片、HTML 和 CSV 文件
└── README.md                 # 代码使用说明
```

## 代码功能

程序围绕 Grover 搜索的状态演化实现以下功能：

1. 构造 2 到 3 量子比特的 Grover 搜索电路；
2. 支持选择任意计算基目标态；
3. 保存 Hadamard 初始化、Oracle 和 Diffusion 后的中间 statevector；
4. 绘制当前步骤的概率幅柱状图和测量概率柱状图；
5. 绘制目标态概率随完整 Grover 迭代次数变化的曲线；
6. 绘制 3D 概率分布对照图；
7. 展示 Qiskit 标准电路图和交互式电路操作视图；
8. 可选运行带噪声 Aer 模拟，用于对比理想概率和测量频率。
