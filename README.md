# PCB 二维自动测量 — 面向非均匀光照与局部强反光条件的稳定性研究

## 项目目标

在非均匀光照和局部强反光条件下，研究 PCB 电路板二维自动测量流程的**稳定性**。核心问题不是单纯"精度更高"，而是：

- **算法 A (基础测量)** 在正常样本可工作，但在强光斑邻接条件下会因 mask 粘连而失效
- **算法 B (强光斑感知改进)** 通过高光分析与连通域约束，将 PCB 区域从反光干扰中分离
- **关键设计原则：算法 B 只改进 PCB mask 与区域分离，不改变后续测量链路**（外框检测、透视矫正、孔检测、尺寸计算方法完全一致），确保对比实验结论不受"测量算法也变了"干扰

## 测量对象与已知参数

| 参数 | 标称值 |
|------|--------|
| 电路板外框尺寸 | 100 mm × 100 mm |
| 安装孔中心间距 | 94 mm × 94 mm |
| 安装孔距板边 | 约 3 mm |
| 棋盘格单格边长 | 10 mm |
| PCB 待测图像数量 | 6 张（同一对象多次采集） |

> ⚠️ **重要**：电路板外框 (100×100 mm) 用作透视矫正的尺度基准，**不作为独立验证指标**。94×94 mm 安装孔中心距用作独立验证指标。

## 样本分类

根据光照干扰特征，待测样本分为两类：

| 类型 | 标签 | 样本 | 特征 | 评价重点 |
|------|------|------|------|----------|
| A | 正常/轻反光 | 01, 02, 03, 04 | PCB 本身局部发亮，背景光斑不严重 | 测量误差、精度保持 |
| B | 强光斑邻接 | 05, 06 | 巨大光斑紧贴 PCB，易导致 mask 粘连 | 板检测成功率、区域分离能力 |

## 算法说明

### 算法 A：基础测量算法

```
畸变校正 → HSV PCB mask → 连通域候选筛选 → 四角点定位 → 透视矫正 → 孔检测 → 尺寸测量
```

采用常规图像处理流程：畸变校正、HSV 绿色区域分割、连通域候选筛选、透视矫正和安装孔检测。

### 算法 B：强光斑感知改进算法

```
畸变校正 → HSV PCB mask → 高光区域检测 → 光斑/PCB 粘连区域分离 → 连通域候选筛选 → 后续测量流程同算法 A
```

在保持后续测量流程完全一致的前提下，引入高光区域检测与光斑-PCB 区域分离模块：

1. **高光检测**：检测 V 高 + S 低的白色反光区域
2. **区域分离**：对 B 类样本，利用高光 mask 引导切除 green_mask 中与反光粘连的区域
3. **安全回退**：区域分离后若无合法候选，自动回退到原始 mask

## 项目结构

```
pcb_measurement_project/
├── data/
│   ├── pcb/                  # PCB 待测图像（.bmp / .jpg / .png）
│   └── calibration/          # 棋盘格标定图像
├── src/
│   ├── config.py             # 所有可调参数集中管理
│   ├── utils.py              # 通用工具函数 (图像 I/O, DetectionResult)
│   ├── calibration/          # 层1: 相机标定
│   │   └── calibrate_camera.py
│   ├── preprocessing/        # 层2: 可替换预处理策略
│   │   ├── baseline.py       #   标准预处理 (算法 A)
│   │   ├── masks.py          #   PCB 绿色 mask、高光 mask、mask 后处理
│   │   ├── highlight.py      #   高光检测与抑制
│   │   └── illumination.py   #   光照场估计、暗区提升
│   ├── board_detection/      # 层3: PCB 外框检测
│   │   ├── board_detector.py     # 主检测入口
│   │   ├── candidate_filter.py   # 候选连通域筛选与打分
│   │   ├── region_separation.py  # 光斑与 PCB 粘连分离
│   │   ├── quad_recovery.py      # row-interval 四边形恢复（核心创新）
│   │   └── perspective.py        # 角点排序、单应性、透视变换
│   ├── measurement/          # 层4: 几何测量
│   │   ├── hole_detector.py  # 俯视图 ROI 孔检测
│   │   └── geometry.py       # 孔距/误差计算、汇总统计
│   └── experiments/          # 层5: 实验与报告
│       ├── configs.py        # ExperimentConfig + 预设配置
│       ├── runner.py         # 统一实验执行引擎
│       ├── statistics.py     # 统计指标与三类对比报告
│       └── report.py         # 可视化与报告图生成
├── outputs/
│   ├── calibration/          # 标定参数与角点可视化
│   ├── experiments/          # 算法中间结果
│   │   ├── algorithm_a/
│   │   └── algorithm_b/
│   └── reports/              # 对比实验 CSV 与报告图
├── main.py                   # 命令行入口
├── requirements.txt
└── README.md
```

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：
- `opencv-python >= 4.8.0`
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `matplotlib >= 3.7.0`

## 使用步骤

### 1. 放置数据

将图像文件放入对应目录：

```
data/pcb/           ← PCB 电路板图像 (01.bmp, 02.bmp, …, 06.bmp)
data/calibration/   ← 棋盘格标定图像 (01.bmp, 02.bmp, …)
```

支持的图像格式：`.bmp`, `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

### 2. 修改配置参数

编辑 `src/config.py`，根据实际情况修改：

- **`CHESSBOARD_PATTERN_SIZE`** — 必须修改！设置为棋盘格的**内角点**数量（不是棋盘格方块数）。
  - 例如：打印了 10×7 的棋盘格 → 内角点为 9×6 → 写 `(9, 6)`
- **`SAMPLE_TYPE_MAP`**（在 `src/experiments/configs.py`）— 按实际样本情况修改光照类型分类
- 其他参数可使用默认值，后续可根据实验结果微调

### 3. 相机标定

```bash
python main.py calibrate
```

输出：
- `outputs/calibration/camera_params.npz` — 相机内参与畸变系数
- `outputs/calibration/corners_*.png` — 每张标定图的角点检测可视化

### 4. 运行对比实验

```bash
python main.py compare
```

该命令自动：
1. 加载相机标定参数
2. **算法 A** 在全部样本上运行；**算法 B** 仅在 B 类（强光斑邻接）样本上运行
3. 两组算法的后续检测流程（board → holes → measure）完全一致
4. 按样本类型 (A/B) 分组统计，输出两类对比报告到 `outputs/reports/`

### 5. 单独运行实验（调试用）

```bash
python main.py algorithm-a       # 算法 A — 全部样本
python main.py algorithm-b       # 算法 B — 仅 B 类样本
```

## 输出说明

对比实验完成后，`outputs/reports/` 目录下包含：

| 文件 | 说明 |
|------|------|
| `measurements_algorithm_a.csv` | 算法 A 测量结果（含 sample_type、分离前后连通域数等） |
| `measurements_algorithm_b.csv` | 算法 B 测量结果 |
| `comparison_summary.csv` | 两组统计对比表 |
| `<图像名>/` | 每张图像的报告图子目录 |

### 统计报告

对比实验输出两类结构化报告：

1. **算法 A 在 A/B 两类样本上的运行结果**
   — 说明强光斑邻接会导致常规流程失效（A 类全成功，B 类全失败）

2. **算法 A 与算法 B 在 B 类样本上的对比**
   — 核心对比：区域分离 + 四边形恢复能否将 B 类从"完全无法测量"恢复为"端到端可测量"
   — 含分离前后连通域变化、四边形恢复触发情况

### 统计指标

| 指标 | 说明 |
|------|------|
| 板检测成功率 | 外框四边形成功定位的比例 |
| 四角点定位成功率 | 四角点正确排序的比例 |
| 孔检测成功率 | 四个安装孔全部检测成功的比例 |
| X/Y 孔距误差 | 与标称值 94.0 mm 的绝对偏差 |
| 平均绝对误差 | X/Y 两方向误差的均值 |
| 重复性标准差 | 多张图像测量结果的离散度 |
| 强光斑分离是否生效 | 区域分离模块是否实际触发 |
| 分离前/后连通域数量 | 区域分离前后 mask 连通域变化 |
| 候选区域面积、宽高比、位置评分 | 候选筛选器行为分析 |

## 注意事项

1. 电路板外框用作透视矫正基准，94 mm 孔间距用作独立验证指标。
2. 棋盘格标定板为 A4 纸打印，单格 10 mm，标定阶段不引入光照矫正。
3. 检测失败时程序不会崩溃，在 CSV 中记录失败原因，精确到失败阶段。
4. 所有参数集中在 `src/config.py`，调整参数无需修改算法代码。
5. 预处理输出多种中间结果（mask、光照场、高光 mask），后续检测模块可选择使用。
6. 算法 B 仅改变 mask 生成与区域分离，后续测量链路与算法 A 完全一致。
