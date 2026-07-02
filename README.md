# PCB 二维尺寸测量 — 非均匀光照矫正方法研究

## 项目目标

在非均匀光照条件下，通过光照矫正预处理算法提高 PCB 电路板二维尺寸测量的精度。

本项目的核心创新点是**基于光照背景估计与校正的图像预处理方法**，并与常规预处理方法进行定量对比，验证光照矫正对测量精度的改进效果。

## 测量对象与已知参数

| 参数 | 标称值 |
|------|--------|
| 电路板外框尺寸 | 100 mm × 100 mm |
| 安装孔中心间距 | 94 mm × 94 mm |
| 安装孔距板边 | 约 3 mm |
| 棋盘格单格边长 | 10 mm |
| PCB 待测图像数量 | 7 张（同一对象多次测量） |

> ⚠️ **重要**：电路板外框 (100×100 mm) 用作透视矫正的尺度基准，**不作为独立验证指标**。94×94 mm 安装孔中心距用作独立验证指标。

## 项目结构

```
pcb_measurement_project/
├── data/
│   ├── board/            # 放置 PCB 待测图像（.bmp / .jpg / .png）
│   └── chessboard/       # 放置棋盘格标定图像
├── src/
│   ├── config.py                    # 所有可调参数
│   ├── calibrate_camera.py          # 相机标定
│   ├── preprocess_baseline.py       # 普通预处理（对照组）
│   ├── preprocess_light_correct.py  # 光照矫正预处理（改进组）
│   ├── detect_board.py              # 电路板外框检测
│   ├── detect_holes.py              # 安装孔检测
│   ├── measure_geometry.py          # 几何测量与误差计算
│   ├── compare_experiment.py        # 对比实验主流程
│   ├── visualize_results.py         # 可视化输出
│   └── utils.py                     # 通用工具函数
├── outputs/
│   ├── calibration/      # 标定参数与角点可视化
│   ├── baseline/         # 普通预处理中间结果
│   ├── light_corrected/  # 光照矫正预处理中间结果
│   └── comparison/       # 对比实验汇总 CSV 与报告图
├── main.py               # 命令行入口
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
data/board/       ← PCB 电路板图像（1.bmp, 2.bmp, …, 7.bmp）
data/chessboard/  ← 棋盘格标定图像（11.bmp, 22.bmp, …）
```

支持的图像格式：`.bmp`, `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

### 2. 修改配置参数

编辑 `src/config.py`，根据实际情况修改：

- **`CHESSBOARD_PATTERN_SIZE`** — 必须修改！设置为棋盘格的**内角点**数量（不是棋盘格方块数）。
  - 例如：打印了 10×7 的棋盘格 → 内角点为 9×6 → 写 `(9, 6)`
- 其他参数可使用默认值，后续可根据实验结果微调。

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

该命令会自动：
1. 加载相机标定参数
2. 对每张 PCB 图像分别执行**普通预处理**（A 组）和**光照矫正预处理**（B 组）
3. 两组后续检测流程（board → holes → measure）完全一致
4. 输出结果到 `outputs/comparison/`

### 5. 单独运行预处理（调试用）

```bash
python main.py run_baseline          # 仅普通预处理
python main.py run_light_corrected   # 仅光照矫正预处理
```

## 输出说明

对比实验完成后，`outputs/comparison/` 目录下包含：

| 文件 | 说明 |
|------|------|
| `measurements_baseline.csv` | 普通预处理组的测量结果 |
| `measurements_light_corrected.csv` | 光照矫正预处理组的测量结果 |
| `comparison_summary.csv` | 两组结果的统计对比表 |
| `<图像名>/` | 每张图像的报告图子目录 |

每张图像的报告图包括：
1. 原图
2. 普通算法二值图
3. 改进算法光照矫正图
4. 普通算法边缘图
5. 改进算法边缘图
6. 普通算法最终检测结果
7. 改进算法最终检测结果
8. 左右对比拼图

## 算法说明

### 普通预处理（Baseline）
```
原图 → 畸变校正 → G 通道提取 → 高斯滤波 → 自适应阈值 → Canny 边缘检测
```

### 光照矫正预处理（Light Corrected）— 创新点
```
原图 → 畸变校正 → ExG 绿色增强 → 大尺度高斯模糊估计光照背景
    → 背景减法矫正 → 高亮抑制 → CLAHE 局部对比度增强
    → 自适应阈值 / Canny 边缘检测
```

核心思路：基于 Retinex 理论，将图像分解为反射分量和光照分量，通过估计并扣除低频光照分量来消除非均匀光照的影响。

### 共同检测流程（两组一致，保证公平对比）
```
预处理结果 → 四边形外框检测 → 透视矫正 → ROI 内圆孔检测 → 几何测量
```

## 注意事项

1. 电路板外框用作透视矫正基准，94 mm 孔间距用作独立验证指标。
2. 棋盘格标定板为 A4 纸打印，单格 10 mm，不对标定板图像使用光照矫正。
3. 检测失败时程序不会崩溃，会在 CSV 中记录失败原因。
4. 初始版本使用 HoughCircles 检测圆孔，后续可改为椭圆拟合提高精度。
5. 所有参数集中在 `src/config.py`，调整参数无需修改算法代码。
