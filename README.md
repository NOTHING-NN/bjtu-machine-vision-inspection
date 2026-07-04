# PCB 二维图像测量系统

本项目实现一个面向 PCB 电路板的二维图像测量系统，研究在非均匀光照和局部强反光条件下，常规测量流程与强光斑感知改进流程的检测稳定性。系统先使用 A4 打印黑白棋盘格完成相机标定和测量平面毫米坐标标定，棋盘格单格边长为 `10 mm`；随后对同一块 PCB 的多张图像进行外框检测、四角点定位、透视矫正、安装孔定位、孔距与孔径计算、板长宽测量和误差统计。

PCB 标称外框尺寸为 `100 mm × 100 mm`，四个安装孔中心距标称值为 `94 mm × 94 mm`，安装孔直径标准值为 `3 mm`，用于最终误差评价。实际尺寸测量的毫米坐标由棋盘格测量平面单应性提供。

## 研究内容

项目将待测 PCB 图像分为两类：

| 类型 | 样本 | 光照特征 | 实验作用 |
|------|------|----------|----------|
| A 类 | 01, 02, 03, 04 | PCB 板面存在局部反光，但背景强光斑干扰较轻 | 验证基础测量流程在常规条件下的可用性 |
| B 类 | 05, 06 | 强光斑紧贴 PCB 边缘，容易与 PCB 区域粘连 | 验证强光斑感知改进算法的恢复能力 |

实验设计包含两条算法路线：

| 算法 | 运行范围 | 作用 |
|------|----------|------|
| 算法 A | 全部样本 | 基础测量流程，用于建立常规方法的性能边界 |
| 算法 B | B 类样本 | 强光斑感知改进流程，用于恢复强光斑邻接条件下的外框检测和端到端测量 |

算法 B 重点改进 PCB mask 生成、强光斑粘连处理和四边形恢复，后续孔检测与几何测量链路与算法 A 保持一致。

## 测量流程

### 标定流程

```text
棋盘格图像
  → 灰度化
  → 棋盘格角点检测
  → 亚像素角点优化
  → 相机内参和畸变系数估计
  → 测量平面 image pixel → world mm 单应性估计
```

标定输出：

| 文件 | 说明 |
|------|------|
| `outputs/calibration/camera_params.npz` | 相机内参和畸变系数 |
| `outputs/calibration/measurement_plane_homography.npz` | 图像像素到测量平面毫米坐标的单应性 |
| `outputs/calibration/corners_*.png` | 棋盘格角点检测可视化 |
| `outputs/calibration/measurement_plane_reference.png` | 测量平面参考图 |

### 算法 A：基础测量

```text
PCB 图像
  → 畸变校正
  → HSV 绿色区域分割
  → 连通域候选筛选与打分
  → PCB 四边形外框检测
  → 透视矫正
  → 俯视图四孔检测与半径精修
  → 原图/俯视图坐标回映射
  → 测量平面毫米坐标计算
  → 板长宽、孔心、孔径、孔距和误差统计
```

安装孔检测先在 PCB 俯视图中提取暗圆形候选，并根据四象限拓扑选择四个安装孔；随后对孔半径进行径向灰度剖面精修，寻找暗孔区域到较亮 PCB 表面的边界。为避免外接圆把焊盘黑环、局部阴影或板边暗区计入孔径，程序还加入同图四孔半径一致性约束和异常半径保护。该过程只使用图像灰度与同类孔一致性，不使用 `3 mm` 孔径标准值作为检测或修正先验。

### 算法 B：强光斑感知改进

```text
PCB 图像
  → 畸变校正
  → 高光区域检测
  → HSV 绿色区域分割
  → 高光引导的 mask 区域分离
  → 常规候选四边形检测
  → row-interval 四边形恢复
  → 后续孔检测与几何测量
```

算法 B 的核心是处理强光斑与 PCB 区域在二值 mask 中粘连的问题。当常规连通域候选无法形成合法四边形时，`quad_recovery.py` 通过逐行扫描前景区间、拟合左右边界并恢复四边形外框，使后续测量流程能够继续执行。

## 核心算法模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 相机标定 | `src/calibration/calibrate_camera.py` | 棋盘格角点检测、相机内参和畸变系数估计 |
| 测量平面标定 | `src/calibration/measurement_plane.py` | 建立图像像素到毫米坐标的平面单应性 |
| 标准预处理 | `src/preprocessing/baseline.py` | 畸变校正、灰度化、滤波等基础处理 |
| 高光处理 | `src/preprocessing/highlight.py` | 高光区域检测和强光区域辅助处理 |
| mask 生成 | `src/preprocessing/masks.py` | PCB 绿色区域分割、高光 mask 后处理 |
| 外框检测 | `src/board_detection/board_detector.py` | PCB 四边形候选选择、角点排序、透视矫正 |
| 候选筛选 | `src/board_detection/candidate_filter.py` | 连通域面积、宽高比、边界和位置约束 |
| 区域分离 | `src/board_detection/region_separation.py` | 强光斑与 PCB mask 粘连分离 |
| 四边形恢复 | `src/board_detection/quad_recovery.py` | 从残损 mask 中恢复 PCB 四边形外框 |
| 孔检测 | `src/measurement/hole_detector.py` | 俯视图中检测四个圆形安装孔，并进行径向灰度半径精修 |
| 几何测量 | `src/measurement/geometry.py` | 板长宽、孔心、孔径、孔距和误差计算 |
| 实验执行 | `src/experiments/runner.py` | 批量运行算法 A/B 并输出 CSV |
| 统计报告 | `src/experiments/statistics.py` | 检测成功率、误差统计、分组报告 |
| 可视化报告 | `src/experiments/report.py` | mask、候选、检测叠加图输出 |

## 项目结构

```text
pcb_measurement_project/
├── data/
│   ├── calibration/          # 棋盘格标定图像
│   └── pcb/                  # PCB 待测图像
├── src/
│   ├── calibration/          # 相机标定与测量平面标定
│   ├── preprocessing/        # 标准预处理、高光检测、mask 生成
│   ├── board_detection/      # 外框检测、区域分离、四边形恢复
│   ├── measurement/          # 安装孔检测与几何测量
│   └── experiments/          # 实验配置、执行、统计和报告
├── outputs/
│   ├── calibration/          # 标定输出
│   ├── experiments/          # 中间实验结果
│   └── reports/              # CSV 和报告图
├── main.py
├── requirements.txt
└── README.md
```

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：

- `opencv-python >= 4.8.0`
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `matplotlib >= 3.7.0`

## 使用方法

### 1. 放置数据

```text
data/calibration/   # 棋盘格标定图像，例如 01.bmp, 02.bmp ...
data/pcb/           # PCB 待测图像，例如 01.bmp ... 06.bmp
```

### 2. 检查配置

主要配置位于 `src/config.py`：

| 参数 | 说明 |
|------|------|
| `CHESSBOARD_PATTERN_SIZE` | 棋盘格内角点数量，当前为 `(10, 10)` |
| `SQUARE_SIZE_MM` | 棋盘格单格边长，当前为 `10.0` |
| `BOARD_WIDTH_MM`, `BOARD_HEIGHT_MM` | PCB 外框标称尺寸，用于误差统计 |
| `HOLE_PITCH_X_MM`, `HOLE_PITCH_Y_MM` | 安装孔中心距标称值，用于误差统计 |

样本类型配置位于 `src/experiments/configs.py` 的 `SAMPLE_TYPE_MAP`。

### 3. 相机与测量平面标定

```bash
python main.py calibrate
```

该命令会生成相机参数和测量平面单应性。

### 4. 运行完整实验

```bash
python main.py compare
```

该命令会：

1. 运行算法 A 处理全部 PCB 图像；
2. 运行算法 B 处理 B 类强光斑邻接样本；
3. 生成测量 CSV、分组统计报告和可视化报告图。

### 5. 单独运行算法

```bash
python main.py algorithm-a
python main.py algorithm-b
```

## 输出结果

`outputs/reports/` 中包含：

| 文件/目录 | 说明 |
|-----------|------|
| `measurements_algorithm_a.csv` | 算法 A 逐图测量结果 |
| `measurements_algorithm_b.csv` | 算法 B 逐图测量结果 |
| `comparison_summary.csv` | B 类样本上的统计对比 |
| `01/` ... `06/` | 每张图像的原图、mask、候选、检测叠加和测量图 |

CSV 主要字段：

| 字段 | 说明 |
|------|------|
| `board_width_mm`, `board_height_mm` | 电路板外框实测长宽 |
| `board_width_error_mm`, `board_height_error_mm` | 板长宽误差 |
| `resolution_*_mm_per_px` | 测量图像空间分辨率，单位 mm/px |
| `resolution_*_px_per_mm` | 测量图像采样密度，单位 px/mm |
| `resolution_mean_um_per_px` | 平均分辨率，单位 um/px |
| `hole*_x_mm`, `hole*_y_mm` | 四个安装孔圆心毫米坐标 |
| `hole*_diameter_mm` | 四个安装孔直径 |
| `pitch_x_mm`, `pitch_y_mm` | X/Y 方向孔中心距 |
| `abs_error_x_mm`, `abs_error_y_mm` | X/Y 方向孔距误差 |
| `mean_abs_error_mm` | X/Y 平均绝对误差 |
| `num_components_before`, `num_components_after` | 区域分离前后连通域数量 |
| `quad_recovery_used` | 是否使用四边形恢复 |

## 当前实验结果

当前 6 张 PCB 图像的实验结果如下：

| 分组 | 算法 | 板检测 | 孔检测 | 平均孔距误差 |
|------|------|--------|--------|--------------|
| A 类 01-04 | 算法 A | 4/4 | 4/4 | `1.9465 mm ± 0.7815 mm` |
| B 类 05-06 | 算法 A | 0/2 | 0/2 | N/A |
| B 类 05-06 | 算法 B | 2/2 | 2/2 | `1.2017 mm ± 0.5935 mm` |

B 类强光斑样本中，算法 B 恢复了外框检测、孔心定位和孔距测量；孔径测量仍会受到局部反光、焊盘黑环和边缘阴影影响，因此 B 类孔径误差大于 A 类正常样本。

A 类样本孔径测量结果如下，标准孔径为 `3 mm`：

| 图像 | H1 | H2 | H3 | H4 |
|------|----|----|----|----|
| 01 | `3.037 mm` | `3.067 mm` | `2.424 mm` | `3.149 mm` |
| 02 | `2.939 mm` | `2.826 mm` | `2.903 mm` | `3.432 mm` |
| 03 | `2.580 mm` | `2.920 mm` | `3.083 mm` | `2.778 mm` |
| 04 | `3.070 mm` | `3.052 mm` | `3.360 mm` | `3.071 mm` |

A 类四孔平均直径：

| 孔位 | 平均直径 | 相对 3 mm 误差 |
|------|----------|----------------|
| H1 | `2.91 mm` | `0.09 mm` |
| H2 | `2.97 mm` | `0.03 mm` |
| H3 | `2.94 mm` | `0.06 mm` |
| H4 | `3.11 mm` | `0.11 mm` |

B 类样本孔径测量结果如下：

| 图像 | H1 | H2 | H3 | H4 |
|------|----|----|----|----|
| 05 | `3.626 mm` | `3.413 mm` | `3.411 mm` | `3.379 mm` |
| 06 | `4.020 mm` | `3.541 mm` | `3.572 mm` | `3.386 mm` |

B 类四孔平均直径：

| 孔位 | 平均直径 | 相对 3 mm 误差 |
|------|----------|----------------|
| H1 | `3.82 mm` | `0.82 mm` |
| H2 | `3.48 mm` | `0.48 mm` |
| H3 | `3.49 mm` | `0.49 mm` |
| H4 | `3.38 mm` | `0.38 mm` |

B 类逐图结果：

| 图像 | 算法 | 板长 | 板宽 | X 孔距 | Y 孔距 | 平均孔距误差 |
|------|------|------|------|--------|--------|--------------|
| 05 | 算法 B | `99.80 mm` | `101.33 mm` | `91.29 mm` | `94.54 mm` | `1.6213 mm` |
| 06 | 算法 B | `101.12 mm` | `101.33 mm` | `93.81 mm` | `92.63 mm` | `0.7820 mm` |

## 测量分辨率

系统在完成 PCB 外框检测和测量平面坐标转换后，根据外框实测尺寸与透视矫正图像尺寸计算当前测量分辨率：

```text
resolution_x_mm_per_px = board_width_mm / warped_width_px
resolution_y_mm_per_px = board_height_mm / warped_height_px
resolution_mean_mm_per_px = 平均(X, Y)
resolution_mean_px_per_mm = 1 / resolution_mean_mm_per_px
```

该指标用于描述当前实验条件下图像测量的空间采样能力，并随每张图像一起写入 CSV。

## 报告图建议

用于展示问题驱动和改进效果时，推荐使用以下图像：

| 图像 | 作用 |
|------|------|
| `outputs/reports/05/05_01_original.png` | B 类强光斑原图 |
| `outputs/reports/05/05_02_algo_a_mask.png` | 算法 A mask 失败现象 |
| `outputs/reports/05/05_02b_algo_a_largest_component.png` | 强光斑与 PCB 共同形成连通域 |
| `outputs/reports/05/05_04b_algo_b_candidates.png` | 算法 B 四边形恢复候选 |
| `outputs/reports/05/05_06b_algo_b_detection.png` | 算法 B 最终检测与测量叠加图 |

06 样本同理。

## 研究结论

1. 基础测量流程在 A 类样本上能够稳定完成 PCB 外框检测、安装孔检测和尺寸测量，孔径平均测量结果接近 `3 mm` 标准值。
2. B 类样本中，强光斑邻接会导致 PCB mask 与背景强光区域形成粘连连通域，使常规候选筛选无法得到合法外框。
3. 算法 B 通过高光感知区域分离和 row-interval 四边形恢复，使 B 类样本从外框检测失败恢复为端到端可测量。
4. 孔径测量采用圆候选检测、径向灰度边界精修和同图四孔一致性约束；在强光斑样本中仍存在残余孔径偏大，主要来自反光、焊盘黑环和阴影对孔边界的干扰。
5. 棋盘格测量平面标定为板长宽、孔心、孔径和孔距提供统一毫米坐标，便于输出完整尺寸测量结果和误差统计。
