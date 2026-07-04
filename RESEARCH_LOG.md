# 研究记录

## 项目概述

本项目完成一个面向 PCB 电路板的二维图像测量系统。系统使用 A4 打印黑白棋盘格进行相机标定和测量平面标定，棋盘格单格边长为 `10 mm`。标定过程仅进行灰度化、棋盘格角点检测和亚像素优化，随后建立图像像素到测量平面毫米坐标的单应性。

待测对象为同一块 PCB，实验图像共 6 张。系统完成畸变校正、PCB 外框检测、四角点定位、透视矫正、安装孔定位、孔心坐标测量、孔径测量、孔距计算、板长宽测量和误差统计。

实验重点是分析非均匀光照和局部强反光条件下，基础测量流程与强光斑感知改进流程的检测稳定性。样本按光照干扰形态分为：

| 类型 | 样本 | 特征 |
|------|------|------|
| A 类 | 01, 02, 03, 04 | PCB 板面存在局部反光，背景强光斑干扰较轻 |
| B 类 | 05, 06 | 强光斑紧贴 PCB 边缘，容易与 PCB 区域粘连 |

## 系统架构

项目采用五层结构：

```text
src/
├── calibration/          # 相机标定与测量平面标定
├── preprocessing/        # 标准预处理、高光检测、mask 生成
├── board_detection/      # PCB 外框检测、区域分离、四边形恢复
├── measurement/          # 安装孔检测与几何测量
└── experiments/          # 实验配置、批量执行、统计和报告
```

核心文件职责如下：

| 文件 | 职责 |
|------|------|
| `src/calibration/calibrate_camera.py` | 棋盘格相机标定，输出内参和畸变系数 |
| `src/calibration/measurement_plane.py` | 建立测量平面毫米坐标单应性 |
| `src/preprocessing/baseline.py` | 标准预处理 |
| `src/preprocessing/highlight.py` | 高光区域检测与辅助处理 |
| `src/preprocessing/masks.py` | PCB 绿色 mask、高光 mask 和后处理 |
| `src/board_detection/candidate_filter.py` | PCB 连通域候选筛选与打分 |
| `src/board_detection/region_separation.py` | 强光斑与 PCB mask 粘连分离 |
| `src/board_detection/quad_recovery.py` | row-interval 四边形恢复 |
| `src/board_detection/board_detector.py` | 外框检测、角点排序、透视矫正 |
| `src/measurement/hole_detector.py` | 俯视图四个圆形安装孔检测 |
| `src/measurement/geometry.py` | 板长宽、孔心、孔径、孔距和误差计算 |
| `src/experiments/runner.py` | 算法 A/B 统一执行引擎 |
| `src/experiments/statistics.py` | 检测成功率和误差统计 |
| `src/experiments/report.py` | 可视化报告图生成 |

## 标定方法

### 相机标定

标定板为黑白棋盘格，内角点数量为 `(10, 10)`，单格边长为 `10 mm`。流程如下：

```text
棋盘格图像
  → 灰度化
  → cv2.findChessboardCorners
  → cv2.cornerSubPix
  → cv2.calibrateCamera
  → 保存 camera_matrix 和 dist_coeffs
```

当前标定结果：

| 指标 | 结果 |
|------|------|
| 标定图像数量 | 17 |
| 成功检测角点图像 | 16 |
| RMS 重投影误差 | `0.9968 px` |
| 输出文件 | `outputs/calibration/camera_params.npz` |

### 测量平面标定

在去畸变后的棋盘格图像中，将棋盘格角点像素坐标与对应毫米坐标建立平面单应性：

```text
image_points_px ↔ world_points_mm
```

输出文件为：

```text
outputs/calibration/measurement_plane_homography.npz
```

后续 PCB 四角点、孔心和孔径测量均统一映射到该毫米坐标系中。

## 算法 A：基础测量流程

算法 A 用于全部样本，作为常规图像测量流程：

```text
原图
  → 畸变校正
  → HSV 绿色区域分割
  → 形态学后处理
  → 连通域候选筛选
  → 四角点定位
  → 透视矫正
  → 四个圆形安装孔检测
  → 毫米坐标测量与误差统计
```

### PCB 外框候选筛选

候选筛选主要依据：

| 条件 | 说明 |
|------|------|
| 面积占比 | 排除过小噪声和过大背景连通域 |
| 宽高比 | 排除明显非板状区域 |
| 边界距离 | 排除贴边干扰连通域 |
| 中心位置 | 排除明显偏离主体区域的候选 |
| 多边形逼近 | 获取四边形角点 |
| 综合评分 | 按形状质量和面积合理性选择最佳候选 |

### 安装孔检测

安装孔检测在透视矫正后的 PCB 俯视图中进行：

```text
俯视灰度图
  → CLAHE 增强
  → 多源暗孔二值化
  → 轮廓圆度筛选
  → HoughCircles 候选补充
  → 去重
  → 四象限拓扑选择
  → 输出四个孔心和半径
```

孔心和孔半径随后反投影回原图坐标，再通过测量平面单应性转换为毫米坐标。孔径通过圆周采样点在毫米平面上的平均半径计算。

## 算法 B：强光斑感知改进流程

算法 B 用于 B 类强光斑邻接样本。其目标是在强光斑与 PCB 区域粘连时恢复 PCB 外框检测。

```text
原图
  → 畸变校正
  → 高光区域检测
  → HSV 绿色区域分割
  → 高光引导 mask 区域分离
  → 常规候选四边形检测
  → row-interval 四边形恢复
  → 透视矫正
  → 四孔检测
  → 毫米坐标测量与误差统计
```

### 高光区域检测

高光区域基于 HSV 空间识别：

```text
V 通道较高 + S 通道较低 → 白亮反光区域
```

高光 mask 用于分析 PCB mask 中可能存在的强光斑粘连区域。

### 区域分离

`region_separation.py` 对 PCB green mask 执行连通域分析：

1. 检查贴近图像边界的大连通域；
2. 计算该连通域与高光 mask 的重叠比例；
3. 对强光重叠区域进行膨胀；
4. 从 PCB mask 中切除高光粘连部分；
5. 通过形态学开运算断开残余细连接。

### 四边形恢复

当常规候选四边形检测无法从残损 mask 中得到合法候选时，`quad_recovery.py` 通过逐行扫描恢复 PCB 外框：

```text
输入: PCB mask
  → 逐行提取前景区间
  → 寻找连续稳定的区间 run
  → 拟合左边界和右边界
  → 由上下端点生成四边形
  → 注入 detect_board() 作为候选
```

四边形恢复只依赖目标外框可表示为四边形这一几何约束，适合强光斑造成的局部 mask 缺损场景。

## 几何测量

几何测量由 `src/measurement/geometry.py` 完成。

### 板长宽测量

PCB 外框检测得到四角点：

```text
tl, tr, br, bl
```

四角点经测量平面单应性转换为毫米坐标后，计算：

```text
上边长度 = distance(tl, tr)
下边长度 = distance(bl, br)
左边长度 = distance(tl, bl)
右边长度 = distance(tr, br)

board_width_mm  = (上边长度 + 下边长度) / 2
board_height_mm = (左边长度 + 右边长度) / 2
```

### 孔心测量

安装孔在俯视图中检测后，通过外框透视变换的逆矩阵反投影回去畸变图像坐标，再映射到测量平面毫米坐标。

输出字段：

```text
hole1_x_mm, hole1_y_mm
hole2_x_mm, hole2_y_mm
hole3_x_mm, hole3_y_mm
hole4_x_mm, hole4_y_mm
```

### 孔径测量

对于每个检测到的圆孔，在俯视图圆周上采样多个点：

```text
center_warped, radius_warped
  → circle samples in warped image
  → inverse homography to image pixels
  → image_to_world homography to mm plane
  → mean radius in mm
  → diameter_mm = 2 × mean_radius_mm
```

输出字段：

```text
hole1_diameter_mm
hole2_diameter_mm
hole3_diameter_mm
hole4_diameter_mm
```

### 分辨率测量

系统同时计算每张图像在透视矫正测量图中的空间分辨率。首先由测量平面单应性得到 PCB 外框的实测毫米尺寸，再结合俯视图像素尺寸计算：

```text
resolution_x_mm_per_px = board_width_mm / warped_width_px
resolution_y_mm_per_px = board_height_mm / warped_height_px
resolution_mean_mm_per_px = 平均(X, Y)
resolution_mean_um_per_px = resolution_mean_mm_per_px × 1000
resolution_mean_px_per_mm = 1 / resolution_mean_mm_per_px
```

输出字段：

```text
resolution_x_mm_per_px
resolution_y_mm_per_px
resolution_mean_mm_per_px
resolution_x_px_per_mm
resolution_y_px_per_mm
resolution_mean_px_per_mm
resolution_mean_um_per_px
```

### 孔距和误差统计

孔心按左上、右上、右下、左下排序，计算：

```text
pitch_x_mm = 平均(上边两孔距离, 下边两孔距离)
pitch_y_mm = 平均(左边两孔距离, 右边两孔距离)
```

误差统计字段：

```text
abs_error_x_mm
abs_error_y_mm
rel_error_x_pct
rel_error_y_pct
mean_abs_error_mm
```

## 实验配置

样本分类配置位于 `src/experiments/configs.py`：

```python
SAMPLE_TYPE_MAP = {
    "01": "A",
    "02": "A",
    "03": "A",
    "04": "A",
    "05": "B",
    "06": "B",
}
```

算法配置：

| 算法 | preprocessing | board_mask | region_separation |
|------|---------------|------------|-------------------|
| algorithm_a | `standard` | `hsv_green` | `False` |
| algorithm_b | `highlight_aware` | `highlight_aware_green` | `True` |

## 当前实验结果

### 算法 A 在 A/B 两类样本上的结果

| 类型 | 样本数 | 板检测 | 孔检测 | 端到端测量 | 平均孔距误差 |
|------|--------|--------|--------|------------|--------------|
| A 类 | 4 | 4/4 | 4/4 | 4/4 | `1.9465 mm ± 0.7815 mm` |
| B 类 | 2 | 0/2 | 0/2 | 0/2 | N/A |

A 类逐图结果：

| 图像 | 板长 | 板宽 | X 孔距 | Y 孔距 | 平均孔距误差 |
|------|------|------|--------|--------|--------------|
| 01 | `98.69 mm` | `98.55 mm` | `92.34 mm` | `91.34 mm` | `2.1608 mm` |
| 02 | `99.02 mm` | `98.46 mm` | `91.02 mm` | `94.17 mm` | `1.5782 mm` |
| 03 | `98.72 mm` | `98.98 mm` | `93.57 mm` | `88.57 mm` | `2.9288 mm` |
| 04 | `99.02 mm` | `98.78 mm` | `92.23 mm` | `93.53 mm` | `1.1183 mm` |

### 算法 A 与算法 B 在 B 类样本上的结果

| 图像 | 算法 A 板检测 | 算法 A 孔检测 | 算法 B 板检测 | 算法 B 孔检测 | 算法 B 平均孔距误差 |
|------|:-------------:|:-------------:|:-------------:|:-------------:|--------------------|
| 05 | 失败 | 失败 | 成功 | 成功 | `1.6213 mm` |
| 06 | 失败 | 失败 | 成功 | 成功 | `0.7820 mm` |

B 类算法 B 逐图测量：

| 图像 | 板长 | 板宽 | X 孔距 | Y 孔距 | 平均孔距误差 | 四边形恢复 |
|------|------|------|--------|--------|--------------|------------|
| 05 | `99.80 mm` | `101.33 mm` | `91.29 mm` | `94.54 mm` | `1.6213 mm` | 是 |
| 06 | `101.12 mm` | `101.33 mm` | `93.81 mm` | `92.63 mm` | `0.7820 mm` | 是 |

### 分组统计

| 指标 | 算法 A A 类 | 算法 A B 类 | 算法 B B 类 |
|------|-------------|-------------|-------------|
| 板检测成功率 | 100% | 0% | 100% |
| 孔检测成功率 | 100% | 0% | 100% |
| 端到端成功率 | 100% | 0% | 100% |
| 平均孔距误差 | `1.9465 mm` | N/A | `1.2017 mm` |
| 误差标准差 | `0.7815 mm` | N/A | `0.5935 mm` |

## 可视化输出

报告图保存在：

```text
outputs/reports/<image_name>/
```

推荐用于报告展示的图像：

| 文件 | 说明 |
|------|------|
| `05_01_original.png` | B 类强光斑原图 |
| `05_02_algo_a_mask.png` | 算法 A 的 PCB mask |
| `05_02b_algo_a_largest_component.png` | 算法 A 中强光斑与 PCB 形成的连通域 |
| `05_04b_algo_b_candidates.png` | 算法 B 四边形恢复候选 |
| `05_05b_algo_b_warped.png` | 算法 B 透视矫正俯视图 |
| `05_06b_algo_b_detection.png` | 算法 B 最终孔检测和测量叠加图 |

06 样本可使用对应同名报告图。

## 研究结论

1. 基础测量流程在 A 类样本中能够稳定完成 PCB 外框检测、四孔检测和尺寸测量。
2. B 类样本中，强光斑紧贴 PCB 边缘会使 HSV green mask 中的 PCB 区域与背景强光区域形成粘连连通域，导致常规候选筛选无法得到合法 PCB 外框。
3. 算法 B 通过高光感知区域分离和 row-interval 四边形恢复，能够从强光斑邻接样本中恢复 PCB 四边形外框，并完成端到端测量。
4. 测量平面单应性将 PCB 四角点、安装孔圆心和孔径统一到毫米坐标系，使系统能够输出板长宽、孔心、孔径和孔距等完整二维尺寸信息。
5. 当前实验中，算法 B 在 B 类样本上实现 `2/2` 板检测成功、`2/2` 孔检测成功，平均孔距误差为 `1.2017 mm ± 0.5935 mm`。

## 当前状态

| 模块 | 状态 |
|------|:----:|
| 相机标定 | 完成 |
| 测量平面标定 | 完成 |
| 算法 A 基础测量流程 | 完成 |
| 算法 B 强光斑感知改进流程 | 完成 |
| 板长宽测量 | 完成 |
| 测量分辨率计算 | 完成 |
| 四孔圆心坐标测量 | 完成 |
| 四孔直径测量 | 完成 |
| 孔距与误差统计 | 完成 |
| CSV 输出 | 完成 |
| 报告图输出 | 完成 |
