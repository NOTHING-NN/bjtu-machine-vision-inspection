"""
config.py — 项目可调参数配置文件

所有路径、标定参数、预处理参数、输出控制参数集中管理，
便于后续调参和实验对比。
"""

from pathlib import Path

# ============================================================
# 1. 数据路径
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 电路板图像目录（待测 PCB 图像）
BOARD_IMAGE_DIR = PROJECT_ROOT / "data" / "board"

# 棋盘格标定图像目录
CHESSBOARD_IMAGE_DIR = PROJECT_ROOT / "data" / "chessboard"

# 输出根目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 各子输出目录
CALIB_OUTPUT_DIR = OUTPUT_DIR / "calibration"
BASELINE_OUTPUT_DIR = OUTPUT_DIR / "baseline"
LIGHT_CORRECTED_OUTPUT_DIR = OUTPUT_DIR / "light_corrected"
COMPARISON_OUTPUT_DIR = OUTPUT_DIR / "comparison"

# ============================================================
# 2. 标定参数
# ============================================================

# 棋盘格内角点数量（列数, 行数）
# ⚠️ 重要：是"内角点"数量，不是棋盘格方块数量！
# 例如：9列×9行棋盘格方块 → 内角点为 8×8 → 填 (8, 8)
#      10列×7行棋盘格方块 → 内角点为 9×6 → 填 (9, 6)
# 请根据你实际打印的棋盘格修改此参数！
CHESSBOARD_PATTERN_SIZE = (9, 9)

# 棋盘格单格边长，单位 mm
SQUARE_SIZE_MM = 10.0

# 亚像素角点优化终止条件
# 格式：(终止条件类型, 最大迭代次数, 精度)
# 在 calibrate_camera.py 中通过 calibrate_camera.get_subpix_criteria() 获取
SUBPIX_MAX_ITER = 30
SUBPIX_EPSILON = 0.001

# ============================================================
# 3. 电路板尺寸参数（标称值）
# ============================================================

# 电路板外框尺寸，单位 mm
BOARD_WIDTH_MM = 100.0
BOARD_HEIGHT_MM = 100.0

# 四个安装孔中心间距（理论值），单位 mm
HOLE_PITCH_X_MM = 94.0
HOLE_PITCH_Y_MM = 94.0

# 安装孔理论中心坐标（相对于电路板左上角），单位 mm
# 假设孔心距板边 3 mm
HOLE_OFFSET_MM = 3.0
EXPECTED_HOLE_POSITIONS_MM = [
    (HOLE_OFFSET_MM, HOLE_OFFSET_MM),                          # 左上
    (BOARD_WIDTH_MM - HOLE_OFFSET_MM, HOLE_OFFSET_MM),         # 右上
    (BOARD_WIDTH_MM - HOLE_OFFSET_MM, BOARD_HEIGHT_MM - HOLE_OFFSET_MM),  # 右下
    (HOLE_OFFSET_MM, BOARD_HEIGHT_MM - HOLE_OFFSET_MM),        # 左下
]

# ============================================================
# 4. 预处理参数
# ============================================================

# --- 高斯滤波 ---
GAUSSIAN_KERNEL_SIZE = (5, 5)       # 高斯核大小（宽, 高），必须为奇数
GAUSSIAN_SIGMA = 1.0                # 高斯核标准差

# --- 大尺度高斯模糊（用于光照估计） ---
LARGE_GAUSSIAN_KERNEL_SIZE = (101, 101)  # 大核尺寸（宽, 高），必须为奇数
LARGE_GAUSSIAN_SIGMA = 30.0              # 大核标准差

# --- CLAHE（自适应直方图均衡化） ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- Canny 边缘检测 ---
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150

# --- 二值化 ---
ADAPTIVE_THRESH_BLOCK_SIZE = 31     # 自适应阈值邻域大小，必须为奇数
ADAPTIVE_THRESH_C = 5               # 自适应阈值常数偏移

# --- 高亮抑制 ---
HIGHLIGHT_PERCENTILE = 98.0         # 高亮像素分位数阈值

# --- ROI 参数 ---
# 在估计的安装孔位置周围开辟 ROI 区域，尺寸单位为 mm
HOLE_ROI_SIZE_MM = 15.0             # ROI 半边长

# --- HoughCircles 参数 ---
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 5                  # 圆心最小距离（像素）
HOUGH_PARAM1 = 100                  # Canny 高阈值
HOUGH_PARAM2 = 30                   # 累加器阈值
HOUGH_MIN_RADIUS = 5
HOUGH_MAX_RADIUS = 30

# --- 轮廓筛选 ---
BOARD_MIN_AREA_RATIO = 0.3          # 电路板轮廓最小面积占比（相对图像面积）
BOARD_EPSILON_RATIO = 0.02          # approxPolyDP 逼近精度系数

# ============================================================
# 5. 输出控制参数
# ============================================================

# 是否保存中间处理步骤图像
SAVE_INTERMEDIATE_IMAGES = True

# 是否在屏幕上显示调试图像（交互式运行时）
SHOW_DEBUG_IMAGES = False

# 可视化图像 DPI
FIGURE_DPI = 150

# 输出图像格式
OUTPUT_IMAGE_EXT = ".png"

# ============================================================
# 6. 图像文件扩展名过滤
# ============================================================

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def ensure_output_dirs() -> None:
    """确保所有输出目录存在。"""
    from pathlib import Path
    dirs = [
        CALIB_OUTPUT_DIR,
        BASELINE_OUTPUT_DIR,
        LIGHT_CORRECTED_OUTPUT_DIR,
        COMPARISON_OUTPUT_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
