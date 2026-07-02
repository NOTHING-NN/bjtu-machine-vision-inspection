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
CHESSBOARD_PATTERN_SIZE = (10, 10)

# 棋盘格单格边长，单位 mm
SQUARE_SIZE_MM = 10.0

# 亚像素角点优化终止条件
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
# 孔心距板边 3 mm
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
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0

# --- 大尺度高斯模糊（用于光照估计） ---
LARGE_GAUSSIAN_KERNEL_SIZE = (101, 101)
LARGE_GAUSSIAN_SIGMA = 30.0

# --- CLAHE（自适应直方图均衡化） ---
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- Canny 边缘检测 ---
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150

# --- 二值化 ---
ADAPTIVE_THRESH_BLOCK_SIZE = 31
ADAPTIVE_THRESH_C = 5

# --- 高光抑制（光照矫正用） ---
# 白亮反光 = V 高 + S 低。同时满足以下条件判定为反光：
HIGHLIGHT_V_THRESH = 190            # V 通道绝对阈值
HIGHLIGHT_S_MAX = 45                # S 通道上限（低饱和度=白色/反光）
HIGHLIGHT_SUPPRESS_FACTOR = 0.35    # 反光区 V 压制系数（越低越强力）

# --- 暗区提升（光照矫正用） ---
SHADOW_V_MAX = 35                   # V 低于此值视为阴影，需提亮
SHADOW_BOOST_FACTOR = 0.7           # 暗区V不低于光照场的该比例

# --- ROI 参数 ---
# 在估计的安装孔位置周围开辟 ROI 区域
HOLE_ROI_SIZE_MM = 15.0             # ROI 半边长，单位 mm

# --- HoughCircles 参数 ---
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 5
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 5
HOUGH_MAX_RADIUS = 30

# --- HSV 绿色分割（PCB 基板提取） ---
HSV_GREEN_H_LOW = 35
HSV_GREEN_H_HIGH = 85
HSV_GREEN_S_LOW = 40
HSV_GREEN_V_LOW = 25
MORPH_CLOSE_KERNEL_SIZE = (7, 7)

# --- PCB 外框轮廓筛选 ---
BOARD_MIN_AREA_RATIO = 0.03
BOARD_MAX_AREA_RATIO = 0.25
BOARD_ASPECT_MIN = 0.75
BOARD_ASPECT_MAX = 1.33
BOUNDARY_MARGIN_PX = 20
BOARD_CENTER_MARGIN_RATIO = 0.15
BOARD_EPSILON_RATIO = 0.02

# ============================================================
# 5. 区域分离参数（光斑与 PCB 粘连分离）
# ============================================================

# 贴边连通域内高光面积占比阈值（超过此值判定为粘连）
HIGHLIGHT_OVERLAP_RATIO = 0.05

# 高光 mask 膨胀核大小（覆盖过渡带）
HIGHLIGHT_DILATE_KERNEL_SIZE = (9, 9)

# 分离后形态学开运算核大小（断开细连接）
REGION_SEPARATION_OPEN_KERNEL_SIZE = (5, 5)

# ============================================================
# 6. 样本光照类型映射
# ============================================================

SAMPLE_TYPE_MAP = {
    "01": "A",   # 板面反光
    "02": "A",
    "03": "A",
    "04": "A",
    "05": "B",   # 邻近强光斑
    "06": "B",
    "07": "B",
}

# ============================================================
# 7. 输出控制参数
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
# 8. 图像文件扩展名过滤
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
