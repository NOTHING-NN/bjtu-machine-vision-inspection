"""
src.board_detection — PCB 外框检测层

提供：
  - detect_board()          — 完整的 PCB 外框检测 + 透视矫正
  - find_board_contours()   — 候选四边形筛选
  - separate_highlight_from_mask() — 光斑与 PCB 粘连分离
  - order_corners() / warp_board() — 透视变换工具
"""

from src.board_detection.board_detector import detect_board
from src.board_detection.candidate_filter import find_board_contours
from src.board_detection.region_separation import separate_highlight_from_mask
from src.board_detection.perspective import order_corners, compute_homography, warp_board
