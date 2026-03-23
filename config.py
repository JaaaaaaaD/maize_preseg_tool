# 配置文件

# 模型配置
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"  # 默认模型路径
SAM_MODEL_TYPE = "vit_b"  # 默认模型类型

# 图像处理配置
SNAP_RADIUS = 8  # 边缘吸附半径
REGION_GROWING_THRESHOLD = 30  # 区域生长颜色差异阈值

# 颜色范围配置（HSV）
LOWER_GREEN = [20, 20, 20]  # 主绿色范围下限
UPPER_GREEN = [100, 255, 255]  # 主绿色范围上限
LOWER_DARK = [0, 0, 0]  # 暗绿色和黑色范围下限
UPPER_DARK = [180, 255, 70]  # 暗绿色和黑色范围上限

# 边缘检测配置
CANNY_THRESHOLD1 = 25  # Canny边缘检测阈值1
CANNY_THRESHOLD2 = 70  # Canny边缘检测阈值2

# 形态学操作配置
KERNEL_DILATE = (3, 3)  # 膨胀操作核大小
KERNEL_CLOSE = (7, 7)  # 闭操作核大小
KERNEL_OPEN = (3, 3)  # 开操作核大小

# 锐化配置
UNSHARP_MASK_KERNEL = (3, 3)  # 锐化核大小
UNSHARP_MASK_SIGMA = 1.0  # 锐化sigma值
UNSHARP_MASK_AMOUNT = 0.8  # 锐化强度
UNSHARP_MASK_THRESHOLD = 5  # 锐化阈值

# 颜色变化检测配置
COLOR_CHANGE_THRESHOLD = 15  # 颜色变化阈值
ROI_SIZE = 8  # ROI区域大小

# 路径配置
ANNOTATION_DIR = "./maize_annotations/projects"  # 标注保存目录

# 快捷键配置
SHORTCUTS = {
    "SAVE_POLYGON": "Return",
    "SAVE_PLANT": "Shift+Return",
    "UNDO": "Ctrl+Z",
    "DELETE_PLANT": "Delete",
    "TOGGLE_EDGE_SNAP": "Shift",
    "LOAD_BATCH": "Ctrl+Shift+O",
    "PREV_IMAGE": "Left",
    "NEXT_IMAGE": "Right",
    "TOGGLE_SAM_SEGMENTATION": "S",
    "TOGGLE_REGION_GROWING": "G"
}

# 版本信息
VERSION = "batch_optimized_1.0"