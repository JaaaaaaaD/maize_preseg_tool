# 配置文件
import os

# 模型配置
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth"  # 默认模型路径
SAM_MODEL_TYPE = "vit_b"  # 默认模型类型
YOLO_DEFAULT_MODEL = "yolo11n-seg.pt"
YOLO_DEFAULT_MODEL_FAMILY = "yolo11n-seg"

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
BASE_WORK_DIR = "./maize_annotations"
ANNOTATION_DIR = os.path.join(BASE_WORK_DIR, "projects")  # 单图标注保存目录
PROJECTS_ROOT = os.path.join(BASE_WORK_DIR, "project_state")  # 项目根目录

# 项目级配置
DEFAULT_CLASS_NAMES = ["plant", "stem", "leaf", "ear"]
DEFAULT_CLASS_ID = 0
AUTO_TRAIN_THRESHOLD = 5
FIXED_VAL_RATIO = 0.2
FIXED_VAL_MIN_COUNT = 1
FIXED_VAL_SEED = 20260324

# 推理配置
INFERENCE_CONFIDENCE_THRESHOLD = 0.55
INFERENCE_MIN_AREA = 80.0
INFERENCE_POLYGON_EPSILON_RATIO = 0.005
INFERENCE_MAX_CANDIDATES = 256

# 训练配置
TRAIN_EPOCHS = 50
TRAIN_IMGSZ = 1024
TRAIN_BATCH = 4
TRAIN_DEVICE = "auto"  # 可选: auto / cpu / 0 / cuda:0
TRAIN_WORKERS = 2

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
VERSION = "project_loop_2.0"
