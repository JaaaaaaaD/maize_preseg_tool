# 单张图片的项目级预标注推理服务
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from config import (
    INFERENCE_CONFIDENCE_THRESHOLD,
    INFERENCE_MAX_CANDIDATES,
    INFERENCE_MIN_AREA,
    INFERENCE_POLYGON_EPSILON_RATIO,
)
from utils.annotation_schema import make_candidate_instance
from utils.helpers import calculate_polygon_area


def _mask_to_polygons(mask, epsilon_ratio):
    """将二值 mask 转成 polygon 列表。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 1:
            continue
        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = []
        for point in approx:
            polygon.append((float(point[0][0]), float(point[0][1])))
        if len(polygon) >= 3:
            if polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
            polygons.append(polygon)
    return polygons


class InferenceService:
    """加载 active YOLO segmentation 模型并输出候选层实例。"""

    @staticmethod
    def predict_candidates(
        model_path,
        image_path,
        class_names,
        confidence_threshold=INFERENCE_CONFIDENCE_THRESHOLD,
        min_area=INFERENCE_MIN_AREA,
        epsilon_ratio=INFERENCE_POLYGON_EPSILON_RATIO,
    ):
        """对单张图片推理，返回候选实例列表。"""
        if not model_path:
            return []

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("未安装 ultralytics，无法执行预标注推理") from error

        try:
            model = YOLO(model_path)
        except Exception as error:
            raise RuntimeError(f"加载 YOLO 模型失败: {error}") from error

        try:
            results = model.predict(
                source=image_path,
                conf=confidence_threshold,
                verbose=False,
                task="segment",
            )
        except Exception as error:
            raise RuntimeError(f"执行 YOLO 推理失败: {error}") from error

        if not results:
            return []

        try:
            result = results[0]
            if result.masks is None or result.boxes is None:
                return []

            boxes = result.boxes
            masks = result.masks.data
            candidates = []
            max_count = min(len(masks), INFERENCE_MAX_CANDIDATES)

            for index in range(max_count):
                try:
                    confidence = float(boxes.conf[index].item()) if boxes.conf is not None else None
                    class_id = int(boxes.cls[index].item()) if boxes.cls is not None else 0

                    mask = masks[index].detach().cpu().numpy().astype(np.uint8) * 255
                    polygons = _mask_to_polygons(mask, epsilon_ratio)
                    polygons = [polygon for polygon in polygons if calculate_polygon_area(polygon) >= min_area]
                    if not polygons:
                        continue

                    total_area = sum(calculate_polygon_area(polygon) for polygon in polygons)
                    if total_area < min_area:
                        continue

                    candidates.append(
                        make_candidate_instance(
                            candidate_id=f"cand_{index + 1:04d}",
                            polygons=polygons,
                            class_names=class_names,
                            class_id=class_id,
                            confidence=confidence,
                            model_version=None,
                        )
                    )
                except Exception as error:
                    # 单个实例处理失败，跳过继续处理其他实例
                    continue

            return candidates
        except Exception as error:
            raise RuntimeError(f"处理 YOLO 推理结果失败: {error}") from error


class InferenceWorker(QThread):
    """放在后台线程执行单张推理，避免切图阻塞 GUI。"""

    inference_succeeded = pyqtSignal(object, str)
    inference_failed = pyqtSignal(str, str)

    def __init__(self, model_path, image_path, class_names, model_version=None):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.class_names = list(class_names or [])
        self.model_version = model_version

    def run(self):
        try:
            candidates = InferenceService.predict_candidates(
                self.model_path,
                self.image_path,
                self.class_names,
            )
            for candidate in candidates:
                candidate["model_version"] = self.model_version
            self.inference_succeeded.emit(candidates, self.image_path)
        except Exception as error:
            traceback.print_exc()
            self.inference_failed.emit(str(error), self.image_path)