import sys
import json
import os
import traceback
import copy
from datetime import datetime
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                             QLabel, QMessageBox, QSizePolicy, QShortcut, QDialog, QGroupBox, QProgressDialog)
from PyQt5.QtCore import Qt, QPoint, QTimer, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QKeySequence, QCursor
from PIL import Image as PILImage

# 尝试导入SAM相关库
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Warning: SAM library not found. Please install segment-anything package.")
    torch = None
    SamPredictor = None

# 全局异常捕获
sys.excepthook = lambda exctype, value, tb: traceback.print_exception(exctype, value, tb)


class ImageLabel(QLabel):
    """图像显示与标注控件（边缘吸附版，无核心修改）"""

    def __init__(self, is_summary=False, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")

        self.raw_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        self.view_center_x = 0.0
        self.view_center_y = 0.0
        self.is_dragging = False
        self.drag_last_pos = QPoint()

        self.plants = []
        self.current_points = []
        self.current_plant_polygons = []
        self.current_plant_id = 1
        self.selected_plant_id = None

        self.is_summary = is_summary
        self.setMouseTracking(True)
        # 设置焦点策略，确保能够接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

        self.edge_snap_enabled = True
        self.snap_radius = 8
        self.foreground_mask = None
        self.edge_map = None
        self.current_snap_point = None
        self.color_image = None  # 存储原始颜色图像数据

        # ==================== 新增：SAM分割相关属性 ====================
        self.sam_segmenting = False  # 是否处于SAM分割模式
        self.sam_predictor = None  # SAM预测器
        self.sam_prompt_points = []  # SAM提示点
        self.sam_mask = None  # SAM分割结果
        # =============================================================

    def set_image(self, pil_image, preprocessed_data=None):
        """
        新增：支持传入预处理数据，避免重复计算
        :param preprocessed_data: (foreground_mask, edge_map) 元组
        """
        if pil_image is None:
            return
        try:
            self.plants = []
            self.current_points = []
            self.current_plant_polygons = []
            self.current_plant_id = 1
            self.selected_plant_id = None
            self.scale_factor = 1.0
            self.current_snap_point = None

            self.view_center_x = pil_image.width / 2.0
            self.view_center_y = pil_image.height / 2.0

            data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
            self.raw_pixmap = QPixmap.fromImage(qimage)

            # 保存原始颜色图像数据
            self.color_image = np.array(pil_image.convert("RGB"))

            # 优先使用传入的预处理数据，否则重新计算
            if preprocessed_data:
                self.foreground_mask, self.edge_map = preprocessed_data
            else:
                self.preprocess_image(pil_image)

            self.update_display()
        except Exception as e:
            print(f"set_image error: {e}")
            traceback.print_exc()

    def preprocess_image(self, pil_image):
        try:
            img_np = np.array(pil_image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 1. 扩展HSV颜色范围，包含更多的绿色和暗色
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # 主绿色范围 - 进一步放宽
            lower_green = np.array([20, 20, 20])
            upper_green = np.array([100, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # 暗绿色和黑色范围 - 进一步放宽
            lower_dark = np.array([0, 0, 0])  # 包含纯黑色
            upper_dark = np.array([180, 255, 70])  # 进一步降低亮度上限
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

            # 合并掩码
            self.foreground_mask = cv2.bitwise_or(mask_green, mask_dark)

            # 2. 增强形态学操作，填充过渡区域
            # 先进行膨胀操作，填充小的孔洞和过渡区域
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.foreground_mask = cv2.morphologyEx(self.foreground_mask, cv2.MORPH_DILATE, kernel_dilate)
            # 再进行闭操作，连接相邻区域
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 增大闭操作核大小
            self.foreground_mask = cv2.morphologyEx(self.foreground_mask, cv2.MORPH_CLOSE, kernel_close)
            # 最后进行开操作，去除噪声
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.foreground_mask = cv2.morphologyEx(self.foreground_mask, cv2.MORPH_OPEN, kernel_open)

            # 3. 多通道边缘检测，特别处理过渡像素
            r, g, b = cv2.split(img_bgr)

            # 定义锐化函数，使用反锐化遮罩技术
            def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
                """应用反锐化遮罩增强图像边界"""
                blurred = cv2.GaussianBlur(image, kernel_size, sigma)
                sharpened = float(amount + 1) * image - float(amount) * blurred
                sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
                sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
                sharpened = sharpened.round().astype(np.uint8)
                if threshold > 0:
                    low_contrast_mask = np.absolute(image - blurred) < threshold
                    np.copyto(sharpened, image, where=low_contrast_mask)
                return sharpened

            # 对每个通道应用适度的锐化处理，避免放大噪声
            # 对绿色通道应用锐化
            sharpened_g = unsharp_mask(g, kernel_size=(3, 3), sigma=1.0, amount=0.8, threshold=5)  # 降低锐化强度，增加阈值
            blurred_g = cv2.GaussianBlur(sharpened_g, (5, 5), 1.2)
            edges_g = cv2.Canny(blurred_g, 30, 80)  # 提高Canny阈值

            # 对红色通道应用锐化
            sharpened_r = unsharp_mask(r, kernel_size=(3, 3), sigma=1.0, amount=0.8, threshold=5)  # 降低锐化强度，增加阈值
            blurred_r = cv2.GaussianBlur(sharpened_r, (5, 5), 1.2)
            edges_r = cv2.Canny(blurred_r, 35, 90)  # 提高Canny阈值

            # 对蓝色通道应用锐化
            sharpened_b = unsharp_mask(b, kernel_size=(3, 3), sigma=1.0, amount=0.8, threshold=5)  # 降低锐化强度，增加阈值
            blurred_b = cv2.GaussianBlur(sharpened_b, (5, 5), 1.2)
            edges_b = cv2.Canny(blurred_b, 35, 90)  # 提高Canny阈值

            # 4. 基于亮度的边缘检测，特别针对暗区域
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # 对灰度图像应用适度锐化
            sharpened_gray = unsharp_mask(gray, kernel_size=(3, 3), sigma=1.0, amount=0.8, threshold=5)  # 降低锐化强度，增加阈值
            blurred_gray = cv2.GaussianBlur(sharpened_gray, (5, 5), 1.2)
            # 提高阈值，减少噪声
            edges_gray = cv2.Canny(blurred_gray, 25, 70)  # 提高Canny阈值

            # 5. 合并所有边缘
            edges = cv2.bitwise_or(edges_g, edges_r)
            edges = cv2.bitwise_or(edges, edges_b)
            edges = cv2.bitwise_or(edges, edges_gray)

            # 6. 增强边缘，处理不连续边缘，同时过滤噪声
            # 先进行闭操作连接不连续的边缘
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 减小闭操作核大小
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

            # 进行开操作去除小的噪声点
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)

            # 适度膨胀操作增强边缘
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel_dilate)

            # 7. 使用前景掩码过滤边缘
            self.edge_map = cv2.bitwise_and(edges, edges, mask=self.foreground_mask)

        except Exception as e:
            print(f"preprocess_image error: {e}")
            self.foreground_mask = None
            self.edge_map = None

    def calculate_snap_point(self, screen_pos):
        if not self.edge_snap_enabled or self.edge_map is None or self.raw_pixmap is None or self.color_image is None:
            return None

        try:
            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor
            img_x = (screen_pos.x() - offset_x) / self.scale_factor
            img_y = (screen_pos.y() - offset_y) / self.scale_factor

            img_h, img_w = self.edge_map.shape
            if not (0 <= img_x < img_w and 0 <= img_y < img_h):
                return None

            # 计算8*8的ROI区域
            roi_size = 8
            x1 = max(0, int(img_x - roi_size // 2))
            y1 = max(0, int(img_y - roi_size // 2))
            x2 = min(img_w, x1 + roi_size)
            y2 = min(img_h, y1 + roi_size)

            # 计算ROI内的颜色变化程度
            roi_color = self.color_image[y1:y2, x1:x2]
            # 计算颜色标准差，衡量颜色变化程度
            color_std = np.std(roi_color)
            # 设置颜色变化阈值，如果颜色变化太小，不认为是边界
            color_change_threshold = 15
            if color_std < color_change_threshold:
                return None

            # 计算ROI内占比最多的颜色作为背景色
            # 将ROI内的像素转换为一维数组
            roi_pixels = roi_color.reshape(-1, 3)
            # 计算每种颜色的出现次数
            unique_colors, counts = np.unique(roi_pixels, axis=0, return_counts=True)
            # 找到出现次数最多的颜色
            background_color = unique_colors[np.argmax(counts)]

            # 恢复使用snap_radius计算边缘检测的ROI
            x1_edge = max(0, int(img_x - self.snap_radius))
            y1_edge = max(0, int(img_y - self.snap_radius))
            x2_edge = min(img_w, int(img_x + self.snap_radius + 1))
            y2_edge = min(img_h, int(img_y + self.snap_radius + 1))

            roi_edges = self.edge_map[y1_edge:y2_edge, x1_edge:x2_edge]
            edge_points = np.column_stack(np.where(roi_edges > 0))

            if len(edge_points) == 0:
                return None

            # 筛选满足条件的边缘点
            valid_edge_points = []
            color_threshold = 30  # 颜色相似度阈值
            line_search_length = 10  # 直线搜索长度

            for point in edge_points:
                py, px = point
                abs_x = x1_edge + px
                abs_y = y1_edge + py

                # 检查该点是否在颜色图像范围内
                if 0 <= abs_x < self.color_image.shape[1] and 0 <= abs_y < self.color_image.shape[0]:
                    # 获取当前点的颜色
                    current_color = self.color_image[abs_y, abs_x]

                    # 检查当前点是否为背景色，如果是则跳过
                    current_color_diff = np.linalg.norm(current_color - background_color)
                    if current_color_diff < color_threshold:
                        continue

                    # 检查多个方向的直线上是否存在颜色相近的像素（非背景色）
                    # 包括上下左右四个方向，以及45度和30度的方向
                    directions = [
                        (0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
                        (1, 1), (1, -1), (-1, 1), (-1, -1),  # 45度方向
                        (1, 2), (2, 1), (-1, 2), (2, -1), (1, -2), (-2, 1), (-1, -2), (-2, -1)  # 30度方向
                    ]
                    has_similar_color = False

                    for dx, dy in directions:
                        # 计算该方向上的最大步长，确保不超出ROI范围
                        max_step = min(line_search_length,
                                       (x2_edge - abs_x) // abs(dx) if dx != 0 else line_search_length,
                                       (y2_edge - abs_y) // abs(dy) if dy != 0 else line_search_length,
                                       (abs_x - x1_edge) // abs(dx) if dx != 0 else line_search_length,
                                       (abs_y - y1_edge) // abs(dy) if dy != 0 else line_search_length)
                        max_step = max(1, max_step)  # 确保至少搜索1步

                        for step in range(1, max_step + 1):
                            nx = abs_x + dx * step
                            ny = abs_y + dy * step

                            # 检查新位置是否在图像范围内和ROI范围内
                            if (0 <= nx < self.color_image.shape[1] and
                                    0 <= ny < self.color_image.shape[0] and
                                    x1_edge <= nx < x2_edge and
                                    y1_edge <= ny < y2_edge):
                                # 计算颜色差异
                                neighbor_color = self.color_image[ny, nx]
                                # 检查邻居像素是否为背景色
                                neighbor_bg_diff = np.linalg.norm(neighbor_color - background_color)
                                if neighbor_bg_diff < color_threshold:
                                    continue  # 跳过背景色像素
                                # 计算与当前点的颜色差异
                                color_diff = np.linalg.norm(current_color - neighbor_color)

                                if color_diff < color_threshold:
                                    has_similar_color = True
                                    break
                            else:
                                break  # 超出范围，停止该方向的搜索

                        if has_similar_color:
                            break

                    if has_similar_color:
                        valid_edge_points.append((px, py))

            if len(valid_edge_points) == 0:
                return None

            # 计算有效边缘点与鼠标位置的距离
            edge_points_roi = np.array(valid_edge_points)
            mouse_roi = np.array([img_x - x1_edge, img_y - y1_edge])
            distances = np.linalg.norm(edge_points_roi - mouse_roi, axis=1)

            # 选择距离最近的点
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist <= self.snap_radius:
                snap_x = x1_edge + edge_points_roi[min_idx][0]
                snap_y = y1_edge + edge_points_roi[min_idx][1]
                return (float(snap_x), float(snap_y))
            else:
                return None

        except Exception as e:
            print(f"calculate_snap_point error: {e}")
            return None

    def update_display(self):
        if self.raw_pixmap is None:
            self.clear()
            return
        try:
            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            scaled_width = int(img_width * self.scale_factor)
            scaled_height = int(img_height * self.scale_factor)

            scaled_pixmap = self.raw_pixmap.scaled(
                scaled_width, scaled_height,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor

            final_pixmap = QPixmap(self.size())
            final_pixmap.fill(QColor(240, 240, 240))
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            painter.drawPixmap(int(offset_x), int(offset_y), scaled_pixmap)

            def img_to_screen(pt):
                return QPoint(
                    int(pt[0] * self.scale_factor + offset_x),
                    int(pt[1] * self.scale_factor + offset_y)
                )

            if self.is_summary:
                for plant in self.plants:
                    plant_color = QColor(*plant["color"])
                    for polygon_points in plant["polygons"]:
                        if len(polygon_points) < 3:
                            continue
                        qpts = [img_to_screen(p) for p in polygon_points]
                        painter.setBrush(QBrush(plant_color))
                        if plant["id"] == self.selected_plant_id:
                            painter.setPen(QPen(QColor(255, 0, 0), 3))
                        else:
                            painter.setPen(QPen(QColor(0, 0, 0), 1))
                        painter.drawPolygon(*qpts)
            else:
                for plant in self.plants:
                    plant_color = QColor(*plant["color"])
                    weak_color = QColor(plant_color.red(), plant_color.green(), plant_color.blue(), 40)
                    for polygon_points in plant["polygons"]:
                        if len(polygon_points) < 3:
                            continue
                        qpts = [img_to_screen(p) for p in polygon_points]
                        painter.setBrush(QBrush(weak_color))
                        painter.setPen(QPen(QColor(100, 100, 100), 1))
                        painter.drawPolygon(*qpts)

                if len(self.current_plant_polygons) > 0:
                    temp_color = QColor(100, 200, 100, 120)
                    painter.setBrush(QBrush(temp_color))
                    painter.setPen(QPen(QColor(0, 150, 0), 2))
                    for polygon_points in self.current_plant_polygons:
                        if len(polygon_points) >= 3:
                            qpts = [img_to_screen(p) for p in polygon_points]
                            painter.drawPolygon(*qpts)

                if len(self.current_points) > 0:
                    qpts = [img_to_screen(p) for p in self.current_points]
                    painter.setBrush(QBrush(QColor(255, 80, 80)))
                    painter.setPen(Qt.NoPen)
                    for p in qpts:
                        painter.drawEllipse(p, 5, 5)
                    if len(qpts) > 1:
                        pen = QPen(QColor(255, 80, 80), 2)
                        painter.setPen(pen)
                        for i in range(len(qpts) - 1):
                            painter.drawLine(qpts[i], qpts[i + 1])
                        painter.drawLine(qpts[-1], qpts[0])

                if self.current_snap_point is not None and self.edge_snap_enabled:
                    snap_screen = img_to_screen(self.current_snap_point)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawEllipse(snap_screen, 8, 8)
                    painter.setBrush(QBrush(QColor(0, 255, 0)))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(snap_screen, 3, 3)

            painter.end()
            self.setPixmap(final_pixmap)
        except Exception as e:
            print(f"update_display error: {e}")
            traceback.print_exc()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        if self.raw_pixmap is None:
            return
        try:
            screen_pos = event.pos()
            old_scale = self.scale_factor
            old_offset_x = self.width() / 2 - self.view_center_x * old_scale
            old_offset_y = self.height() / 2 - self.view_center_y * old_scale
            mouse_img_x = (screen_pos.x() - old_offset_x) / old_scale
            mouse_img_y = (screen_pos.y() - old_offset_y) / old_scale

            delta = event.angleDelta().y()
            if delta > 0:
                new_scale = min(old_scale * 1.1, self.max_scale)
            else:
                new_scale = max(old_scale * 0.9, self.min_scale)

            self.view_center_x = mouse_img_x + (self.width() / 2 - screen_pos.x()) / new_scale
            self.view_center_y = mouse_img_y + (self.height() / 2 - screen_pos.y()) / new_scale

            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            self.view_center_x = max(0.0, min(img_width, self.view_center_x))
            self.view_center_y = max(0.0, min(img_height, self.view_center_y))

            self.scale_factor = new_scale
            self.update_display()

            main_win = self.get_main_window()
            if main_win and not self.is_summary:
                main_win.sync_summary_view()
        except Exception as e:
            print(f"wheelEvent error: {e}")
            traceback.print_exc()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            try:
                delta = event.pos() - self.drag_last_pos
                self.view_center_x -= delta.x() / self.scale_factor
                self.view_center_y -= delta.y() / self.scale_factor

                img_width = self.raw_pixmap.width()
                img_height = self.raw_pixmap.height()
                self.view_center_x = max(0.0, min(img_width, self.view_center_x))
                self.view_center_y = max(0.0, min(img_height, self.view_center_y))

                self.drag_last_pos = event.pos()
                self.update_display()

                main_win = self.get_main_window()
                if main_win and not self.is_summary:
                    main_win.sync_summary_view()
            except Exception as e:
                print(f"mouseMoveEvent (drag) error: {e}")
        elif not self.is_summary:
            self.current_snap_point = self.calculate_snap_point(event.pos())
            self.update_display()

    def mousePressEvent(self, event):
        if self.raw_pixmap is None:
            return
        try:
            if event.button() == Qt.RightButton:
                self.is_dragging = True
                self.drag_last_pos = event.pos()
                self.setCursor(QCursor(Qt.ClosedHandCursor))
            elif event.button() == Qt.LeftButton and not self.is_summary:
                # 检查是否在SAM分割模式
                if self.sam_segmenting:
                    # SAM分割模式：添加提示点
                    screen_pos = event.pos()
                    offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
                    offset_y = self.height() / 2 - self.view_center_y * self.scale_factor
                    img_x = (screen_pos.x() - offset_x) / self.scale_factor
                    img_y = (screen_pos.y() - offset_y) / self.scale_factor

                    img_width = self.raw_pixmap.width()
                    img_height = self.raw_pixmap.height()
                    if 0 <= img_x < img_width and 0 <= img_y < img_height:
                        # 添加正例点 (1表示正例)
                        self.sam_prompt_points.append(((img_x, img_y), 1))
                        self.perform_sam_segmentation()
                    self.update_display()
                else:
                    # 检测是否按下了Shift键
                    if event.modifiers() & Qt.ShiftModifier:
                        # 切换自动吸附模式
                        self.edge_snap_enabled = not self.edge_snap_enabled
                        # 更新显示
                        self.update_display()
                        # 通知主窗口更新按钮状态
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, 'update_snap_button_state'):
                            main_win.update_snap_button_state()
                    else:
                        # 正常的左键点击，添加点
                        if self.current_snap_point is not None and self.edge_snap_enabled:
                            self.current_points.append(self.current_snap_point)
                        else:
                            screen_pos = event.pos()
                            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
                            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor
                            img_x = (screen_pos.x() - offset_x) / self.scale_factor
                            img_y = (screen_pos.y() - offset_y) / self.scale_factor

                            img_width = self.raw_pixmap.width()
                            img_height = self.raw_pixmap.height()
                            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                                self.current_points.append((img_x, img_y))

                    self.current_snap_point = None
                    self.update_display()
                    main_win = self.get_main_window()
                    if main_win:
                        main_win.update_status_bar()
                        main_win.mark_annotation_changed()  # 新增：标记标注变化
        except Exception as e:
            print(f"mousePressEvent error: {e}")
            traceback.print_exc()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def keyPressEvent(self, event):
        """处理键盘按下事件"""
        if event.key() == Qt.Key_Shift and not self.is_summary:
            # 切换自动吸附模式
            self.edge_snap_enabled = not self.edge_snap_enabled
            # 更新显示
            self.update_display()
            # 通知主窗口更新按钮状态
            main_win = self.get_main_window()
            if main_win and hasattr(main_win, 'update_snap_button_state'):
                main_win.update_snap_button_state()

    def keyReleaseEvent(self, event):
        """处理键盘释放事件"""
        # 不需要特殊处理Shift键释放

    def perform_sam_segmentation(self):
        """执行SAM分割"""
        if not self.sam_predictor or not self.sam_prompt_points:
            return

        try:
            # 准备提示点
            input_points = np.array([point[0] for point in self.sam_prompt_points])
            input_labels = np.array([point[1] for point in self.sam_prompt_points])

            # 执行分割
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # 选择得分最高的掩码
            best_mask_idx = np.argmax(scores)
            self.sam_mask = masks[best_mask_idx]

            # 转换掩码为多边形
            self.convert_mask_to_polygon()

        except Exception as e:
            print(f"SAM segmentation error: {e}")
            traceback.print_exc()

    def convert_mask_to_polygon(self):
        """将掩码转换为多边形"""
        if self.sam_mask is None:
            return

        try:
            # 查找轮廓
            mask = (self.sam_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 选择最大的轮廓
                contour = max(contours, key=cv2.contourArea)

                # 简化轮廓
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 转换为多边形点
                self.current_points = [(point[0][0], point[0][1]) for point in approx]

                # 通知主窗口保存撤销状态
                main_win = self.get_main_window()
                if main_win:
                    main_win.save_undo_state()
                    main_win.sync_summary_view()
        except Exception as e:
            print(f"Convert mask to polygon error: {e}")
            traceback.print_exc()
        pass

    def get_plant_color(self, plant_id):
        np.random.seed(plant_id)
        return (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255), 120)

    def save_current_polygon(self):
        if self.is_summary:
            return False
        if len(self.current_points) < 3:
            return False

        unique_points = list(dict.fromkeys(self.current_points))
        if len(unique_points) < 3:
            return False
        contour = np.array(unique_points, dtype=np.float32).reshape((-1, 1, 2))
        area = cv2.contourArea(contour)
        if area <= 5:
            return False

        self.current_plant_polygons.append(unique_points)
        self.current_points = []
        self.current_snap_point = None
        self.update_display()
        return True

    def confirm_preview_and_save(self):
        if self.is_summary or len(self.current_plant_polygons) == 0:
            return False

        total_area = 0
        for polygon_points in self.current_plant_polygons:
            contour = np.array(polygon_points, dtype=np.float32).reshape((-1, 1, 2))
            total_area += cv2.contourArea(contour)

        new_plant = {
            "id": self.current_plant_id,
            "polygons": copy.deepcopy(self.current_plant_polygons),
            "color": self.get_plant_color(self.current_plant_id),
            "total_area": float(total_area)
        }
        self.plants.append(new_plant)
        saved_plant_id = self.current_plant_id
        self.current_plant_id += 1

        self.current_points = []
        self.current_plant_polygons = []
        self.current_snap_point = None
        self.update_display()
        return saved_plant_id

    def undo_last_action(self):
        if self.is_summary:
            return False
        if self.current_points:
            self.current_points.pop()
            self.current_snap_point = None
            self.update_display()
            return True
        elif self.current_plant_polygons:
            self.current_plant_polygons.pop()
            self.current_snap_point = None
            self.update_display()
            return True
        return False

    def delete_plant(self, plant_id):
        if self.is_summary:
            return False
        self.plants = [p for p in self.plants if p["id"] != plant_id]
        if self.selected_plant_id == plant_id:
            self.selected_plant_id = None
        self.update_display()
        return True

    def select_plant(self, plant_id):
        self.selected_plant_id = plant_id
        self.update_display()

    def get_main_window(self):
        parent = self.parent()
        while parent and not isinstance(parent, QMainWindow):
            parent = parent.parent()
        return parent


class HelpDialog(QDialog):
    """使用说明弹窗（更新快捷键说明）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("玉米植株标注工具 使用说明")
        self.setGeometry(200, 200, 780, 680)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        help_text = QLabel()
        help_text.setWordWrap(True)
        help_text.setText("""
        <h2>玉米植株多区域标注工具 使用说明</h2>

        <h3>一、批量标注</h3>
        <ul>
            <li><b>批量加载图片</b>：点击「批量加载图片」或按 <b>Ctrl+Shift+O</b></li>
            <li><b>自动保存进度</b>：仅当标注修改时自动保存，切换图片无卡顿</li>
            <li><b>切换图片</b>：点击「上一张」「下一张」或按 <b>←</b> <b>→</b> 方向键</li>
        </ul>

        <h3>二、核心操作（更新快捷键）</h3>
        <ul>
            <li><b>边缘吸附</b>：默认开启，按 <b>Shift</b> 切换开关</li>
            <li><b>绘制顶点</b>：鼠标左键点击</li>
            <li><b>暂存当前区域</b>：按 <b>Enter</b></li>
            <li><b>保存整株</b>：按 <b>Shift+Enter</b>（已更新）</li>
            <li><b>智能撤销</b>：按 <b>Ctrl+Z</b></li>
        </ul>

        <h3>三、图像浏览操作</h3>
        <ul>
            <li><b>缩放图像</b>：鼠标滚轮滚动</li>
            <li><b>拖动图像</b>：鼠标右键按下并拖动</li>
        </ul>
        """)
        layout.addWidget(help_text)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玉米植株标注工具（性能优化+快捷键调整版）")
        self.setGeometry(100, 100, 1700, 800)

        self.image_paths = []
        self.current_image_index = -1
        self.annotation_dir = "./maize_annotations/projects"
        self.image_annotation_status = {}  # 图片标注状态: {image_path: True/False}

        # ==================== 新增：性能优化相关属性 ====================
        self.preprocess_cache = {}  # 单图预处理缓存：{image_path: (foreground_mask, edge_map)}
        self.annotation_changed = False  # 标注变化标记：仅True时自动保存
        # ================================================================

        self.current_image = None
        self.current_image_path = ""
        self.undo_stack = []
        self.redo_stack = []
        self.is_undo_redo = False

        # ==================== 新增：SAM模型相关 ====================
        self.sam_predictor = None
        self.sam_model_loaded = False
        self.sam_model_path = "sam_vit_h_4b8939.pth"  # 默认模型路径
        self.sam_model_type = "vit_h"  # 默认模型类型
        self.sam_prompt_points = []  # SAM提示点
        self.sam_segmenting = False  # 是否正在进行分割
        # ============================================================

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        main_layout.addLayout(toolbar)

        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QHBoxLayout()
        file_group.setLayout(file_layout)

        self.btn_load_batch = QPushButton("批量加载图片 (Ctrl+Shift+O)")
        self.btn_load_batch.clicked.connect(self.load_batch_images)
        file_layout.addWidget(self.btn_load_batch)

        self.btn_load_single = QPushButton("加载单张图片")
        self.btn_load_single.clicked.connect(self.load_single_image)
        file_layout.addWidget(self.btn_load_single)

        toolbar.addWidget(file_group)
        toolbar.addSpacing(15)

        # 导航组
        nav_group = QGroupBox("导航")
        nav_layout = QHBoxLayout()
        nav_group.setLayout(nav_layout)

        self.btn_prev = QPushButton("上一张 (←)")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一张 (→)")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)

        toolbar.addWidget(nav_group)
        toolbar.addSpacing(15)

        # 标注操作组
        annotate_group = QGroupBox("标注操作")
        annotate_layout = QHBoxLayout()
        annotate_group.setLayout(annotate_layout)

        self.btn_toggle_snap = QPushButton("边缘吸附: 开启 (Shift)")
        self.btn_toggle_snap.clicked.connect(self.toggle_edge_snap)
        annotate_layout.addWidget(self.btn_toggle_snap)

        self.btn_save_polygon = QPushButton("暂存当前区域 (Enter)")
        self.btn_save_polygon.clicked.connect(self.save_current_polygon)
        annotate_layout.addWidget(self.btn_save_polygon)

        # ==================== 修改：按钮文本更新为Shift+Enter ====================
        self.btn_save_plant = QPushButton("保存整株 (Shift+Enter)")
        self.btn_save_plant.clicked.connect(self.save_plant)
        annotate_layout.addWidget(self.btn_save_plant)
        # ================================================================

        # ==================== 新增：SAM相关按钮 ====================
        self.btn_load_sam = QPushButton("加载SAM模型")
        self.btn_load_sam.clicked.connect(self.load_sam_model)
        annotate_layout.addWidget(self.btn_load_sam)

        self.btn_sam_segment = QPushButton("SAM分割 (S)")
        self.btn_sam_segment.clicked.connect(self.toggle_sam_segmentation)
        self.btn_sam_segment.setEnabled(False)  # 初始禁用
        annotate_layout.addWidget(self.btn_sam_segment)
        # =========================================================

        self.btn_undo = QPushButton("撤销 (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        annotate_layout.addWidget(self.btn_undo)

        toolbar.addWidget(annotate_group)
        toolbar.addSpacing(15)

        # 植株管理组
        plant_group = QGroupBox("植株管理")
        plant_layout = QHBoxLayout()
        plant_group.setLayout(plant_layout)

        self.combo_plants = QComboBox()
        self.combo_plants.setMinimumWidth(150)
        self.combo_plants.currentTextChanged.connect(self.on_plant_selected)
        plant_layout.addWidget(self.combo_plants)

        self.btn_delete = QPushButton("删除选中植株 (Delete)")
        self.btn_delete.clicked.connect(self.delete_plant)
        plant_layout.addWidget(self.btn_delete)

        toolbar.addWidget(plant_group)
        toolbar.addSpacing(15)

        # 导出组
        export_group = QGroupBox("导出")
        export_layout = QHBoxLayout()
        export_group.setLayout(export_layout)

        self.btn_export_json = QPushButton("导出当前JSON")
        self.btn_export_json.clicked.connect(self.export_simple_json)
        export_layout.addWidget(self.btn_export_json)

        self.btn_export_coco = QPushButton("导出当前COCO")
        self.btn_export_coco.clicked.connect(self.export_coco_format)
        export_layout.addWidget(self.btn_export_coco)

        self.btn_export_annotated = QPushButton("批量导出已标注")
        self.btn_export_annotated.clicked.connect(self.export_annotated_images)
        export_layout.addWidget(self.btn_export_annotated)

        toolbar.addWidget(export_group)
        toolbar.addSpacing(15)

        # 辅助组
        aux_group = QGroupBox("辅助")
        aux_layout = QHBoxLayout()
        aux_group.setLayout(aux_layout)

        self.btn_help = QPushButton("使用说明")
        self.btn_help.clicked.connect(self.show_help)
        aux_layout.addWidget(self.btn_help)

        self.btn_toggle_annotation = QPushButton("标记为已标注")
        self.btn_toggle_annotation.clicked.connect(self.toggle_annotation_status)
        aux_layout.addWidget(self.btn_toggle_annotation)

        toolbar.addWidget(aux_group)

        # 添加图片进度标签
        self.image_progress_label = QLabel("0/0")
        self.image_progress_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        toolbar.addWidget(self.image_progress_label)

        toolbar.addStretch()

        images_layout = QHBoxLayout()
        main_layout.addLayout(images_layout)

        self.left_label = ImageLabel(is_summary=False, parent=self)
        self.left_label.setToolTip("左键添加顶点 | Enter暂存区域 | Shift+Enter保存整株 | 右键拖动 | 滚轮缩放")
        images_layout.addWidget(self.left_label, 5)

        self.right_label = ImageLabel(is_summary=True, parent=self)
        self.right_label.setToolTip("已保存植株总结预览")
        images_layout.addWidget(self.right_label, 5)

        self.init_shortcuts()

        os.makedirs(self.annotation_dir, exist_ok=True)

        self.update_status_bar()

    def init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Return), self, self.save_current_polygon)
        # ==================== 修改：保存整株快捷键改为Shift+Enter ====================
        QShortcut(QKeySequence("Shift+Return"), self, self.save_plant)
        # ================================================================
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_plant)
        QShortcut(QKeySequence(Qt.Key_Shift), self, self.toggle_edge_snap)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, self.load_batch_images)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_image)

    # ==================== 新增：性能优化方法 ====================
    def mark_annotation_changed(self):
        """标记当前图片的标注已修改，切换时自动保存"""
        self.annotation_changed = True

    def clear_annotation_changed(self):
        """清除标注变化标记"""
        self.annotation_changed = False

    # ================================================================

    def load_batch_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "批量选择图片（可多选）",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not file_paths:
            return

        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.image_paths = [
            path for path in file_paths
            if os.path.splitext(path.lower())[1] in valid_extensions
        ]

        if len(self.image_paths) == 0:
            QMessageBox.warning(self, "警告", "未选择有效的图片文件")
            return

        # 清空预处理缓存（避免旧数据干扰）
        self.preprocess_cache.clear()

        self.current_image_index = -1
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.goto_image(0)

        QMessageBox.information(
            self,
            "加载成功",
            f"成功加载 {len(self.image_paths)} 张图片\n仅当标注修改时自动保存，切换无卡顿"
        )

    def load_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单张图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = -1
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.preprocess_cache.clear()
            self.goto_image(0)

    def goto_image(self, index):
        if index < 0 or index >= len(self.image_paths):
            return

        # 1. 仅当标注真的修改时才自动保存
        if self.current_image_index >= 0 and self.current_image is not None and self.annotation_changed:
            self.save_current_annotation_auto()
            self.clear_annotation_changed()

        # 2. 加载新图片
        new_image_path = self.image_paths[index]
        try:
            self.current_image = PILImage.open(new_image_path).convert("RGBA")
            self.current_image_path = new_image_path
            self.current_image_index = index

            # 3. 优先从缓存获取预处理数据，否则重新计算并存入缓存
            preprocessed_data = self.preprocess_cache.get(new_image_path, None)
            if not preprocessed_data:
                # 临时用左侧画布计算预处理数据
                temp_label = ImageLabel()
                temp_label.preprocess_image(self.current_image)
                preprocessed_data = (temp_label.foreground_mask, temp_label.edge_map)
                self.preprocess_cache[new_image_path] = preprocessed_data

            # 4. 重置画布（传入预处理数据）
            self.left_label.set_image(self.current_image, preprocessed_data)
            self.right_label.set_image(self.current_image, preprocessed_data)

            # 5. 尝试加载已有标注
            self.load_current_annotation_auto()

            # 6. 更新UI
            self.update_plant_list()
            self.clear_undo_stack()
            self.clear_annotation_changed()  # 新图片初始状态无变化
            # 更新标注状态按钮文本
            is_annotated = self.image_annotation_status.get(self.current_image_path, False)
            if is_annotated:
                self.btn_toggle_annotation.setText("标记为未标注")
            else:
                self.btn_toggle_annotation.setText("标记为已标注")
            self.update_status_bar()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
            traceback.print_exc()

    def prev_image(self):
        if self.current_image_index > 0:
            self.goto_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.goto_image(self.current_image_index + 1)

    def get_auto_save_path(self, image_path):
        if not image_path:
            return None
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        path_hash = abs(hash(image_path)) % 10000
        return os.path.join(self.annotation_dir, f"{image_name}_{path_hash}.maize")

    def save_current_annotation_auto(self):
        save_path = self.get_auto_save_path(self.current_image_path)
        if not save_path:
            return

        try:
            project_data = {
                "image_path": self.current_image_path,
                "image_size": {"width": self.current_image.width, "height": self.current_image.height},
                "plants": self.left_label.plants,
                "current_plant_id": self.left_label.current_plant_id,
                "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "batch_optimized_1.0",
                "is_auto_save": True
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"自动保存失败: {e}")

    def load_current_annotation_auto(self):
        load_path = self.get_auto_save_path(self.current_image_path)
        if not load_path or not os.path.exists(load_path):
            return

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                project_data = json.load(f)

            for plant in project_data.get("plants", []):
                if "points" in plant and "polygons" not in plant:
                    plant["polygons"] = [plant["points"]]
                    plant["total_area"] = plant.get("area", 0)

            self.left_label.plants = project_data.get("plants", [])
            self.left_label.current_plant_id = project_data.get("current_plant_id", 1)
            # ==================== 优化：右侧画布直接同步左侧初始状态 ====================
            self.right_label.plants = self.left_label.plants
            self.right_label.current_plant_id = self.left_label.current_plant_id
            # ================================================================

        except Exception as e:
            print(f"自动加载失败: {e}")

    def toggle_edge_snap(self):
        self.left_label.edge_snap_enabled = not self.left_label.edge_snap_enabled
        self.update_snap_button_state()
        if not self.left_label.edge_snap_enabled:
            self.left_label.current_snap_point = None
            self.left_label.update_display()
        self.update_status_bar()

    def update_snap_button_state(self):
        """更新边缘吸附按钮的状态"""
        if self.left_label.edge_snap_enabled:
            self.btn_toggle_snap.setText("边缘吸附: 开启 (Shift)")
        else:
            self.btn_toggle_snap.setText("边缘吸附: 关闭 (Shift)")

    # ==================== 新增：SAM模型相关方法 ====================
    def load_sam_model(self):
        """加载SAM模型"""
        if SamPredictor is None:
            QMessageBox.warning(self, "警告", "SAM库未安装，请安装segment-anything包")
            return

        # 选择模型文件
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM模型文件", ".", "PTH文件 (*.pth)"
        )

        if not model_path:
            return

        # 显示加载进度
        progress = QProgressDialog("正在加载SAM模型...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)

        try:
            # 加载模型
            progress.setValue(20)
            sam = sam_model_registry[self.sam_model_type](checkpoint=model_path)

            progress.setValue(60)
            # 移动到GPU（如果可用）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sam.to(device=device)

            progress.setValue(80)
            # 创建预测器
            self.sam_predictor = SamPredictor(sam)
            self.sam_model_loaded = True

            progress.setValue(100)
            QMessageBox.information(self, "成功", f"SAM模型加载成功！使用设备: {device}")

            # 启用分割按钮
            self.btn_sam_segment.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载SAM模型失败: {str(e)}")
        finally:
            progress.close()

    def toggle_sam_segmentation(self):
        """切换SAM分割模式"""
        if not self.sam_model_loaded:
            QMessageBox.warning(self, "警告", "请先加载SAM模型")
            return

        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        # 切换分割模式
        self.sam_segmenting = not self.sam_segmenting

        if self.sam_segmenting:
            self.btn_sam_segment.setText("退出分割 (S)")
            self.left_label.sam_segmenting = True
            self.left_label.sam_predictor = self.sam_predictor
            self.left_label.sam_prompt_points = []
            # 设置当前图像到SAM预测器
            img_np = np.array(self.current_image.convert("RGB"))
            self.sam_predictor.set_image(img_np)
        else:
            self.btn_sam_segment.setText("SAM分割 (S)")
            self.left_label.sam_segmenting = False
            self.left_label.sam_prompt_points = []

        self.left_label.update_display()
        self.update_status_bar()

    # =============================================================

    def show_help(self):
        dialog = HelpDialog(self)
        dialog.exec_()

    def save_current_polygon(self):
        if self.left_label.save_current_polygon():
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()

    def save_plant(self):
        saved_id = self.left_label.confirm_preview_and_save()
        if saved_id:
            saved_plant = next((p for p in self.left_label.plants if p["id"] == saved_id), None)
            if saved_plant:
                self.push_undo_action("add_plant", saved_plant)
            # ==================== 优化：右侧画布按需同步 ====================
            self.right_label.plants = copy.deepcopy(self.left_label.plants)
            self.right_label.current_plant_id = self.left_label.current_plant_id
            # ================================================================
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
        else:
            QMessageBox.warning(self, "警告", "请先暂存至少一个区域")

    def undo(self):
        if self.left_label.undo_last_action():
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
            return
        if not self.undo_stack:
            return
        self.is_undo_redo = True
        try:
            action = self.undo_stack.pop()
            if action["type"] == "add_plant":
                plant = action["data"]
                self.left_label.delete_plant(plant["id"])
                self.right_label.delete_plant(plant["id"])
                self.redo_stack.append(action)
            elif action["type"] == "delete_plant":
                plant = action["data"]
                self.left_label.plants.append(plant)
                self.left_label.plants.sort(key=lambda x: x["id"])
                self.right_label.plants.append(copy.deepcopy(plant))
                self.right_label.plants.sort(key=lambda x: x["id"])
                self.redo_stack.append(action)
            self.update_plant_list()
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
        except Exception as e:
            print(f"undo error: {e}")
        finally:
            self.is_undo_redo = False
            self.update_undo_redo_state()

    def redo(self):
        if not self.redo_stack:
            return
        self.is_undo_redo = True
        try:
            action = self.redo_stack.pop()
            if action["type"] == "add_plant":
                plant = action["data"]
                self.left_label.plants.append(plant)
                self.left_label.plants.sort(key=lambda x: x["id"])
                self.right_label.plants.append(copy.deepcopy(plant))
                self.right_label.plants.sort(key=lambda x: x["id"])
                self.undo_stack.append(action)
            elif action["type"] == "delete_plant":
                plant = action["data"]
                self.left_label.delete_plant(plant["id"])
                self.right_label.delete_plant(plant["id"])
                self.undo_stack.append(action)
            self.update_plant_list()
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
        except Exception as e:
            print(f"redo error: {e}")
        finally:
            self.is_undo_redo = False
            self.update_undo_redo_state()

    def push_undo_action(self, action_type, data):
        if self.is_undo_redo:
            return
        self.undo_stack.append({"type": action_type, "data": copy.deepcopy(data)})
        self.redo_stack.clear()
        self.update_undo_redo_state()

    def clear_undo_stack(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_state()

    def update_undo_redo_state(self):
        self.btn_undo.setEnabled(len(self.undo_stack) > 0 or len(self.left_label.current_points) > 0 or len(
            self.left_label.current_plant_polygons) > 0)

    def delete_plant(self):
        selected_text = self.combo_plants.currentText()
        if not selected_text:
            QMessageBox.warning(self, "警告", "请先选择要删除的植株")
            return
        try:
            plant_id = int(selected_text.split("_")[1])
            target_plant = next((p for p in self.left_label.plants if p["id"] == plant_id), None)
            if target_plant:
                self.push_undo_action("delete_plant", target_plant)
            self.left_label.delete_plant(plant_id)
            self.right_label.delete_plant(plant_id)
            self.update_plant_list()
            self.mark_annotation_changed()  # 标记变化
            self.update_status_bar()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"删除失败: {str(e)}")

    def on_plant_selected(self, selected_text):
        if not selected_text:
            return
        try:
            plant_id = int(selected_text.split("_")[1])
            self.left_label.select_plant(plant_id)
            self.right_label.select_plant(plant_id)
        except Exception as e:
            pass

    def update_plant_list(self):
        self.combo_plants.blockSignals(True)
        self.combo_plants.clear()
        for plant in self.left_label.plants:
            self.combo_plants.addItem(f"plant_{plant['id']}")
        self.combo_plants.blockSignals(False)
        self.right_label.update_display()

    def update_status_bar(self):
        base_msg = "就绪"

        if len(self.image_paths) > 0 and self.current_image_index >= 0:
            progress_msg = f" | 图片: {self.current_image_index + 1}/{len(self.image_paths)}"
            image_name = os.path.basename(self.current_image_path)
            # 检查当前图片的标注状态
            is_annotated = self.image_annotation_status.get(self.current_image_path, False)
            status_str = "[已标注]" if is_annotated else "[未标注]"
            base_msg += f" | 当前: {image_name} {status_str}"
            base_msg += progress_msg
            # 更新图片进度标签
            self.image_progress_label.setText(f"{self.current_image_index + 1}/{len(self.image_paths)}")
        else:
            # 没有图片时重置标签
            self.image_progress_label.setText("0/0")

        if hasattr(self.left_label, 'edge_snap_enabled'):
            snap_msg = " | 边缘吸附: 开启" if self.left_label.edge_snap_enabled else " | 边缘吸附: 关闭"
            base_msg += snap_msg

        if hasattr(self.left_label, 'current_points') and hasattr(self.left_label, 'current_plant_polygons'):
            if len(self.left_label.current_points) > 0 or len(self.left_label.current_plant_polygons) > 0:
                state_msg = f" | 当前多边形顶点: {len(self.left_label.current_points)} | 已暂存区域: {len(self.left_label.current_plant_polygons)}"
                base_msg += state_msg

        if hasattr(self.left_label, 'plants'):
            base_msg += f" | 已标注植株: {len(self.left_label.plants)}"

        # ==================== 新增：显示标注变化状态 ====================
        if self.annotation_changed:
            base_msg += " | 【有未保存的修改】"
        # ================================================================

        self.statusBar().showMessage(base_msg)

    def sync_summary_view(self):
        self.right_label.scale_factor = self.left_label.scale_factor
        self.right_label.view_center_x = self.left_label.view_center_x
        self.right_label.view_center_y = self.left_label.view_center_y
        self.right_label.update_display()

    def toggle_annotation_status(self):
        """切换当前图片的标注状态"""
        if self.current_image_path:
            current_status = self.image_annotation_status.get(self.current_image_path, False)
            new_status = not current_status
            self.image_annotation_status[self.current_image_path] = new_status
            self.update_status_bar()
            # 更新按钮文本
            if new_status:
                self.btn_toggle_annotation.setText("标记为未标注")
            else:
                self.btn_toggle_annotation.setText("标记为已标注")
            QMessageBox.information(self, "状态更新", f"图片已{'标记为已标注' if new_status else '标记为未标注'}")

    def export_simple_json(self):
        if not self.left_label.plants:
            QMessageBox.warning(self, "警告", "当前图片没有标注数据可导出")
            return
        if self.current_image is None:
            return
        try:
            out_dir = "./maize_annotations"
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            json_path = os.path.join(out_dir, f"{image_name}_simple_{timestamp}.json")
            export_data = {
                "image_info": {
                    "image_path": self.current_image_path,
                    "image_name": os.path.basename(self.current_image_path),
                    "width": self.current_image.width,
                    "height": self.current_image.height
                },
                "annotation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_plants": len(self.left_label.plants),
                "plants": [
                    {
                        "id": p["id"],
                        "polygons": p["polygons"],
                        "total_area": p["total_area"],
                        "bbox": self.get_bbox_from_polygons(p["polygons"])
                    } for p in self.left_label.plants
                ]
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "导出成功", f"当前图片的简易JSON已保存至: {json_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def export_coco_format(self):
        if not self.left_label.plants:
            QMessageBox.warning(self, "警告", "当前图片没有标注数据可导出")
            return
        if self.current_image is None:
            return
        try:
            out_dir = "./maize_annotations"
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            coco_path = os.path.join(out_dir, f"{image_name}_coco_{timestamp}.json")
            img_name = os.path.basename(self.current_image_path)
            img_width = self.current_image.width
            img_height = self.current_image.height

            coco_data = {
                "info": {
                    "description": "Maize Plant Multi-Region Instance Segmentation Dataset (Optimized)",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [],
                "categories": [
                    {"id": 1, "name": "maize_plant", "supercategory": "plant"}
                ],
                "images": [
                    {
                        "id": 1,
                        "file_name": img_name,
                        "width": img_width,
                        "height": img_height,
                        "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ],
                "annotations": []
            }

            annotation_id = 1
            for plant in self.left_label.plants:
                polygons = plant["polygons"]
                bbox = self.get_bbox_from_polygons(polygons)
                segmentation = []
                for polygon_points in polygons:
                    seg = [coord for point in polygon_points for coord in point]
                    segmentation.append(seg)
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": plant["total_area"],
                    "bbox": bbox,
                    "iscrowd": 0
                })
                annotation_id += 1

            with open(coco_path, "w", encoding="utf-8") as f:
                json.dump(coco_data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "导出成功", f"当前图片的COCO格式已保存至: {coco_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def get_bbox_from_polygons(self, polygons):
        all_x = []
        all_y = []
        for polygon_points in polygons:
            all_x.extend([p[0] for p in polygon_points])
            all_y.extend([p[1] for p in polygon_points])
        xmin = min(all_x)
        ymin = min(all_y)
        xmax = max(all_x)
        ymax = max(all_y)
        width = xmax - xmin
        height = ymax - ymin
        return [round(xmin, 2), round(ymin, 2), round(width, 2), round(height, 2)]

    def export_annotated_images(self):
        """批量导出所有已标注的图片"""
        annotated_images = [path for path in self.image_paths if self.image_annotation_status.get(path, False)]

        if not annotated_images:
            QMessageBox.warning(self, "警告", "没有已标注的图片可导出")
            return

        try:
            out_dir = "./maize_annotations/annotated_batch"
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 准备COCO格式的批量导出数据
            coco_data = {
                "info": {
                    "description": "Maize Plant Multi-Region Instance Segmentation Dataset (Batch Export)",
                    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "licenses": [],
                "categories": [
                    {"id": 1, "name": "maize_plant", "supercategory": "plant"}
                ],
                "images": [],
                "annotations": []
            }

            image_id = 1
            annotation_id = 1

            for image_path in annotated_images:
                # 加载图片和标注
                try:
                    image = PILImage.open(image_path).convert("RGBA")
                    load_path = self.get_auto_save_path(image_path)

                    if os.path.exists(load_path):
                        with open(load_path, "r", encoding="utf-8") as f:
                            project_data = json.load(f)

                        plants = project_data.get("plants", [])
                        if plants:
                            # 添加到COCO数据
                            img_name = os.path.basename(image_path)
                            coco_data["images"].append({
                                "id": image_id,
                                "file_name": img_name,
                                "width": image.width,
                                "height": image.height,
                                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

                            for plant in plants:
                                polygons = plant["polygons"]
                                bbox = self.get_bbox_from_polygons(polygons)
                                segmentation = []
                                for polygon_points in polygons:
                                    seg = [coord for point in polygon_points for coord in point]
                                    segmentation.append(seg)
                                coco_data["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": 1,
                                    "segmentation": segmentation,
                                    "area": plant["total_area"],
                                    "bbox": bbox,
                                    "iscrowd": 0
                                })
                                annotation_id += 1

                            image_id += 1
                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {e}")
                    continue

            # 导出COCO格式
            if coco_data["images"]:
                coco_path = os.path.join(out_dir, f"annotated_batch_coco_{timestamp}.json")
                with open(coco_path, "w", encoding="utf-8") as f:
                    json.dump(coco_data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "导出成功",
                                        f"已成功导出 {len(coco_data['images'])} 张已标注图片到: {coco_path}")
            else:
                QMessageBox.warning(self, "警告", "没有找到已标注的图片数据")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量导出失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())