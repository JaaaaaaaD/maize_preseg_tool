# 图像标注控件
import cv2
import numpy as np
import traceback
from PyQt5.QtWidgets import QLabel, QSizePolicy, QProgressDialog
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QCursor
from PIL import Image as PILImage
from utils.image_processor import preprocess_image, calculate_snap_point
from utils.auxiliary_algorithms import perform_region_growing, convert_mask_to_polygon
from utils.helpers import get_plant_color, calculate_polygon_area
from config import SNAP_RADIUS, ROI_SIZE, COLOR_CHANGE_THRESHOLD


class ImageLabel(QLabel):
    """图像显示与标注控件"""

    def __init__(self, is_summary=False, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")

        # 图像基础属性
        self.raw_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.view_center_x = 0.0
        self.view_center_y = 0.0

        # 拖动相关属性（修复：初始化状态）
        self.is_dragging = False
        self.drag_last_pos = QPoint()
        self.last_mouse_pos = QPoint()

        # 标注核心数据
        self.plants = []
        self.current_points = []
        self.current_plant_polygons = []
        self.current_plant_id = 1
        self.selected_plant_id = None
        self.is_summary = is_summary

        # 控件基础设置
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # 边缘吸附相关
        self.edge_snap_enabled = True
        self.snap_radius = SNAP_RADIUS
        self.foreground_mask = None
        self.edge_map = None
        self.current_snap_point = None
        self.color_image = None  # 存储原始颜色图像数据

        # SAM分割相关属性
        self.sam_segmenting = False  # 是否处于SAM分割模式
        self.sam_predictor = None  # SAM预测器
        self.sam_prompt_points = []  # SAM提示点
        self.sam_mask = None  # SAM分割结果

        # 区域生长相关属性
        self.region_growing_enabled = False  # 是否启用区域生长
        self.region_growing_threshold = 30  # 颜色差异阈值
        self.region_growing_mask = None  # 区域生长结果掩码

    def set_image(self, pil_image, preprocessed_data=None):
        """
        设置图像，支持传入预处理数据
        :param pil_image: PIL图像对象
        :param preprocessed_data: (foreground_mask, edge_map) 元组
        """
        if pil_image is None:
            return
        try:
            # 重置所有标注状态
            self.plants = []
            self.current_points = []
            self.current_plant_polygons = []
            self.current_plant_id = 1
            self.selected_plant_id = None
            self.scale_factor = 1.0
            self.current_snap_point = None

            # 修复：重置拖动状态，避免切换图片后状态残留
            self.is_dragging = False
            self.drag_last_pos = QPoint()

            # 初始化视图中心为图片中心
            self.view_center_x = pil_image.width / 2.0
            self.view_center_y = pil_image.height / 2.0

            # 转换图像格式
            data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
            self.raw_pixmap = QPixmap.fromImage(qimage)

            # 保存原始颜色图像数据
            self.color_image = np.array(pil_image.convert("RGB"))

            # 优先使用传入的预处理数据，否则重新计算
            if preprocessed_data:
                self.foreground_mask, self.edge_map = preprocessed_data
            else:
                self.foreground_mask, self.edge_map = preprocess_image(pil_image)

            self.update_display()
        except Exception as e:
            print(f"set_image error: {e}")

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
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
                        main_win.mark_annotation_changed()  # 标记标注变化
        except Exception as e:
            print(f"mousePressEvent error: {e}")
            traceback.print_exc()

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.RightButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        # 更新鼠标位置
        self.last_mouse_pos = event.pos()
        
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

    def wheelEvent(self, event):
        """处理鼠标滚轮事件"""
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

    def calculate_snap_point(self, screen_pos):
        """计算边缘吸附点"""
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

    def screen_to_image(self, screen_pos):
        """将屏幕坐标转换为图像坐标"""
        if not self.raw_pixmap:
            return None
        try:
            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor

            img_x = (screen_pos.x() - offset_x) / self.scale_factor
            img_y = (screen_pos.y() - offset_y) / self.scale_factor

            # 严格校验坐标是否在图像范围内，避免越界
            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                return img_x, img_y

            return None
        except Exception as e:
            print(f"screen_to_image error: {e}")
            return None

    def image_to_screen(self, image_pos):
        """将图像坐标转换为屏幕坐标"""
        if not self.raw_pixmap:
            return None
        try:
            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor

            screen_x = image_pos[0] * self.scale_factor + offset_x
            screen_y = image_pos[1] * self.scale_factor + offset_y
            return screen_x, screen_y
        except Exception as e:
            print(f"image_to_screen error: {e}")
            return None

    def add_vertex(self, image_pos):
        """添加顶点"""
        # 如果启用了边缘吸附，使用吸附点
        if self.edge_snap_enabled and self.current_snap_point:
            self.current_points.append(self.current_snap_point)
        else:
            self.current_points.append(image_pos)
        self.update_display()

    def save_current_polygon(self):
        """保存当前多边形"""
        if self.is_summary:
            return False
        if len(self.current_points) < 3:
            return False

        unique_points = list(dict.fromkeys(self.current_points))
        if len(unique_points) < 3:
            return False

        # 确保多边形闭合，连接最后一个点和第一个点
        if unique_points[0] != unique_points[-1]:
            unique_points.append(unique_points[0])

        area = calculate_polygon_area(unique_points)
        if area <= 5:
            return False

        self.current_plant_polygons.append(unique_points)
        self.current_points = []
        self.current_snap_point = None
        self.update_display()
        return True

    def confirm_preview_and_save(self):
        """确认预览并保存整株"""
        if self.is_summary or len(self.current_plant_polygons) == 0:
            return False

        total_area = 0
        for polygon_points in self.current_plant_polygons:
            total_area += calculate_polygon_area(polygon_points)

        new_plant = {
            "id": self.current_plant_id,
            "polygons": self.current_plant_polygons.copy(),
            "color": get_plant_color(self.current_plant_id),
            "total_area": float(total_area)
        }
        self.plants.append(new_plant)
        saved_plant_id = self.current_plant_id
        self.current_plant_id += 1

        # 重置当前绘制状态
        self.current_points = []
        self.current_plant_polygons = []
        self.current_snap_point = None
        self.update_display()

        return saved_plant_id

    def undo_last_action(self):
        """撤销上一个操作"""
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
        """删除植株"""
        self.plants = [p for p in self.plants if p["id"] != plant_id]
        if self.selected_plant_id == plant_id:
            self.selected_plant_id = None

        self.update_display()
        return True

    def select_plant(self, plant_id):
        """选择植株"""
        self.selected_plant_id = plant_id
        self.update_display()

    def perform_region_growing(self, seed_point):
        """执行区域生长算法"""
        if self.color_image is None:
            return

        # 显示处理进度
        progress = QProgressDialog("正在执行膨胀点选...", "取消", 0, 100, self.get_main_window())
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.show()

        def progress_callback(value):
            progress.setValue(value)

        try:
            # 执行区域生长
            mask = perform_region_growing(self.color_image, seed_point,
                                          self.region_growing_threshold, progress_callback)

            if mask is not None:
                self.region_growing_mask = mask
                # 转换掩码为多边形
                self.current_points = convert_mask_to_polygon(mask)
                # 通知主窗口保存撤销状态
                main_win = self.get_main_window()
                if main_win:
                    main_win.save_undo_state()
                    main_win.sync_summary_view()
        except Exception as e:
            print(f"Region growing error: {e}")
        finally:
            progress.close()

        self.update_display()

    def perform_sam_segmentation(self, point):
        """执行SAM分割"""
        if not self.sam_predictor or self.color_image is None:
            return

        try:
            # 添加提示点
            self.sam_prompt_points.append(point)

            # 准备提示点数据
            point_coords = np.array(self.sam_prompt_points)
            point_labels = np.ones(len(point_coords), dtype=np.int32)

            # 执行分割
            masks, _, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # 选择最佳掩码
            if masks is not None and len(masks) > 0:
                # 选择面积最大的掩码
                best_mask_idx = np.argmax([np.sum(mask) for mask in masks])
                self.sam_mask = masks[best_mask_idx]

                # 转换掩码为多边形
                mask = (self.sam_mask * 255).astype(np.uint8)
                self.current_points = convert_mask_to_polygon(mask)

                # 通知主窗口保存撤销状态
                main_win = self.get_main_window()
                if main_win:
                    main_win.save_undo_state()
                    main_win.sync_summary_view()
        except Exception as e:
            print(f"SAM segmentation error: {e}")

        self.update_display()

    def update_display(self):
        """更新显示"""
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
                    # 绘制空心红色小圆
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    for p in qpts:
                        painter.drawEllipse(p, 4, 4)
                    # 绘制已确定的线段
                    if len(qpts) > 1:
                        pen = QPen(QColor(255, 0, 0), 2)
                        painter.setPen(pen)
                        for i in range(len(qpts) - 1):
                            painter.drawLine(qpts[i], qpts[i + 1])
                    # 绘制鼠标到最后一个点的虚线
                    if self.underMouse() and hasattr(self, 'last_mouse_pos'):
                        mouse_pos = self.last_mouse_pos
                        if mouse_pos:
                            last_point = qpts[-1]
                            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
                            painter.setPen(pen)
                            painter.drawLine(last_point, mouse_pos)

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

    def get_main_window(self):
        """获取主窗口"""
        parent = self.parent()
        while parent and not hasattr(parent, "toggle_edge_snap"):
            parent = parent.parent()
        return parent