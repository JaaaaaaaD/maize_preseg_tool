# 图像标注控件
import copy
import math
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QProgressDialog, QSizePolicy

from config import SNAP_RADIUS
from utils.annotation_schema import (
    ensure_plant_groups,
    make_formal_instance,
    make_plant_group,
    next_instance_id,
    next_plant_group_id,
    normalize_candidate_instance,
    normalize_formal_instance,
    normalize_polygons,
    set_instance_owner,
    touch_instance,
)
from utils.auxiliary_algorithms import convert_mask_to_polygon, perform_region_growing
from utils.helpers import calculate_polygon_area, get_plant_color
from utils.image_processor import preprocess_image


class ImageLabel(QLabel):
    """图像显示与标注控件。

    说明：
    - `plants` 仍沿用旧字段名，但现在语义是“正式实例列表”；
    - `candidate_instances` 是独立候选层，不会被自动保存到 .maize；
    - `plant_groups` 只服务于交互归属，不进入 YOLO 训练标签。
    """

    def __init__(self, is_summary=False, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(180, 180)
        self.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")

        self.raw_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.view_center_x = 0.0
        self.view_center_y = 0.0
        self.color_image = None

        self.is_dragging = False
        self.drag_last_pos = QPoint()
        self.last_mouse_pos = QPoint()
        self.vertex_drag_info = None
        self.vertex_hit_radius = 8

        self.plants = []
        self.plant_groups = []
        self.current_points = []
        self.current_plant_polygons = []
        self.current_plant_id = 1
        self.next_owner_plant_id = 1
        self.selected_plant_id = None

        self.candidate_instances = []
        self.selected_entity_kind = None
        self.selected_entity_id = None
        self.class_names = ["plant"]

        self.is_summary = is_summary
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.edge_snap_enabled = True
        self.snap_radius = SNAP_RADIUS
        self.foreground_mask = None
        self.edge_map = None
        self.current_snap_point = None

        # self.sam_segmenting = False
        # self.sam_predictor = None
        # self.sam_prompt_points = []
        # self.sam_mask = None

        self.region_growing_enabled = False
        self.region_growing_threshold = 30
        self.region_growing_mask = None
        
        # 忽略区域相关属性
        self.ignored_regions = []  # 存储所有忽略区域的多边形
        self.current_ignored_points = []  # 当前正在绘制的忽略区域顶点
        self.ignoring_region = False  # 是否处于忽略区域绘制模式
        
        # 去除区域相关属性
        self.current_removal_points = []  # 当前正在绘制的去除区域顶点
        self.removal_regions = []  # 存储当前植株的去除区域
        self.removing_region = False  # 是否处于去除区域绘制模式

    def set_class_names(self, class_names):
        """更新当前项目类别配置。"""
        if class_names:
            self.class_names = list(class_names)

    def set_image(self, pil_image, preprocessed_data=None):
        """设置图像，支持传入预处理数据。"""
        if pil_image is None:
            return
        try:
            self.current_points = []
            self.current_plant_polygons = []
            self.selected_entity_kind = None
            self.selected_entity_id = None
            self.selected_plant_id = None
            self.candidate_instances = []
            self.vertex_drag_info = None
            self.scale_factor = 1.0
            self.current_snap_point = None
            self.is_dragging = False
            self.drag_last_pos = QPoint()

            self.view_center_x = pil_image.width / 2.0
            self.view_center_y = pil_image.height / 2.0

            data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
            self.raw_pixmap = QPixmap.fromImage(qimage)
            self.color_image = np.array(pil_image.convert("RGB"))

            if preprocessed_data:
                self.foreground_mask, self.edge_map = preprocessed_data
            else:
                self.foreground_mask, self.edge_map = preprocess_image(pil_image)

            self.update_display()
        except Exception as error:
            print(f"set_image error: {error}")
            traceback.print_exc()

    def set_annotation_state(self, plants, plant_groups=None, current_plant_id=None, next_owner_plant_id=None):
        """加载正式层状态。"""
        normalized_plants = []
        for index, plant in enumerate(plants or [], start=1):
            normalized_plants.append(normalize_formal_instance(plant, self.class_names, index))
        self.plants = normalized_plants
        self.plant_groups = ensure_plant_groups(self.plants, plant_groups or [])
        self.current_plant_id = next_instance_id(self.plants, current_plant_id or 1)
        self.next_owner_plant_id = next_owner_plant_id or next_plant_group_id(self.plant_groups)
        self.selected_entity_kind = None
        self.selected_entity_id = None
        self.selected_plant_id = None
        self.current_points = []
        self.current_plant_polygons = []
        self.candidate_instances = []
        self.update_display()

    def set_candidates(self, candidates):
        """替换候选层。"""
        normalized_candidates = []
        for index, candidate in enumerate(candidates or [], start=1):
            normalized_candidates.append(normalize_candidate_instance(candidate, self.class_names, index))
        self.candidate_instances = normalized_candidates
        if self.selected_entity_kind == "candidate":
            if not any(candidate["candidate_id"] == self.selected_entity_id for candidate in self.candidate_instances):
                self.selected_entity_kind = None
                self.selected_entity_id = None
        self.update_display()

    def clear_candidates(self):
        """清空候选层。"""
        self.candidate_instances = []
        if self.selected_entity_kind == "candidate":
            self.selected_entity_kind = None
            self.selected_entity_id = None
        self.update_display()

    def get_annotation_state(self):
        """导出正式层状态，供主窗口保存。"""
        return {
            "plants": copy.deepcopy(self.plants),
            "plant_groups": copy.deepcopy(self.plant_groups),
            "current_plant_id": self.current_plant_id,
            "next_owner_plant_id": self.next_owner_plant_id,
        }

    def create_plant_group(self, plant_name=None):
        """新建植株归属组。"""
        plant_group = make_plant_group(self.next_owner_plant_id, plant_name)
        self.plant_groups.append(plant_group)
        self.next_owner_plant_id += 1
        self.update_display()
        return plant_group

    def get_selected_entity(self):
        """获取当前选中的正式实例或候选实例。"""
        if self.selected_entity_kind == "formal":
            return self.selected_entity_kind, next(
                (plant for plant in self.plants if int(plant.get("id", 0)) == int(self.selected_entity_id)),
                None,
            )
        if self.selected_entity_kind == "candidate":
            return self.selected_entity_kind, next(
                (
                    candidate
                    for candidate in self.candidate_instances
                    if candidate.get("candidate_id") == self.selected_entity_id
                ),
                None,
            )
        return None, None

    def select_entity(self, entity_kind, entity_id):
        """选中正式实例或候选实例。"""
        if self.selected_entity_kind == entity_kind and self.selected_entity_id == entity_id:
            return
        self.selected_entity_kind = entity_kind
        self.selected_entity_id = entity_id
        self.selected_plant_id = entity_id if entity_kind == "formal" else None
        self.update_display()
        self._notify_selection_changed()

    def select_plant(self, plant_id):
        """兼容旧接口：按正式实例 id 选中。"""
        self.select_entity("formal", plant_id)

    def set_selected_entity_class(self, class_id):
        """修改选中对象的类别。"""
        entity_kind, entity = self.get_selected_entity()
        if not entity:
            return False
        class_id = max(0, min(len(self.class_names) - 1, int(class_id)))
        entity["class_id"] = class_id
        entity["class_name"] = self.class_names[class_id]
        if entity_kind == "formal" and entity.get("source") in ("ai_accepted", "ai_assisted"):
            entity["source"] = "ai_modified"
            touch_instance(entity)
        self.update_display()
        return True

    def set_selected_entity_owner(self, owner_plant_id):
        """修改选中对象归属。"""
        entity_kind, entity = self.get_selected_entity()
        if not entity:
            return False
        set_instance_owner(entity, self.plant_groups, owner_plant_id)
        if entity_kind == "formal" and entity.get("source") in ("ai_accepted", "ai_assisted"):
            entity["source"] = "ai_modified"
            touch_instance(entity)
        self.update_display()
        return True

    def accept_selected_candidate(self):
        """接受当前候选并转为正式实例。"""
        if self.selected_entity_kind != "candidate":
            return None
        accepted_id = self._accept_candidate(self.selected_entity_id)
        if accepted_id:
            self.select_entity("formal", accepted_id)
        return accepted_id

    def accept_all_candidates(self):
        """接受全部候选。"""
        accepted_ids = []
        for candidate in list(self.candidate_instances):
            accepted_id = self._accept_candidate(candidate.get("candidate_id"))
            if accepted_id:
                accepted_ids.append(accepted_id)
        if accepted_ids:
            self.select_entity("formal", accepted_ids[-1])
        return accepted_ids

    def _accept_candidate(self, candidate_id):
        """候选转正式。"""
        candidate = next(
            (item for item in self.candidate_instances if item.get("candidate_id") == candidate_id),
            None,
        )
        if not candidate:
            return None

        formal_instance = make_formal_instance(
            instance_id=self.current_plant_id,
            polygons=candidate.get("polygons", []),
            class_names=self.class_names,
            class_id=candidate.get("class_id", 0),
            source="ai_accepted",
            origin_model_version=candidate.get("model_version"),
            origin_confidence=candidate.get("confidence"),
            owner_plant_id=candidate.get("owner_plant_id"),
        )
        if formal_instance.get("owner_plant_id") is not None:
            set_instance_owner(formal_instance, self.plant_groups, formal_instance.get("owner_plant_id"))

        self.plants.append(formal_instance)
        self.current_plant_id += 1
        self.candidate_instances = [
            item for item in self.candidate_instances if item.get("candidate_id") != candidate_id
        ]
        if self.selected_entity_kind == "candidate" and self.selected_entity_id == candidate_id:
            self.selected_entity_kind = None
            self.selected_entity_id = None
        self.update_display()
        return formal_instance["id"]

    def delete_selected_entity(self):
        """删除当前选中候选或正式实例。"""
        if self.selected_entity_kind == "candidate":
            before = len(self.candidate_instances)
            self.candidate_instances = [
                item for item in self.candidate_instances if item.get("candidate_id") != self.selected_entity_id
            ]
            deleted = len(self.candidate_instances) != before
        elif self.selected_entity_kind == "formal":
            before = len(self.plants)
            self.plants = [item for item in self.plants if int(item.get("id", 0)) != int(self.selected_entity_id)]
            deleted = len(self.plants) != before
        else:
            deleted = False

        if deleted:
            self.selected_entity_kind = None
            self.selected_entity_id = None
            self.selected_plant_id = None
            self.update_display()
        return deleted

    def delete_plant(self, plant_id):
        """兼容旧接口：删除正式实例。"""
        plant_id = int(plant_id)
        self.plants = [plant for plant in self.plants if int(plant.get("id", 0)) != plant_id]
        if self.selected_plant_id == plant_id:
            self.selected_plant_id = None
        if self.selected_entity_kind == "formal" and int(self.selected_entity_id or 0) == plant_id:
            self.selected_entity_kind = None
            self.selected_entity_id = None
        self.update_display()
        return True

    def save_current_polygon(self):
        """保存当前多边形到临时实例预览层。"""
        if self.is_summary:
            return False
        if len(self.current_points) < 3:
            return False

        unique_points = []
        for point in self.current_points:
            if not unique_points or unique_points[-1] != point:
                unique_points.append(point)
        if len(unique_points) < 3:
            return False

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
    
    def save_current_ignored_region(self):
        """保存当前忽略区域。"""
        if self.is_summary:
            return False
        if len(self.current_ignored_points) < 3:
            return False

        unique_points = []
        for point in self.current_ignored_points:
            if not unique_points or unique_points[-1] != point:
                unique_points.append(point)
        if len(unique_points) < 3:
            return False

        if unique_points[0] != unique_points[-1]:
            unique_points.append(unique_points[0])

        area = calculate_polygon_area(unique_points)
        if area <= 5:
            return False

        self.ignored_regions.append(unique_points)
        self.current_ignored_points = []
        self.update_display()
        return True

    def save_current_removal_region(self):
        """保存当前去除区域。"""
        if self.is_summary:
            return False
        if len(self.current_removal_points) < 3:
            return False

        unique_points = []
        for point in self.current_removal_points:
            if not unique_points or unique_points[-1] != point:
                unique_points.append(point)
        if len(unique_points) < 3:
            return False

        if unique_points[0] != unique_points[-1]:
            unique_points.append(unique_points[0])

        area = calculate_polygon_area(unique_points)
        if area <= 5:
            return False

        self.removal_regions.append(unique_points)
        self.current_removal_points = []
        self.update_display()
        return True

    def confirm_preview_and_save(self):
        """将当前预览中的多 polygon 手工实例转为正式实例。"""
        if self.is_summary:
            return False

        # 为了保持“可用优先”，保存整株时自动尝试把当前正在绘制的 polygon 暂存。
        # 这样用户在画完最后一个区域后，直接按 Shift+Enter 也能正常保存。
        if self.current_points:
            self.save_current_polygon()

        # 保存当前正在绘制的去除区域
        if self.current_removal_points:
            self.save_current_removal_region()

        if len(self.current_plant_polygons) == 0:
            return False

        # 构建最终多边形列表：第一个为外轮廓，后续为内孔
        final_polygons = []
        
        # 添加外轮廓（第一个主多边形）
        for poly in self.current_plant_polygons:
            if len(poly) >= 3:
                # 确保多边形是闭合的
                if poly[0] != poly[-1]:
                    poly.append(poly[0])
                # 确保外轮廓为顺时针方向
                if self._get_polygon_area(poly) < 0:
                    poly = poly[::-1]  # 反转顶点顺序
                final_polygons.append(poly)
                break  # 只使用第一个主多边形作为外轮廓

        # 添加内孔（所有去除区域）
        for poly in self.removal_regions:
            if len(poly) >= 3:
                # 确保多边形是闭合的
                if poly[0] != poly[-1]:
                    poly.append(poly[0])
                
                # 计算去除区域与外轮廓的交集
                outer_contour = final_polygons[0] if final_polygons else None
                if outer_contour:
                    intersection_poly = self._polygon_intersection(outer_contour, poly)
                    if intersection_poly and len(intersection_poly) >= 3:
                        # 确保内孔为逆时针方向（与外轮廓相反）
                        if self._get_polygon_area(intersection_poly) > 0:
                            intersection_poly = intersection_poly[::-1]  # 反转顶点顺序
                        final_polygons.append(intersection_poly)

        # 如果最终多边形列表为空，返回失败
        if not final_polygons:
            return False

        # 创建实例
        new_instance = make_formal_instance(
            instance_id=self.current_plant_id,
            polygons=final_polygons,
            class_names=self.class_names,
            class_id=0,  # 暂时默认为0，后续可以从UI选择
            source="manual",
            owner_plant_id=self.next_owner_plant_id,
        )

        # 保存实例
        self.plants.append(new_instance)
        saved_id = self.current_plant_id
        self.current_plant_id += 1

        # 清空当前暂存的多边形和去除区域
        self.current_plant_polygons = []
        self.removal_regions = []

        # 通知主窗口更新
        self._notify_annotation_changed()
        self.update_display()

        return saved_id

    def undo_last_action(self):
        """撤销手工绘制中的上一步。"""
        if self.is_summary:
            return False

        if self.current_points:
            self.current_points.pop()
            self.current_snap_point = None
            self.update_display()
            return True
        if self.current_plant_polygons:
            self.current_plant_polygons.pop()
            self.current_snap_point = None
            self.update_display()
            return True
        if self.current_ignored_points:
            self.current_ignored_points.pop()
            self.current_snap_point = None
            self.update_display()
            return True
        if self.current_removal_points:
            self.current_removal_points.pop()
            self.current_snap_point = None
            self.update_display()
            return True
        if self.removal_regions:
            self.removal_regions.pop()
            self.update_display()
            return True
        return False

    def mousePressEvent(self, event):
        """处理鼠标按下事件。"""
        if self.raw_pixmap is None:
            return

        try:
            if event.button() == Qt.RightButton:
                self.is_dragging = True
                self.drag_last_pos = event.pos()
                self.setCursor(QCursor(Qt.ClosedHandCursor))
                return

            if event.button() != Qt.LeftButton or self.is_summary:
                return

            # if self.sam_segmenting:
            #     image_pos = self.screen_to_image(event.pos())
            #     if image_pos:
            #         self.perform_sam_segmentation(image_pos)
            #     return

            if event.modifiers() & Qt.ShiftModifier:
                self.edge_snap_enabled = not self.edge_snap_enabled
                main_win = self.get_main_window()
                if main_win and hasattr(main_win, "update_snap_button_state"):
                    main_win.update_snap_button_state()
                self.update_display()
                return

            image_pos = self.screen_to_image(event.pos())
            if not image_pos:
                return

            if self.ignoring_region:
                if self.current_ignored_points:
                    self.current_ignored_points.append(image_pos)
                else:
                    self.current_ignored_points = [image_pos]
                self._notify_annotation_changed()
                return

            if self.removing_region:
                if self.current_removal_points:
                    self.current_removal_points.append(image_pos)
                else:
                    self.current_removal_points = [image_pos]
                self._notify_annotation_changed()
                return

            if self.current_points or self.current_plant_polygons:
                self._append_current_point(image_pos)
                self._notify_annotation_changed()
                return

            if self.region_growing_enabled:
                self.perform_region_growing(image_pos)
                self._notify_annotation_changed()
                return

            vertex_hit = self._find_vertex_hit(image_pos)
            if vertex_hit:
                self.vertex_drag_info = vertex_hit
                self.select_entity(vertex_hit["kind"], vertex_hit["entity_id"])
                return

            hit_kind, hit_id = self._find_hit_entity(image_pos)
            if hit_kind:
                self.select_entity(hit_kind, hit_id)
                return

            self._append_current_point(image_pos)
            self._notify_annotation_changed()
        except Exception as error:
            print(f"mousePressEvent error: {error}")
            traceback.print_exc()

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件。"""
        if event.button() == Qt.RightButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            return

        if event.button() == Qt.LeftButton and self.vertex_drag_info:
            self.vertex_drag_info = None
            self._notify_annotation_changed()
            main_win = self.get_main_window()
            if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                main_win.on_entity_geometry_modified()

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件。"""
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
            except Exception as error:
                print(f"mouseMoveEvent drag error: {error}")
            return

        if self.vertex_drag_info:
            image_pos = self.screen_to_image(event.pos())
            if image_pos:
                self._update_dragging_vertex(image_pos)
            return

        if not self.is_summary:
            self.current_snap_point = self.calculate_snap_point(event.pos())
            self.update_display()

    def wheelEvent(self, event):
        """处理滚轮缩放。"""
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
            new_scale = min(old_scale * 1.1, self.max_scale) if delta > 0 else max(old_scale * 0.9, self.min_scale)

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
        except Exception as error:
            print(f"wheelEvent error: {error}")
            traceback.print_exc()

    def _append_current_point(self, image_pos):
        """向当前手工 polygon 添加一个点。"""
        if self.edge_snap_enabled and self.current_snap_point is not None:
            self.current_points.append(self.current_snap_point)
        else:
            self.current_points.append((float(image_pos[0]), float(image_pos[1])))
        self.current_snap_point = None
        self.update_display()

    def _find_vertex_hit(self, image_pos):
        """查找当前选中对象是否有顶点命中。"""
        entity_kind, entity = self.get_selected_entity()
        if not entity:
            return None

        hit_radius = self.vertex_hit_radius / max(self.scale_factor, 0.1)
        polygons = entity.get("polygons", [])
        for polygon_index, polygon in enumerate(polygons):
            point_count = len(polygon)
            if point_count < 3:
                continue
            limit = point_count - 1 if polygon[0] == polygon[-1] else point_count
            for point_index in range(limit):
                point = polygon[point_index]
                distance = math.dist((float(point[0]), float(point[1])), (float(image_pos[0]), float(image_pos[1])))
                if distance <= hit_radius:
                    return {
                        "kind": entity_kind,
                        "entity_id": entity.get("id") if entity_kind == "formal" else entity.get("candidate_id"),
                        "polygon_index": polygon_index,
                        "point_index": point_index,
                    }
        return None

    def _update_dragging_vertex(self, image_pos):
        """实时更新拖拽顶点。"""
        drag = self.vertex_drag_info
        if not drag:
            return

        entity_kind, entity = self.get_selected_entity()
        if not entity:
            return

        polygon = entity["polygons"][drag["polygon_index"]]
        polygon[drag["point_index"]] = (float(image_pos[0]), float(image_pos[1]))

        if polygon and polygon[0] == polygon[-1]:
            if drag["point_index"] == 0:
                polygon[-1] = polygon[0]
            elif drag["point_index"] == len(polygon) - 1:
                polygon[0] = polygon[-1]

        if entity_kind == "formal":
            entity["polygons"] = normalize_polygons(entity["polygons"])
            if entity.get("source") in ("ai_accepted", "ai_assisted"):
                entity["source"] = "ai_modified"
            touch_instance(entity)

        self.update_display()
        main_win = self.get_main_window()
        if main_win and entity_kind == "formal":
            main_win.sync_summary_view()

    def _find_hit_entity(self, image_pos):
        """按点击位置查找实例命中，候选优先于正式层。"""
        for candidate in reversed(self.candidate_instances):
            if self._point_hits_polygons(image_pos, candidate.get("polygons", [])):
                return "candidate", candidate.get("candidate_id")

        for plant in reversed(self.plants):
            if self._point_hits_polygons(image_pos, plant.get("polygons", [])):
                return "formal", plant.get("id")

        return None, None

    def _point_hits_polygons(self, image_pos, polygons):
        """判断点是否命中 polygon。"""
        point = (float(image_pos[0]), float(image_pos[1]))
        for polygon in polygons or []:
            contour = np.array(polygon, dtype=np.float32)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return True
        return False

    def _get_polygon_area(self, polygon):
        """计算多边形的面积，用于判断顶点顺序。
        面积为正表示逆时针，面积为负表示顺时针。"""
        area = 0.0
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)
        return area / 2.0

    def _polygon_intersection(self, poly1, poly2):
        """计算两个多边形的交集。"""
        try:
            import cv2
            import numpy as np
            
            # 确保多边形是闭合的
            if poly1[0] != poly1[-1]:
                poly1 = poly1 + [poly1[0]]
            if poly2[0] != poly2[-1]:
                poly2 = poly2 + [poly2[0]]
            
            # 转换为numpy数组
            poly1_np = np.array(poly1, dtype=np.int32)
            poly2_np = np.array(poly2, dtype=np.int32)
            
            # 创建一个足够大的画布
            all_points = poly1 + poly2
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            # 确保画布大小合理
            width = max(1, int(max_x - min_x) + 10)
            height = max(1, int(max_y - min_y) + 10)
            
            # 创建掩码
            mask1 = np.zeros((height, width), dtype=np.uint8)
            mask2 = np.zeros((height, width), dtype=np.uint8)
            
            # 调整多边形坐标到画布坐标系
            poly1_adjusted = [(int(p[0] - min_x), int(p[1] - min_y)) for p in poly1]
            poly2_adjusted = [(int(p[0] - min_x), int(p[1] - min_y)) for p in poly2]
            
            # 绘制多边形
            cv2.fillPoly(mask1, [np.array(poly1_adjusted)], 255)
            cv2.fillPoly(mask2, [np.array(poly2_adjusted)], 255)
            
            # 计算交集
            intersection = cv2.bitwise_and(mask1, mask2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 取最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 转换回原始坐标系
            intersection_poly = []
            for point in largest_contour:
                x = point[0][0] + min_x
                y = point[0][1] + min_y
                intersection_poly.append((float(x), float(y)))
            
            # 确保多边形是闭合的
            if intersection_poly and intersection_poly[0] != intersection_poly[-1]:
                intersection_poly.append(intersection_poly[0])
            
            return intersection_poly
        except Exception as e:
            print(f"_polygon_intersection error: {e}")
            return None

    def calculate_bbox_from_polygons(self, polygons):
        """从多边形列表中计算边界框。"""
        if not polygons:
            return [0, 0, 0, 0]
        
        x_coords = []
        y_coords = []
        
        for polygon in polygons:
            for point in polygon:
                x_coords.append(point[0])
                y_coords.append(point[1])
        
        if not x_coords or not y_coords:
            return [0, 0, 0, 0]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def calculate_snap_point(self, screen_pos):
        """计算边缘吸附点。"""
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

            x1 = max(0, int(img_x - self.snap_radius))
            y1 = max(0, int(img_y - self.snap_radius))
            x2 = min(img_w, int(img_x + self.snap_radius + 1))
            y2 = min(img_h, int(img_y + self.snap_radius + 1))

            roi_edges = self.edge_map[y1:y2, x1:x2]
            edge_points = np.column_stack(np.where(roi_edges > 0))
            if len(edge_points) == 0:
                return None

            mouse_roi = np.array([img_y - y1, img_x - x1])
            distances = np.linalg.norm(edge_points - mouse_roi, axis=1)
            min_idx = int(np.argmin(distances))
            min_dist = float(distances[min_idx])
            if min_dist <= self.snap_radius:
                snap_y = y1 + edge_points[min_idx][0]
                snap_x = x1 + edge_points[min_idx][1]
                return (float(snap_x), float(snap_y))
            return None
        except Exception as error:
            print(f"calculate_snap_point error: {error}")
            return None

    def screen_to_image(self, screen_pos):
        """屏幕坐标转图像坐标。"""
        if not self.raw_pixmap:
            return None
        try:
            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor
            img_x = (screen_pos.x() - offset_x) / self.scale_factor
            img_y = (screen_pos.y() - offset_y) / self.scale_factor

            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                return img_x, img_y
            return None
        except Exception as error:
            print(f"screen_to_image error: {error}")
            return None

    def image_to_screen(self, image_pos):
        """图像坐标转屏幕坐标。"""
        if not self.raw_pixmap:
            return None
        try:
            offset_x = self.width() / 2 - self.view_center_x * self.scale_factor
            offset_y = self.height() / 2 - self.view_center_y * self.scale_factor
            screen_x = image_pos[0] * self.scale_factor + offset_x
            screen_y = image_pos[1] * self.scale_factor + offset_y
            return screen_x, screen_y
        except Exception as error:
            print(f"image_to_screen error: {error}")
            return None

    def get_view_rect(self):
        """获取当前视图区域在图像坐标系中的矩形。"""
        if not self.raw_pixmap:
            return None
        try:
            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            
            # 计算视图中心在图像中的坐标
            center_x = self.view_center_x
            center_y = self.view_center_y
            
            # 计算视图区域的半宽和半高
            view_half_width = (self.width() / 2) / self.scale_factor
            view_half_height = (self.height() / 2) / self.scale_factor
            
            # 计算视图区域的边界
            x1 = max(0, center_x - view_half_width)
            y1 = max(0, center_y - view_half_height)
            x2 = min(img_width, center_x + view_half_width)
            y2 = min(img_height, center_y + view_half_height)
            
            return (x1, y1, x2, y2)
        except Exception as error:
            print(f"get_view_rect error: {error}")
            return None

    def perform_region_growing(self, seed_point):
        """执行区域生长。"""
        if self.color_image is None:
            return

        progress = QProgressDialog("正在执行膨胀点选...", "取消", 0, 100, self.get_main_window())
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.show()

        def progress_callback(value):
            progress.setValue(value)

        try:
            mask = perform_region_growing(
                self.color_image,
                seed_point,
                self.region_growing_threshold,
                progress_callback,
            )
            if mask is not None:
                self.region_growing_mask = mask
                self.current_points = convert_mask_to_polygon(mask)
        except Exception as error:
            print(f"Region growing error: {error}")
        finally:
            progress.close()

        self.update_display()

    # def perform_sam_segmentation(self, point=None):
    #     """执行 SAM 分割。"""
    #     if not self.sam_predictor or self.color_image is None:
    #         return

    #     try:
    #         if point is not None:
    #             self.sam_prompt_points.append([float(point[0]), float(point[1])])

    #         if not self.sam_prompt_points:
    #             return

    #         point_coords = np.array(self.sam_prompt_points)
    #         point_labels = np.ones(len(point_coords), dtype=np.int32)
    #         masks, _, _ = self.sam_predictor.predict(
    #             point_coords=point_coords,
    #             point_labels=point_labels,
    #             multimask_output=True,
    #         )

    #         if masks is not None and len(masks) > 0:
    #             best_mask_idx = int(np.argmax([np.sum(mask) for mask in masks]))
    #             self.sam_mask = masks[best_mask_idx]
    #             mask = (self.sam_mask * 255).astype(np.uint8)
    #             self.current_points = convert_mask_to_polygon(mask)
    #     except Exception as error:
    #         print(f"SAM segmentation error: {error}")

    #     self.update_display()

    def update_display(self):
        """更新显示。"""
        if self.raw_pixmap is None:
            self.clear()
            return

        try:
            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            scaled_width = int(img_width * self.scale_factor)
            scaled_height = int(img_height * self.scale_factor)

            scaled_pixmap = self.raw_pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
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
                    int(pt[1] * self.scale_factor + offset_y),
                )

            # 绘制已保存的忽略区域
            for region in self.ignored_regions:
                if len(region) >= 3:
                    qpts = [img_to_screen(point) for point in region]
                    # 使用黑色斜线阴影，透明度0%
                    brush = QBrush()
                    brush.setColor(QColor(0, 0, 0, 255))  # 黑色，完全不透明
                    brush.setStyle(Qt.Dense7Pattern)  # 斜线阴影模式
                    painter.setBrush(brush)
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.drawPolygon(*qpts)

            if self.is_summary:
                self._draw_formal_instances(painter, img_to_screen, summary_mode=True)
            else:
                self._draw_formal_instances(painter, img_to_screen, summary_mode=False)
                self._draw_candidate_instances(painter, img_to_screen)
                self._draw_current_preview(painter, img_to_screen)
                self._draw_snap_point(painter, img_to_screen)

            # 绘制投影框
            if hasattr(self, 'projection_rect') and self.projection_rect:
                x1, y1, x2, y2 = self.projection_rect
                qpt1 = img_to_screen((x1, y1))
                qpt2 = img_to_screen((x2, y2))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(0, 0, 255), 2, Qt.DashLine))
                painter.drawRect(qpt1.x(), qpt1.y(), qpt2.x() - qpt1.x(), qpt2.y() - qpt1.y())

            painter.end()
            self.setPixmap(final_pixmap)
        except Exception as error:
            print(f"update_display error: {error}")
            traceback.print_exc()

    def _draw_formal_instances(self, painter, img_to_screen, summary_mode):
        """绘制正式层。"""
        for plant in self.plants:
            plant_color = QColor(*plant.get("color", get_plant_color(int(plant.get("id", 0)))))
            is_selected = self.selected_entity_kind == "formal" and int(self.selected_entity_id or 0) == int(plant.get("id", 0))

            polygons = plant.get("polygons", [])
            if polygons:
                # 创建临时QPixmap用于绘制填充和去除区域
                temp_pixmap = QPixmap(self.size())
                temp_pixmap.fill(Qt.transparent)
                temp_painter = QPainter(temp_pixmap)
                temp_painter.setRenderHint(QPainter.Antialiasing)
                
                # 绘制外轮廓（第一个多边形）
                if len(polygons[0]) >= 3:
                    qpts = [img_to_screen(point) for point in polygons[0]]
                    if summary_mode:
                        temp_painter.setBrush(QBrush(plant_color))
                        temp_painter.setPen(QPen(QColor(255, 0, 0), 3) if is_selected else QPen(QColor(0, 0, 0), 1))
                    else:
                        weak_color = QColor(plant_color.red(), plant_color.green(), plant_color.blue(), 55)
                        temp_painter.setBrush(QBrush(weak_color))
                        temp_painter.setPen(QPen(QColor(255, 80, 80), 3) if is_selected else QPen(QColor(100, 100, 100), 1))
                    temp_painter.drawPolygon(*qpts)
                
                # 绘制内孔（后续多边形）
                for i in range(1, len(polygons)):
                    polygon = polygons[i]
                    if len(polygon) >= 3:
                        qpts = [img_to_screen(point) for point in polygon]
                        # 使用组合模式实现挖空
                        temp_painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
                        temp_painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
                        temp_painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
                        temp_painter.drawPolygon(*qpts)
                
                temp_painter.end()
                # 将临时pixmap绘制到主画布
                painter.drawPixmap(0, 0, temp_pixmap)

            if not summary_mode and is_selected:
                self._draw_polygon_vertices(painter, img_to_screen, plant.get("polygons", []), QColor(255, 60, 60))

    def _draw_candidate_instances(self, painter, img_to_screen):
        """绘制候选层。"""
        for candidate in self.candidate_instances:
            is_selected = (
                self.selected_entity_kind == "candidate"
                and self.selected_entity_id == candidate.get("candidate_id")
            )
            fill_color = QColor(0, 180, 255, 60 if not is_selected else 90)
            line_color = QColor(0, 150, 255) if not is_selected else QColor(255, 165, 0)
            painter.setPen(QPen(line_color, 2, Qt.DashLine))
            painter.setBrush(QBrush(fill_color))
            for polygon in candidate.get("polygons", []):
                if len(polygon) < 3:
                    continue
                painter.drawPolygon(*[img_to_screen(point) for point in polygon])
            if is_selected:
                self._draw_polygon_vertices(painter, img_to_screen, candidate.get("polygons", []), QColor(0, 150, 255))

    def _draw_current_preview(self, painter, img_to_screen):
        """绘制当前手工绘制中的预览层。"""
        # 创建临时QPixmap用于绘制填充和去除区域
        if self.current_plant_polygons or self.removal_regions:
            temp_pixmap = QPixmap(self.size())
            temp_pixmap.fill(Qt.transparent)
            temp_painter = QPainter(temp_pixmap)
            temp_painter.setRenderHint(QPainter.Antialiasing)
            
            # 绘制主多边形（填充区域）
            if self.current_plant_polygons:
                temp_color = QColor(100, 200, 100, 120)
                temp_painter.setBrush(QBrush(temp_color))
                temp_painter.setPen(QPen(QColor(0, 150, 0), 2))
                for polygon in self.current_plant_polygons:
                    if len(polygon) >= 3:
                        qpts = [img_to_screen(point) for point in polygon]
                        temp_painter.drawPolygon(*qpts)
            
            # 绘制去除区域（挖空效果）
            if self.removal_regions:
                for region in self.removal_regions:
                    if len(region) >= 3:
                        qpts = [img_to_screen(point) for point in region]
                        # 使用组合模式实现挖空
                        temp_painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
                        temp_painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
                        temp_painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
                        temp_painter.drawPolygon(*qpts)
            
            temp_painter.end()
            # 将临时pixmap绘制到主画布
            painter.drawPixmap(0, 0, temp_pixmap)

        # 绘制当前正在绘制的点
        if self.current_points:
            qpts = [img_to_screen(point) for point in self.current_points]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            for point in qpts:
                painter.drawEllipse(point, 4, 4)

            if len(qpts) > 1:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                for index in range(len(qpts) - 1):
                    painter.drawLine(qpts[index], qpts[index + 1])

            if self.underMouse() and self.last_mouse_pos:
                painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
                painter.drawLine(qpts[-1], self.last_mouse_pos)
        
        # 绘制当前正在绘制的忽略区域
        if self.current_ignored_points:
            qpts = [img_to_screen(point) for point in self.current_ignored_points]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            for point in qpts:
                painter.drawEllipse(point, 4, 4)

            if len(qpts) > 1:
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                for index in range(len(qpts) - 1):
                    painter.drawLine(qpts[index], qpts[index + 1])

            if self.underMouse() and self.last_mouse_pos:
                painter.setPen(QPen(QColor(0, 0, 0), 2, Qt.DashLine))
                painter.drawLine(qpts[-1], self.last_mouse_pos)

        # 绘制当前正在绘制的去除区域
        if self.current_removal_points:
            qpts = [img_to_screen(point) for point in self.current_removal_points]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            for point in qpts:
                painter.drawEllipse(point, 4, 4)

            if len(qpts) > 1:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                for index in range(len(qpts) - 1):
                    painter.drawLine(qpts[index], qpts[index + 1])

            if self.underMouse() and self.last_mouse_pos:
                painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
                painter.drawLine(qpts[-1], self.last_mouse_pos)

    def _draw_snap_point(self, painter, img_to_screen):
        """绘制边缘吸附预览点。"""
        if self.current_snap_point is not None and self.edge_snap_enabled:
            snap_screen = img_to_screen(self.current_snap_point)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawEllipse(snap_screen, 8, 8)
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(snap_screen, 3, 3)

    def _draw_polygon_vertices(self, painter, img_to_screen, polygons, color):
        """绘制选中对象的顶点和连线。"""
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        for polygon in polygons or []:
            limit = len(polygon) - 1 if polygon and polygon[0] == polygon[-1] else len(polygon)
            # 绘制顶点
            for index in range(limit):
                painter.drawEllipse(img_to_screen(polygon[index]), 4, 4)
            # 绘制连线
            if limit > 1:
                painter.setPen(QPen(color, 2))
                for index in range(limit - 1):
                    painter.drawLine(img_to_screen(polygon[index]), img_to_screen(polygon[index + 1]))
                # 闭合多边形
                if polygon and polygon[0] == polygon[-1]:
                    painter.drawLine(img_to_screen(polygon[-2]), img_to_screen(polygon[0]))
                else:
                    painter.drawLine(img_to_screen(polygon[-1]), img_to_screen(polygon[0]))

    def _notify_selection_changed(self):
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "on_canvas_entity_selected"):
            main_win.on_canvas_entity_selected(self.selected_entity_kind, self.selected_entity_id)

    def _notify_annotation_changed(self):
        main_win = self.get_main_window()
        if main_win:
            if hasattr(main_win, "mark_annotation_changed"):
                main_win.mark_annotation_changed()
            if hasattr(main_win, "update_status_bar"):
                main_win.update_status_bar()
            if hasattr(main_win, "sync_summary_view") and self.selected_entity_kind != "candidate":
                main_win.sync_summary_view()
            if hasattr(main_win, "refresh_properties_panel"):
                main_win.refresh_properties_panel()

    def get_main_window(self):
        """获取主窗口。"""
        parent = self.parent()
        while parent and not hasattr(parent, "toggle_edge_snap"):
            parent = parent.parent()
        return parent