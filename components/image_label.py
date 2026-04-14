# 图像标注控件
import math
import traceback
import copy

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QProgressDialog, QSizePolicy

from config import SNAP_RADIUS
from utils.annotation_schema import (
    make_formal_instance,
    next_instance_id,
    normalize_candidate_instance,
    normalize_formal_instance,
    normalize_polygons,
    touch_instance,
)
from utils.auxiliary_algorithms import convert_mask_to_polygon, perform_region_growing
from utils.helpers import calculate_polygon_area, get_plant_color
from utils.image_processor import preprocess_image
from utils.sam_utils import mask_to_polygons, process_sam_polygons


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
        self.current_points = []
        self.current_plant_polygons = []
        self.current_plant_labels = []  # 存储每个暂存区域的 label
        self.current_plant_id = 1
        self.selected_plant_id = None
        self.next_owner_plant_id = None
        self.delete_plant_stack = []  # 存储删除植株的操作，深度为2

        self.candidate_instances = []
        self.selected_entity_kind = None
        self.selected_entity_id = None


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
        
        # 操作栈
        self.main_stack = []  # 标注区域和去除区域共用的栈
        self.ignore_stack = []  # 忽略区域操作栈
        self.redo_main_stack = []  # 标注区域和去除区域共用的redo栈
        self.redo_ignore_stack = []  # 忽略区域操作的redo栈
        self.fine_tune_stack = []  # 微调模式的undo栈
        self.fine_tune_redo_stack = []  # 微调模式的redo栈
        self.max_stack_depth = 20  # 栈深度限制

        # 模式管理
        self.mode = "normal"  # normal, select_and_fine_tune, fine_tune
        self.fine_tune_instance_id = None  # 微调模式下选中的实例ID
        self.fine_tune_original_data = {}  # 微调前的原始数据
        self.add_vertex_mode = False  # 是否处于添加顶点模式
        self.delete_vertex_mode = False  # 是否处于删除顶点模式
        self.split_staging_mode = False
        self.split_line_dragging = False
        self.split_line_start = None
        self.split_line_end = None
        self.merge_staging_mode = False
        self.merge_staging_clicks = []
        self.preannotation_box_mode = False
        self.preannotation_box_dragging = False
        self.preannotation_box_start = None
        self.preannotation_box_end = None
        self.preannotation_box_rect = None

    def set_mode(self, mode):
        """设置操作模式"""
        self.mode = mode

    def set_preannotation_box_mode(self, enabled):
        """切换预标注框选模式。"""
        self.preannotation_box_mode = bool(enabled)
        if not enabled:
            self.preannotation_box_dragging = False
            self.preannotation_box_start = None
            self.preannotation_box_end = None
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()

    def clear_preannotation_box(self):
        """清除当前框选预览。"""
        self.preannotation_box_dragging = False
        self.preannotation_box_start = None
        self.preannotation_box_end = None
        self.preannotation_box_rect = None
        self.update_display()
    
    def set_split_staging_mode(self, enabled):
        self.split_staging_mode = bool(enabled)
        if not enabled:
            self.split_line_dragging = False
            self.split_line_start = None
            self.split_line_end = None
        elif self.merge_staging_mode:
            self.set_merge_staging_mode(False)
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()

    def set_merge_staging_mode(self, enabled):
        self.merge_staging_mode = bool(enabled)
        if not enabled:
            self.merge_staging_clicks = []
        elif self.split_staging_mode:
            self.set_split_staging_mode(False)
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()

    def _is_fine_tune_interaction_active(self):
        return self.mode == "fine_tune" and bool(self.fine_tune_instance_id)

    @staticmethod
    def _label_for_index(labels, index, default="stem"):
        if index < len(labels) and labels[index]:
            return labels[index]
        return default

    @staticmethod
    def _make_staging_entity_id(owner_kind, owner_id, polygon_index):
        if owner_kind == "preview":
            return f"preview:{int(polygon_index)}"
        return f"formal:{int(owner_id)}:{int(polygon_index)}"

    @staticmethod
    def _make_removal_entity_id(owner_kind, owner_id, polygon_index):
        if owner_kind == "preview":
            return f"preview_removal:{int(polygon_index)}"
        return f"formal_removal:{int(owner_id)}:{int(polygon_index)}"

    @staticmethod
    def _parse_staging_entity_id(entity_id):
        parts = str(entity_id or "").split(":")
        if len(parts) == 2 and parts[0] == "preview":
            try:
                return {"owner_kind": "preview", "owner_id": None, "polygon_index": int(parts[1])}
            except (TypeError, ValueError):
                return None
        if len(parts) == 3 and parts[0] == "formal":
            try:
                return {"owner_kind": "formal", "owner_id": int(parts[1]), "polygon_index": int(parts[2])}
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_removal_entity_id(entity_id):
        parts = str(entity_id or "").split(":")
        if len(parts) == 2 and parts[0] == "preview_removal":
            try:
                return {"owner_kind": "preview", "owner_id": None, "polygon_index": int(parts[1])}
            except (TypeError, ValueError):
                return None
        if len(parts) == 3 and parts[0] == "formal_removal":
            try:
                return {"owner_kind": "formal", "owner_id": int(parts[1]), "polygon_index": int(parts[2])}
            except (TypeError, ValueError):
                return None
        return None

    def _ensure_label_slots(self, labels, polygon_count):
        normalized = list(labels or [])
        while len(normalized) < polygon_count:
            normalized.append("stem")
        return normalized

    def _is_outer_polygon(self, polygon):
        return len(polygon or []) >= 3 and self._get_polygon_area(polygon) <= 0

    def _get_outer_polygon_indices(self, polygons):
        return [index for index, polygon in enumerate(polygons or []) if self._is_outer_polygon(polygon)]

    def _normalize_labels_for_polygons(self, labels, polygons):
        outer_indices = self._get_outer_polygon_indices(polygons)
        normalized = list(labels or [])[:len(outer_indices)]
        while len(normalized) < len(outer_indices):
            normalized.append("stem")
        return normalized, outer_indices

    def _get_inner_polygon_indices(self, polygons):
        return [index for index, polygon in enumerate(polygons or []) if len(polygon or []) >= 3 and self._get_polygon_area(polygon) > 0]

    def _split_instance_polygons_for_preview(self, polygons, labels=None):
        normalized_polygons = normalize_polygons(copy.deepcopy(polygons or []))
        normalized_labels, outer_indices = self._normalize_labels_for_polygons(labels or [], normalized_polygons)
        removal_indices = self._get_inner_polygon_indices(normalized_polygons)
        preview_polygons = [copy.deepcopy(normalized_polygons[index]) for index in outer_indices]
        preview_labels = [self._label_for_index(normalized_labels, index) for index in range(len(outer_indices))]
        removal_regions = [copy.deepcopy(normalized_polygons[index]) for index in removal_indices]
        return preview_polygons, preview_labels, removal_regions

    def _extract_split_polygons_from_mask(self, mask, min_x, min_y, source_polygon):
        split_polygons = []

        local_polygons = process_sam_polygons(mask_to_polygons(mask, pixel_interval=50))
        for polygon in local_polygons:
            shifted_points = [(float(point[0] + min_x), float(point[1] + min_y)) for point in polygon]
            normalized = normalize_polygons([shifted_points])
            if not normalized:
                continue
            split_polygons.append(normalized[0])

        split_polygons.sort(key=lambda item: abs(calculate_polygon_area(item)), reverse=True)
        if len(split_polygons) > 2:
            split_polygons = split_polygons[:2]
        return split_polygons

    @staticmethod
    def _same_staging_owner(first_staging, second_staging):
        if not first_staging or not second_staging:
            return False
        if first_staging.get("owner_kind") != second_staging.get("owner_kind"):
            return False
        if first_staging.get("owner_kind") == "formal":
            return int(first_staging.get("owner_id") or 0) == int(second_staging.get("owner_id") or 0)
        return True

    def _extract_single_polygon_from_mask(self, mask, min_x, min_y):
        polygons = []
        local_polygons = process_sam_polygons(mask_to_polygons(mask, pixel_interval=50))
        for polygon in local_polygons:
            shifted_points = [(float(point[0] + min_x), float(point[1] + min_y)) for point in polygon]
            normalized = normalize_polygons([shifted_points])
            if normalized:
                polygons.append(normalized[0])
        polygons.sort(key=lambda item: abs(calculate_polygon_area(item)), reverse=True)
        return polygons

    @staticmethod
    def _open_polygon_vertices(polygon):
        points = list(polygon or [])
        if len(points) >= 2 and points[0] == points[-1]:
            points = points[:-1]
        return points

    @staticmethod
    def _cyclic_path(points, start_index, end_index):
        count = len(points)
        if count == 0:
            return []
        path = [points[start_index]]
        index = start_index
        while index != end_index:
            index = (index + 1) % count
            path.append(points[index])
        return path

    def _path_distance_score(self, path_points, other_polygon):
        contour = np.array(other_polygon, dtype=np.float32)
        if contour.ndim != 2 or contour.shape[0] < 3:
            return float("inf")

        samples = []
        for index, point in enumerate(path_points):
            samples.append(point)
            if index + 1 < len(path_points):
                next_point = path_points[index + 1]
                samples.append(
                    (
                        (float(point[0]) + float(next_point[0])) / 2.0,
                        (float(point[1]) + float(next_point[1])) / 2.0,
                    )
                )

        if not samples:
            return float("inf")

        distances = []
        for sample in samples:
            distance = cv2.pointPolygonTest(
                contour,
                (float(sample[0]), float(sample[1])),
                True,
            )
            distances.append(abs(float(distance)))
        return sum(distances) / len(distances)

    def _select_preserved_merge_path(self, polygon, start_index, end_index, other_polygon):
        vertices = self._open_polygon_vertices(polygon)
        if len(vertices) < 3:
            return None
        if start_index == end_index:
            return None
        if start_index >= len(vertices) or end_index >= len(vertices):
            return None

        forward_path = self._cyclic_path(vertices, start_index, end_index)
        backward_path = list(reversed(self._cyclic_path(vertices, end_index, start_index)))
        if len(forward_path) < 2 or len(backward_path) < 2:
            return None

        forward_score = self._path_distance_score(forward_path, other_polygon)
        backward_score = self._path_distance_score(backward_path, other_polygon)
        if forward_score > backward_score:
            return forward_path
        if backward_score > forward_score:
            return backward_path

        if len(forward_path) > len(backward_path):
            return forward_path
        return backward_path

    @staticmethod
    def _append_unique_points(target, points):
        for point in points:
            normalized_point = (float(point[0]), float(point[1]))
            if not target or target[-1] != normalized_point:
                target.append(normalized_point)
        return target

    def _merge_polygons_via_points(self, first_click, second_click, bridge_width=5):
        first_staging = self._resolve_staging_entity((first_click or {}).get("staging_id"))
        second_staging = self._resolve_staging_entity((second_click or {}).get("staging_id"))
        if not first_staging or not second_staging:
            return None, "请依次点击两个暂存区域"

        first_polygon = self._open_polygon_vertices(first_staging.get("polygon", []))
        second_polygon = self._open_polygon_vertices(second_staging.get("polygon", []))
        if len(first_polygon) < 3 or len(second_polygon) < 3:
            return None, "用于合并的区域无效"

        first_clicks = [click for click in self.merge_staging_clicks if click.get("staging_id") == first_click.get("staging_id")]
        second_clicks = [click for click in self.merge_staging_clicks if click.get("staging_id") == second_click.get("staging_id")]
        if len(first_clicks) < 2 or len(second_clicks) < 2:
            return None, "每个暂存区域需要点击两个点"

        first_indices = [int(click.get("point_index", -1)) for click in first_clicks[:2]]
        second_indices = [int(click.get("point_index", -1)) for click in second_clicks[:2]]
        if any(index < 0 for index in first_indices + second_indices):
            return None, "请选择有效的顶点"
        if len(set(first_indices)) < 2 or len(set(second_indices)) < 2:
            return None, "同一区域的两个点不能是同一个顶点"

        first_preserved_path = self._select_preserved_merge_path(
            first_polygon,
            first_indices[0],
            first_indices[1],
            second_polygon,
        )
        second_preserved_path = self._select_preserved_merge_path(
            second_polygon,
            second_indices[0],
            second_indices[1],
            first_polygon,
        )
        if not first_preserved_path or not second_preserved_path:
            return None, "无法根据选中的顶点生成合并边界"

        merged_polygon = []
        self._append_unique_points(merged_polygon, first_preserved_path)
        self._append_unique_points(merged_polygon, list(reversed(second_preserved_path)))
        if len(merged_polygon) < 3:
            return None, "合并后未形成有效区域"
        if merged_polygon[0] != merged_polygon[-1]:
            merged_polygon.append(merged_polygon[0])

        normalized = normalize_polygons([merged_polygon])
        if not normalized:
            return None, "合并后未形成有效区域"
        merged_polygon = normalized[0]
        if self._get_polygon_area(merged_polygon) > 0:
            merged_polygon = merged_polygon[::-1]
        return merged_polygon, None

    def _handle_merge_staging_click(self, image_pos):
        staging_areas = list(self._iter_active_fine_tune_staging_areas())
        vertex_hit = self._find_vertex_hit(image_pos, entity_candidates=staging_areas)
        if vertex_hit is not None:
            staging_id = str(vertex_hit.get("entity_id"))
            staging = self._resolve_staging_entity(staging_id)
            if not staging:
                return False, "请先点击暂存区域"

            point = staging.get("polygon", [])[vertex_hit["point_index"]]
            click_entry = {
                "staging_id": staging_id,
                "point": (float(point[0]), float(point[1])),
                "point_index": int(vertex_hit["point_index"]),
            }
            self.select_entity("staging", staging_id)

            click_count = len(self.merge_staging_clicks)
            if click_count == 0:
                self.merge_staging_clicks = [click_entry]
                self.update_display()
                return True, "请在第一个暂存区域中点击第二个顶点"

            first_click = self.merge_staging_clicks[0]
            first_staging = self._resolve_staging_entity(first_click.get("staging_id"))
            if not first_staging:
                self.merge_staging_clicks = [click_entry]
                self.update_display()
                return True, "请在第一个暂存区域中点击第二个顶点"

            if click_count == 1:
                if staging_id != first_click.get("staging_id"):
                    self.merge_staging_clicks = [click_entry]
                    self.update_display()
                    return False, "前两个点需要落在同一个暂存区域上"
                self.merge_staging_clicks.append(click_entry)
                self.update_display()
                return True, "请在第二个暂存区域中点击第一个顶点"

            second_staging_id = self.merge_staging_clicks[2].get("staging_id") if click_count >= 3 else None
            second_staging = self._resolve_staging_entity(second_staging_id) if second_staging_id else None

            if click_count == 2:
                if staging_id == first_click.get("staging_id"):
                    self.merge_staging_clicks = [click_entry]
                    self.update_display()
                    return False, "第三个点需要落在另一个暂存区域上"
                if not self._same_staging_owner(first_staging, staging):
                    self.merge_staging_clicks = [click_entry]
                    self.update_display()
                    return False, "只能合并同一实例下的两个暂存区域"

                first_label = first_staging.get("label") or "stem"
                second_label = staging.get("label") or "stem"
                if first_label != second_label:
                    from PyQt5.QtWidgets import QInputDialog

                    label, ok = QInputDialog.getItem(
                        self.parent(),
                        "选择合并后的标签",
                        f"两个暂存区域的标签不一致\n第一个区域: {first_label}\n第二个区域: {second_label}\n请选择合并后的标签:",
                        [first_label, second_label, "stem"],
                        0,
                        False,
                    )
                    if not ok:
                        self.update_display()
                        return False, "取消合并操作"

                    self.select_entity("staging", staging_id)
                    self.update_selected_staging_label(label)
                    self.select_entity("staging", first_click.get("staging_id"))
                    self.update_selected_staging_label(label)

                self.merge_staging_clicks.append(click_entry)
                self.update_display()
                return True, "请在第二个暂存区域中点击第二个顶点"

            if click_count == 3:
                if not second_staging or staging_id != second_staging_id:
                    self.merge_staging_clicks = [click_entry]
                    self.update_display()
                    return False, "后两个点需要落在同一个暂存区域上"

                self.merge_staging_clicks.append(click_entry)
                self.update_display()

                success, message = self.merge_selected_staging_polygons(
                    self.merge_staging_clicks[0],
                    self.merge_staging_clicks[2],
                )
                if not success:
                    self.merge_staging_clicks = self.merge_staging_clicks[:2]
                    self.update_display()
                    return False, message
                self.merge_staging_clicks = []
                return True, message

            self.merge_staging_clicks = [click_entry]
            self.update_display()
            return True, "已重置选择，请在第一个暂存区域中点击第一个顶点"

        return False, "请点击暂存区域的顶点"


    def merge_selected_staging_polygons(self, first_click, second_click, bridge_width=5):
        first_staging = self._resolve_staging_entity((first_click or {}).get("staging_id"))
        second_staging = self._resolve_staging_entity((second_click or {}).get("staging_id"))
        if not first_staging or not second_staging:
            return False, "请依次点击两个暂存区域"
        if not self._same_staging_owner(first_staging, second_staging):
            return False, "只能合并同一实例下的两个暂存区域"

        merged_polygon, error_message = self._merge_polygons_via_points(
            first_click,
            second_click,
            bridge_width=bridge_width,
        )
        if merged_polygon is None:
            return False, error_message or "暂存区域合并失败"

        if first_staging["owner_kind"] == "preview":
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            first_index, second_index = sorted((int(first_staging["polygon_index"]), int(second_staging["polygon_index"])))
            labels = self._ensure_label_slots(self.current_plant_labels, len(old_polygons))
            merged_label = first_staging.get("label") or second_staging.get("label") or "stem"

            new_polygons = []
            for index, polygon in enumerate(self.current_plant_polygons):
                if index == first_index:
                    new_polygons.append(merged_polygon)
                elif index == second_index:
                    continue
                else:
                    new_polygons.append(polygon)
            self.current_plant_polygons = normalize_polygons(new_polygons)
            self.current_plant_labels = labels[:first_index] + [merged_label] + labels[first_index + 1:second_index] + labels[second_index + 1:]
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "merge_polygon",
                {
                    "first_polygon_index": first_index,
                    "second_polygon_index": second_index,
                    "first_point": copy.deepcopy(first_click.get("point")),
                    "second_point": copy.deepcopy(second_click.get("point")),
                },
            )
            self.select_entity("staging", self._make_staging_entity_id("preview", None, first_index))
        else:
            plant = first_staging.get("plant")
            if not plant or int(plant.get("id", 0)) != int(second_staging.get("owner_id") or 0):
                return False, "只能合并同一实例下的两个暂存区域"

            old_polygons = copy.deepcopy(plant.get("polygons", []))
            old_labels = copy.deepcopy(plant.get("labels", []))
            first_label_index, second_label_index = sorted(
                (int(first_staging["polygon_index"]), int(second_staging["polygon_index"]))
            )
            first_actual_index, second_actual_index = sorted(
                (
                    int(first_staging.get("actual_polygon_index", first_label_index)),
                    int(second_staging.get("actual_polygon_index", second_label_index)),
                )
            )
            merged_label = first_staging.get("label") or second_staging.get("label") or "stem"
            polygons = list(plant.get("polygons", []) or [])
            merged_polygons = []
            for index, polygon in enumerate(polygons):
                if index == first_actual_index:
                    merged_polygons.append(merged_polygon)
                elif index == second_actual_index:
                    continue
                else:
                    merged_polygons.append(polygon)
            plant["polygons"] = normalize_polygons(merged_polygons)

            labels = list(plant.get("labels", []) or [])
            while len(labels) <= max(first_label_index, second_label_index):
                labels.append("stem")
            plant["labels"] = (
                labels[:first_label_index]
                + [merged_label]
                + labels[first_label_index + 1:second_label_index]
                + labels[second_label_index + 1:]
            )
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "merge_polygon",
                {
                    "first_polygon_index": first_label_index,
                    "second_polygon_index": second_label_index,
                    "first_actual_polygon_index": first_actual_index,
                    "second_actual_polygon_index": second_actual_index,
                    "first_point": copy.deepcopy(first_click.get("point")),
                    "second_point": copy.deepcopy(second_click.get("point")),
                },
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "merge_staging_polygon",
                {
                    "first_polygon_index": first_label_index,
                    "second_polygon_index": second_label_index,
                    "first_actual_polygon_index": first_actual_index,
                    "second_actual_polygon_index": second_actual_index,
                    "first_point": copy.deepcopy(first_click.get("point")),
                    "second_point": copy.deepcopy(second_click.get("point")),
                },
            )
            self.select_entity("staging", self._make_staging_entity_id("formal", plant.get("id"), first_label_index))

        self.set_merge_staging_mode(False)
        self._notify_annotation_changed()
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
            main_win.on_entity_geometry_modified()
        self.update_display()
        return True, "暂存区域已合并"

    def _compose_preview_instance_data(self):
        preview_polygons = normalize_polygons(copy.deepcopy(self.current_plant_polygons))
        if not preview_polygons:
            return [], []

        preview_labels = self._ensure_label_slots(copy.deepcopy(self.current_plant_labels), len(preview_polygons))
        final_polygons = []
        final_labels = []

        for index, polygon in enumerate(preview_polygons):
            normalized_polygon = copy.deepcopy(polygon)
            if self._get_polygon_area(normalized_polygon) > 0:
                normalized_polygon = normalized_polygon[::-1]
            final_polygons.append(normalized_polygon)
            final_labels.append(self._label_for_index(preview_labels, index))

        normalized_removals = normalize_polygons(copy.deepcopy(self.removal_regions))
        for removal_polygon in normalized_removals:
            removal_copy = copy.deepcopy(removal_polygon)
            if self._get_polygon_area(removal_copy) < 0:
                removal_copy = removal_copy[::-1]

            x_coords = [point[0] for point in removal_copy]
            y_coords = [point[1] for point in removal_copy]
            center_point = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

            for outer_contour in final_polygons:
                if self._get_polygon_area(outer_contour) > 0:
                    continue
                if not self._point_in_polygon(center_point, outer_contour):
                    continue
                intersection_poly = self._polygon_intersection(outer_contour, removal_copy)
                if not intersection_poly or len(intersection_poly) < 3:
                    continue
                if self._get_polygon_area(intersection_poly) < 0:
                    intersection_poly = intersection_poly[::-1]
                final_polygons.append(intersection_poly)
                break

        return normalize_polygons(final_polygons), final_labels

    def load_preview_from_formal_instance(self, plant):
        preview_polygons, preview_labels, removal_regions = self._split_instance_polygons_for_preview(
            plant.get("polygons", []),
            plant.get("labels", []),
        )
        self.current_plant_polygons = preview_polygons
        self.current_plant_labels = preview_labels
        self.removal_regions = removal_regions
        self.current_points = []
        self.current_removal_points = []
        self.current_snap_point = None
        self.selected_entity_kind = None
        self.selected_entity_id = None
        return bool(preview_polygons or removal_regions)

    def build_preview_formal_instance(self, instance_id=None, source="manual"):
        final_polygons, final_labels = self._compose_preview_instance_data()
        if not final_polygons:
            return None

        preview_id = instance_id if instance_id is not None else self.current_plant_id
        instance = make_formal_instance(instance_id=preview_id, polygons=final_polygons, source=source)
        instance["labels"] = final_labels
        instance["confirmed"] = False
        return instance

    def _find_plant_by_id(self, plant_id):
        for plant in self.plants:
            if int(plant.get("id", 0)) == int(plant_id):
                return plant
        return None

    def _resolve_staging_entity(self, entity_id=None):
        parsed = self._parse_staging_entity_id(entity_id or self.selected_entity_id)
        if not parsed:
            return None

        polygon_index = parsed["polygon_index"]
        if parsed["owner_kind"] == "preview":
            if polygon_index < 0 or polygon_index >= len(self.current_plant_polygons):
                return None
            labels = self._ensure_label_slots(self.current_plant_labels, len(self.current_plant_polygons))
            self.current_plant_labels = labels
            return {
                "id": self._make_staging_entity_id("preview", None, polygon_index),
                "owner_kind": "preview",
                "owner_id": None,
                "polygon_index": polygon_index,
                "polygon": self.current_plant_polygons[polygon_index],
                "polygons": [self.current_plant_polygons[polygon_index]],
                "label": self._label_for_index(labels, polygon_index),
            }

        plant = self._find_plant_by_id(parsed["owner_id"])
        if not plant:
            return None
        for area in self._iter_staging_areas_for_plant(plant):
            if int(area.get("polygon_index", -1)) == int(polygon_index):
                return area
        return None

    def enter_fine_tune_mode(self, instance_id):
        """进入微调模式"""
        # 保存微调前的原始数据
        self.fine_tune_original_data = {}
        for plant in self.plants:
            if int(plant.get("id", 0)) == int(instance_id):
                # 深拷贝植物数据
                import copy
                self.fine_tune_original_data[instance_id] = copy.deepcopy(plant)
                break
        
        # 清空微调模式的栈
        self.fine_tune_stack = []
        self.fine_tune_redo_stack = []
        
        self.mode = "fine_tune"
        self.fine_tune_instance_id = instance_id
        self.vertex_drag_info = None  # 拖拽中的顶点信息
        self.add_vertex_mode = False
        self.delete_vertex_mode = False
        self.update_display()

        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        if main_win and hasattr(main_win, "on_fine_tune_session_started"):
            main_win.on_fine_tune_session_started(instance_id)
    
    def enter_add_vertex_mode(self):
        """进入添加顶点模式"""
        if self.mode != "fine_tune":
            return
        self.add_vertex_mode = True
        self.set_merge_staging_mode(False)
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()
        
    def exit_add_vertex_mode(self):
        """退出添加顶点模式"""
        self.add_vertex_mode = False
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()

    def enter_delete_vertex_mode(self):
        """进入删除顶点模式"""
        if self.mode != "fine_tune":
            return
        self.delete_vertex_mode = True
        self.set_merge_staging_mode(False)
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()

    def exit_delete_vertex_mode(self):
        """退出删除顶点模式"""
        self.delete_vertex_mode = False
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        self.update_display()

    def exit_fine_tune_mode(self, save_changes=False):
        """退出微调模式"""
        if self.mode != "fine_tune":
            return True

        # 检查是否有修改
        has_changes = len(self.fine_tune_stack) > 0
        saved = bool(save_changes)
        
        if has_changes and not save_changes:
            # 弹出对话框询问是否保存
            from PyQt5.QtWidgets import QMessageBox
            main_win = self.get_main_window()
            parent = main_win if main_win else self
            reply = QMessageBox.question(
                parent,
                "退出微调模式",
                "您在微调模式下进行了修改，是否保存这些修改？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Cancel:
                return False  # 取消退出
            elif reply == QMessageBox.No:
                # 恢复原始数据
                if self.fine_tune_instance_id in self.fine_tune_original_data:
                    original_data = self.fine_tune_original_data[self.fine_tune_instance_id]
                    # 找到并替换当前数据
                    for i, plant in enumerate(self.plants):
                        if int(plant.get("id", 0)) == int(self.fine_tune_instance_id):
                            self.plants[i] = original_data
                            break
                saved = False
            else:
                saved = True

        # 退出微调模式
        exited_instance_id = self.fine_tune_instance_id
        self.mode = "normal"
        self.fine_tune_instance_id = None
        self.dragging_vertex = None
        self.fine_tune_original_data = {}
        self.add_vertex_mode = False
        self.delete_vertex_mode = False
        self.set_merge_staging_mode(False)
        self.current_points = []
        self.current_ignored_points = []
        self.ignoring_region = False
        self.current_removal_points = []
        self.removing_region = False
        self.current_snap_point = None
        self.selected_entity_kind = None
        self.selected_entity_id = None
        self.update_display()
        
        # 通知主窗口更新
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "sync_interaction_state"):
            main_win.sync_interaction_state()
        if main_win:
            main_win.mark_annotation_changed()
            main_win.sync_summary_view()
            main_win.update_plant_list()
            if hasattr(main_win, "on_fine_tune_session_finished"):
                main_win.on_fine_tune_session_finished(exited_instance_id, saved)
        self._notify_selection_changed()
        return True

    

    def set_image(self, pil_image, preprocessed_data=None):
        """设置图像，支持传入预处理数据。"""
        if pil_image is None:
            return
        try:
            self.current_points = []
            self.current_plant_polygons = []
            self.current_plant_labels = []
            self.current_ignored_points = []
            self.ignoring_region = False
            self.current_removal_points = []
            self.removal_regions = []
            self.removing_region = False
            self.selected_entity_kind = None
            self.selected_entity_id = None
            self.selected_plant_id = None
            self.candidate_instances = []
            self.vertex_drag_info = None
            self.scale_factor = 1.0
            self.current_snap_point = None
            self.is_dragging = False
            self.drag_last_pos = QPoint()
            self.set_merge_staging_mode(False)
            self.set_split_staging_mode(False)
            self.clear_preannotation_box()

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

    def set_annotation_state(self, plants, current_plant_id=None):
        """加载正式层状态。"""
        normalized_plants = []
        for index, plant in enumerate(plants or [], start=1):
            normalized_plants.append(normalize_formal_instance(plant, index))
        self.plants = normalized_plants
        self.current_plant_id = next_instance_id(self.plants, current_plant_id or 1)
        self.selected_entity_kind = None
        self.selected_entity_id = None
        self.selected_plant_id = None
        self.current_points = []
        self.current_plant_polygons = []
        self.current_plant_labels = []
        self.current_ignored_points = []
        self.ignoring_region = False
        self.current_removal_points = []
        self.removal_regions = []
        self.removing_region = False
        self.current_snap_point = None
        self.candidate_instances = []
        self.set_split_staging_mode(False)
        self.update_display()





    def get_annotation_state(self):
        """获取当前标注状态。"""
        return {
            "plants": copy.deepcopy(self.plants),
            "current_plant_id": self.current_plant_id,
        }

    def select_plant(self, plant_id):
        """兼容旧接口：按正式实例 id 选中。"""
        self.select_entity("formal", plant_id)

    def select_entity(self, kind, entity_id):
        """选择指定类型和ID的实体。"""
        self.selected_entity_kind = kind
        self.selected_entity_id = str(entity_id)
        self.update_display()
        self._notify_selection_changed()

    def get_selected_entity(self):
        """获取当前选中的实体。"""
        if not self.selected_entity_id or not self.selected_entity_kind:
            return None, None

        if self.selected_entity_kind == "formal":
            for plant in self.plants:
                if int(plant.get("id", 0)) == int(self.selected_entity_id):
                    return "formal", plant
        elif self.selected_entity_kind == "candidate":
            for candidate in self.candidate_instances:
                if candidate.get("candidate_id") == self.selected_entity_id:
                    return "candidate", candidate
        elif self.selected_entity_kind == "staging":
            staging = self._resolve_staging_entity(self.selected_entity_id)
            if staging:
                return "staging", staging
        elif self.selected_entity_kind == "removal":
            removal = self._resolve_removal_entity(self.selected_entity_id)
            if removal:
                return "removal", removal

        return None, None


    def _record_preview_state_change(self, old_polygons, old_labels, action_name, details=None):
        self.main_stack.append(
            {
                "action": "replace_preview_state",
                "action_name": action_name,
                "old_polygons": copy.deepcopy(old_polygons),
                "old_labels": copy.deepcopy(old_labels),
                "new_polygons": copy.deepcopy(self.current_plant_polygons),
                "new_labels": copy.deepcopy(self.current_plant_labels),
                "details": copy.deepcopy(details or {}),
            }
        )
        if len(self.main_stack) > self.max_stack_depth:
            self.main_stack.pop(0)
        self.redo_main_stack = []

    def _record_fine_tune_state_change(self, plant, old_polygons, old_labels, action_name, details=None):
        self.fine_tune_stack.append(
            {
                "action": "replace_entity_state",
                "action_name": action_name,
                "entity_id": plant.get("id"),
                "old_polygons": copy.deepcopy(old_polygons),
                "old_labels": copy.deepcopy(old_labels),
                "new_polygons": copy.deepcopy(plant.get("polygons", [])),
                "new_labels": copy.deepcopy(plant.get("labels", [])),
                "details": copy.deepcopy(details or {}),
            }
        )
        if len(self.fine_tune_stack) > self.max_stack_depth:
            self.fine_tune_stack.pop(0)
        self.fine_tune_redo_stack = []

    def _notify_preannotation_adjustment(self, instance_id, action_type, details):
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "record_preannotation_adjustment_action"):
            main_win.record_preannotation_adjustment_action(instance_id, action_type, copy.deepcopy(details))

    def _legacy_update_selected_staging_label_unused(self, new_label):
        selected_kind, staging = self.get_selected_entity()
        if selected_kind != "staging" or not staging:
            return False, "请先选择一个暂存区域"

        polygon_index = staging["polygon_index"]
        if staging["owner_kind"] == "preview":
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            self.current_plant_labels = self._ensure_label_slots(self.current_plant_labels, len(self.current_plant_polygons))
            self.current_plant_labels[polygon_index] = new_label
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "update_label",
                {"polygon_index": polygon_index, "label": new_label},
            )
        else:
            plant = staging["plant"]
            old_polygons = copy.deepcopy(plant.get("polygons", []))
            old_labels = copy.deepcopy(plant.get("labels", []))
            actual_polygon_index = staging.get("actual_polygon_index", polygon_index)
            if staging.get("label_storage") == "per_polygon_fallback":
                labels = self._ensure_label_slots(plant.get("labels", []), len(plant.get("polygons", [])))
                labels[actual_polygon_index] = new_label
            else:
                labels, _ = self._normalize_labels_for_polygons(plant.get("labels", []), plant.get("polygons", []))
                labels[polygon_index] = new_label
            plant["labels"] = labels
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "update_label",
                {"polygon_index": polygon_index, "label": new_label},
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "update_staging_label",
                {"polygon_index": polygon_index, "label": new_label},
            )

        self._notify_annotation_changed()
        self.update_display()
        return True, "暂存区域标签已更新"

    def delete_selected_staging_polygon(self):
        selected_kind, staging = self.get_selected_entity()
        if selected_kind != "staging" or not staging:
            return False, "请先选择一个暂存区域"

        polygon_index = staging["polygon_index"]
        if staging["owner_kind"] == "preview":
            if len(self.current_plant_polygons) <= 1:
                return False, "至少保留一个暂存区域后再删除"
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            del self.current_plant_polygons[polygon_index]
            labels = self._ensure_label_slots(self.current_plant_labels, len(old_polygons))
            del labels[polygon_index]
            self.current_plant_labels = labels
            self.selected_entity_kind = None
            self.selected_entity_id = None
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "delete_polygon",
                {"polygon_index": polygon_index},
            )
        else:
            plant = staging["plant"]
            polygons = plant.get("polygons", [])
            if len(polygons) <= 1:
                return False, "至少保留一个暂存区域后再删除"
            old_polygons = copy.deepcopy(polygons)
            old_labels = copy.deepcopy(plant.get("labels", []))
            actual_polygon_index = staging.get("actual_polygon_index", polygon_index)
            del polygons[actual_polygon_index]
            labels = list(plant.get("labels", []) or [])
            while len(labels) <= polygon_index:
                labels.append("stem")
            del labels[polygon_index]
            plant["polygons"] = normalize_polygons(polygons)
            plant["labels"] = labels
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self.selected_entity_kind = "formal"
            self.selected_entity_id = str(plant.get("id"))
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "delete_polygon",
                {"polygon_index": polygon_index},
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "delete_staging_polygon",
                {"polygon_index": polygon_index},
            )

        self.set_split_staging_mode(False)
        self._notify_annotation_changed()
        self.update_display()
        return True, "暂存区域已删除"

    def split_selected_staging_polygon(self, line_start, line_end, gap=5):
        selected_kind, staging = self.get_selected_entity()
        if selected_kind != "staging" or not staging:
            return False, "请先选择一个暂存区域"

        polygon = staging.get("polygon", [])
        if len(polygon) < 3:
            return False, "当前暂存区域无法切割"

        if math.dist((float(line_start[0]), float(line_start[1])), (float(line_end[0]), float(line_end[1]))) < 3:
            return False, "切割线太短"

        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        padding = max(8, int(gap) + 6)
        min_x = math.floor(min(x_coords)) - padding
        min_y = math.floor(min(y_coords)) - padding
        max_x = math.ceil(max(x_coords)) + padding
        max_y = math.ceil(max(y_coords)) + padding

        width = max(2, int(max_x - min_x + 1))
        height = max(2, int(max_y - min_y + 1))
        mask = np.zeros((height, width), dtype=np.uint8)

        local_polygon = np.array(
            [[int(round(point[0] - min_x)), int(round(point[1] - min_y))] for point in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [local_polygon], 255)

        local_start = (int(round(line_start[0] - min_x)), int(round(line_start[1] - min_y)))
        local_end = (int(round(line_end[0] - min_x)), int(round(line_end[1] - min_y)))
        cv2.line(mask, local_start, local_end, 0, thickness=max(5, int(gap)))

        split_polygons = self._extract_split_polygons_from_mask(mask, min_x, min_y, polygon)
        if len(split_polygons) < 2:
            return False, "切割后没有得到两个有效区域"

        polygon_index = staging["polygon_index"]
        label = staging.get("label", "stem")
        if staging["owner_kind"] == "preview":
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            self.current_plant_polygons = (
                self.current_plant_polygons[:polygon_index]
                + split_polygons
                + self.current_plant_polygons[polygon_index + 1:]
            )
            labels = self._ensure_label_slots(self.current_plant_labels, len(old_polygons))
            self.current_plant_labels = labels[:polygon_index] + [label, label] + labels[polygon_index + 1:]
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "split_polygon",
                {"polygon_index": polygon_index, "gap": gap},
            )
            self.select_entity("staging", self._make_staging_entity_id("preview", None, polygon_index))
        else:
            plant = staging["plant"]
            old_polygons = copy.deepcopy(plant.get("polygons", []))
            old_labels = copy.deepcopy(plant.get("labels", []))
            actual_polygon_index = staging.get("actual_polygon_index", polygon_index)
            polygons = plant.get("polygons", [])
            plant["polygons"] = (
                polygons[:actual_polygon_index]
                + split_polygons
                + polygons[actual_polygon_index + 1:]
            )
            labels = list(plant.get("labels", []) or [])
            while len(labels) <= polygon_index:
                labels.append("stem")
            plant["labels"] = labels[:polygon_index] + [label, label] + labels[polygon_index + 1:]
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "split_polygon",
                {"polygon_index": polygon_index, "gap": gap},
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "split_staging_polygon",
                {"polygon_index": polygon_index, "gap": gap},
            )
            self.select_entity("staging", self._make_staging_entity_id("formal", plant.get("id"), polygon_index))

        self.set_split_staging_mode(False)
        self._notify_annotation_changed()
        self.update_display()
        return True, "暂存区域已切割为两个区域"

    def delete_plant(self, plant_id):
        """兼容旧接口：删除正式实例。"""
        plant_id = int(plant_id)
        # 找到要删除的植株
        deleted_plant = None
        for plant in self.plants:
            if int(plant.get("id", 0)) == plant_id:
                deleted_plant = plant.copy()
                break
        self.plants = [plant for plant in self.plants if int(plant.get("id", 0)) != plant_id]
        # 将删除的植株信息存入栈，限制深度为2
        if deleted_plant:
            self.delete_plant_stack.append(deleted_plant)
            if len(self.delete_plant_stack) > 2:
                self.delete_plant_stack.pop(0)
        if self.selected_plant_id == plant_id:
            self.selected_plant_id = None
        if self.selected_entity_kind == "formal" and int(self.selected_entity_id or 0) == plant_id:
            self.selected_entity_kind = None
            self.selected_entity_id = None
        self.update_display()
        return True

    def undo_delete_plant(self):
        """撤销删除植株操作。"""
        if self.delete_plant_stack:
            # 弹出最后删除的植株
            deleted_plant = self.delete_plant_stack.pop()
            # 将植株添加回列表
            self.plants.append(deleted_plant)
            # 更新显示
            self.update_display()
            return True
        return False

    def save_current_polygon(self, label="stem"):
        """保存当前多边形到临时实例预览层。"""
        # 检查图片状态是否为已完成
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, 'current_image_state'):
            if main_win.current_image_state.get('annotation_completed', False):
                from PyQt5.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    main_win,
                    "图片已完成标注",
                    "此图片已标记为完成标注。是否取消已完成状态并继续标注？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
                else:
                    # 取消已完成状态
                    main_win.current_image_state['annotation_completed'] = False
                    main_win.mark_annotation_changed()
                    main_win.update_status_bar()
        
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

        # 记录操作到栈中，以便撤销时恢复到保存前的状态
        self.main_stack.append({
            'action': 'save_polygon',
            'current_points': self.current_points.copy(),
            'current_plant_polygons': self.current_plant_polygons.copy(),
            'current_plant_labels': self.current_plant_labels.copy()
        })
        
        self.current_plant_polygons.append(unique_points)
        self.current_plant_labels.append(label)  # 存储当前区域的 label
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

        # 记录操作前的状态
        self.ignore_stack.append({
            'action': 'save_region',
            'regions': self.ignored_regions.copy()
        })

        self.ignored_regions.append(unique_points)
        self.current_ignored_points = []
        self.update_display()
        return True

    def save_current_removal_region(self):
        """保存当前去除区域。"""
        # 检查图片状态是否为已完成
        main_win = self.get_main_window()
        if main_win and hasattr(main_win, 'current_image_state'):
            if main_win.current_image_state.get('annotation_completed', False):
                from PyQt5.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    main_win,
                    "图片已完成标注",
                    "此图片已标记为完成标注。是否取消已完成状态并继续标注？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
                else:
                    # 取消已完成状态
                    main_win.current_image_state['annotation_completed'] = False
                    main_win.mark_annotation_changed()
                    main_win.update_status_bar()
        
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

        if self.mode == "fine_tune" and self.fine_tune_instance_id:
            return self._apply_fine_tune_removal_region(unique_points)

        if not self.current_plant_polygons:
            return False

        # 记录操作到栈中
        self.main_stack.append({
            'action': 'save_removal_region',
            'regions': self.removal_regions.copy(),
            'current_removal_points': self.current_removal_points.copy()
        })

        self.removal_regions.append(unique_points)
        self.current_removal_points = []
        self.update_display()
        return True

    def _apply_fine_tune_removal_region(self, removal_polygon):
        """将去除区域作为内孔应用到微调中的实例。"""
        entity = None
        for plant in self.plants:
            if int(plant.get("id", 0)) == int(self.fine_tune_instance_id):
                entity = plant
                break
        if not entity:
            return False

        old_polygons = copy.deepcopy(entity.get("polygons", []))
        final_polygons = copy.deepcopy(entity.get("polygons", []))
        if not final_polygons:
            return False

        x_coords = [p[0] for p in removal_polygon]
        y_coords = [p[1] for p in removal_polygon]
        center_point = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

        added_hole = False
        for outer_contour in old_polygons:
            if len(outer_contour) < 3 or self._get_polygon_area(outer_contour) > 0:
                continue
            if not self._point_in_polygon(center_point, outer_contour):
                continue
            intersection_poly = self._polygon_intersection(outer_contour, removal_polygon)
            if not intersection_poly or len(intersection_poly) < 3:
                continue
            if self._get_polygon_area(intersection_poly) < 0:
                intersection_poly = intersection_poly[::-1]
            final_polygons.append(intersection_poly)
            added_hole = True
            break

        if not added_hole:
            return False

        entity["polygons"] = normalize_polygons(final_polygons)
        if entity.get("source") in ("ai_accepted", "ai_assisted"):
            entity["source"] = "ai_modified"
        touch_instance(entity)

        self.fine_tune_stack.append({
            'action': 'add_hole',
            'entity_id': entity.get("id"),
            'old_polygons': old_polygons,
            'new_polygons': copy.deepcopy(entity.get("polygons", [])),
            'removal_polygon': copy.deepcopy(removal_polygon),
        })
        if len(self.fine_tune_stack) > self.max_stack_depth:
            self.fine_tune_stack.pop(0)
        self.fine_tune_redo_stack = []

        self.current_removal_points = []
        if self.selected_entity_kind == "staging":
            self.selected_entity_kind = None
            self.selected_entity_id = None
        self.set_split_staging_mode(False)
        self.current_snap_point = None
        self.update_display()
        self._notify_annotation_changed()

        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "record_preannotation_adjustment_action"):
            main_win.record_preannotation_adjustment_action(
                entity.get("id"),
                "add_hole",
                {
                    "removal_polygon": copy.deepcopy(removal_polygon),
                    "result_polygons": copy.deepcopy(entity.get("polygons", [])),
                },
            )
        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
            main_win.on_entity_geometry_modified()
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
        final_labels = []  # 存储每个区域的 label
        
        # 添加外轮廓（所有暂存区域）
        for i, poly in enumerate(self.current_plant_polygons):
            if len(poly) >= 3:
                # 确保多边形是闭合的
                if poly[0] != poly[-1]:
                    poly = poly + [poly[0]]
                # 确保外轮廓为顺时针方向（面积为负）
                if self._get_polygon_area(poly) > 0:
                    poly = poly[::-1]  # 反转顶点顺序
                final_polygons.append(poly)
                # 保存对应的 label
                if i < len(self.current_plant_labels):
                    final_labels.append(self.current_plant_labels[i])
                else:
                    final_labels.append("stem")

        # 添加内孔（所有去除区域）
        for removal_poly in self.removal_regions:
            if len(removal_poly) >= 3:
                # 确保多边形是闭合的
                if removal_poly[0] != removal_poly[-1]:
                    removal_copy = removal_poly + [removal_poly[0]]
                else:
                    removal_copy = removal_poly.copy()
                
                # 计算去除区域的中心点
                x_coords = [p[0] for p in removal_copy]
                y_coords = [p[1] for p in removal_copy]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                center_point = (center_x, center_y)
                
                # 找到中心点所在的外多边形
                for i, outer_contour in enumerate(final_polygons):
                    # 检查中心点是否在外多边形内部
                    if self._point_in_polygon(center_point, outer_contour):
                        # 计算去除区域与外轮廓的交集
                        intersection_poly = self._polygon_intersection(outer_contour, removal_copy)
                        if intersection_poly and len(intersection_poly) >= 3:
                            # 确保内孔为逆时针方向（面积为正）
                            if self._get_polygon_area(intersection_poly) < 0:
                                intersection_poly = intersection_poly[::-1]  # 反转顶点顺序
                            final_polygons.append(intersection_poly)
                            # 内孔使用与外轮廓相同的 label
                            final_labels.append(final_labels[i])
                        break  # 一个去除区域只关联一个外多边形

        # 如果最终多边形列表为空，返回失败
        if not final_polygons:
            return False

        # 检查是否有原始植株ID（用于继续标注）
        if hasattr(self, '_original_plant_id') and self._original_plant_id:
            instance_id = self._original_plant_id
            # 清除临时保存的原始ID
            delattr(self, '_original_plant_id')
        else:
            instance_id = self.current_plant_id
            self.current_plant_id += 1

        # 创建实例
        new_instance = make_formal_instance(
            instance_id=instance_id,
            polygons=final_polygons,
            
            source="manual",
        )

        # 添加 label 信息
        new_instance["labels"] = final_labels

        # 保存实例
        self.plants.append(new_instance)
        saved_id = instance_id

        # 清空当前暂存的多边形、label、去除区域和正在绘制的点
        self.current_plant_polygons = []
        self.current_plant_labels = []
        self.removal_regions = []
        self.current_points = []
        self.current_removal_points = []

        # 保存整个植株时清空撤销栈
        if self.selected_entity_kind == "staging":
            self.selected_entity_kind = None
            self.selected_entity_id = None
        self.set_split_staging_mode(False)
        self.main_stack = []
        
        # 通知主窗口更新
        self._notify_annotation_changed()
        self.update_display()

        return saved_id

    def undo_last_action(self):
        """撤销手工绘制中的上一步。"""
        if self.is_summary:
            return False

        # 微调模式
        if self.mode == "fine_tune":
            if self.fine_tune_stack:
                last_action = self.fine_tune_stack.pop()
                if last_action.get('action') == 'replace_entity_state':
                    entity = self._find_plant_by_id(last_action['entity_id'])
                    if entity:
                        current_state = {
                            'action': 'replace_entity_state',
                            'action_name': last_action.get('action_name'),
                            'entity_id': entity.get('id'),
                            'old_polygons': copy.deepcopy(entity.get('polygons', [])),
                            'old_labels': copy.deepcopy(entity.get('labels', [])),
                            'new_polygons': copy.deepcopy(last_action.get('new_polygons', [])),
                            'new_labels': copy.deepcopy(last_action.get('new_labels', [])),
                            'details': copy.deepcopy(last_action.get('details', {})),
                        }
                        self.fine_tune_redo_stack.append(current_state)
                        if len(self.fine_tune_redo_stack) > self.max_stack_depth:
                            self.fine_tune_redo_stack.pop(0)

                        entity["polygons"] = normalize_polygons(copy.deepcopy(last_action.get('old_polygons', [])))
                        entity["labels"] = self._ensure_label_slots(
                            copy.deepcopy(last_action.get('old_labels', [])),
                            len(entity["polygons"]),
                        )
                        touch_instance(entity)
                        self.current_snap_point = None
                        self.update_display()
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                            main_win.on_entity_geometry_modified()
                        return True
                
                # 处理添加顶点操作的撤销
                if last_action['action'] == 'add_vertex':
                    # 保存当前状态到redo栈
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    # 找到对应的实体
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        current_polygon = entity.get("polygons", [])[polygon_index].copy() if polygon_index < len(entity.get("polygons", [])) else []
                        current_state = {
                            'action': 'add_vertex',
                            'entity_id': entity_id,
                            'polygon_index': polygon_index,
                            'new_polygon': current_polygon
                        }
                        self.fine_tune_redo_stack.append(current_state)
                        # 限制栈深度
                        if len(self.fine_tune_redo_stack) > self.max_stack_depth:
                            self.fine_tune_redo_stack.pop(0)
                        
                        # 恢复到添加顶点前的状态
                        polygons = entity.get("polygons", [])
                        if polygon_index < len(polygons):
                            polygons[polygon_index] = last_action['old_polygon']
                            entity["polygons"] = normalize_polygons(entity["polygons"])
                            touch_instance(entity)
                            self.update_display()
                            return True
                
                # 处理拖拽顶点操作的撤销
                elif last_action['action'] == 'drag_vertex':
                    # 保存当前状态到redo栈
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    point_index = last_action['point_index']
                    # 找到对应的实体
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        polygons = entity.get("polygons", [])
                        if polygon_index < len(polygons):
                            polygon = polygons[polygon_index]
                            if point_index < len(polygon):
                                current_position = polygon[point_index]
                                current_state = {
                                    'action': 'drag_vertex',
                                    'entity_id': entity_id,
                                    'polygon_index': polygon_index,
                                    'point_index': point_index,
                                    'old_position': current_position,
                                    'new_position': last_action['old_position']
                                }
                                self.fine_tune_redo_stack.append(current_state)
                                # 限制栈深度
                                if len(self.fine_tune_redo_stack) > self.max_stack_depth:
                                    self.fine_tune_redo_stack.pop(0)
                                
                                # 恢复到拖拽前的状态
                                polygon[point_index] = last_action['old_position']
                                # 确保多边形仍然闭合
                                if polygon[0] == polygon[-1]:
                                    if point_index == 0:
                                        polygon[-1] = polygon[0]
                                    elif point_index == len(polygon) - 1:
                                        polygon[0] = polygon[-1]
                                entity["polygons"] = normalize_polygons(entity["polygons"])
                                touch_instance(entity)
                                self.update_display()
                                return True
                elif last_action['action'] == 'delete_vertex':
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        polygons = entity.get("polygons", [])
                        current_polygon = polygons[polygon_index].copy() if polygon_index < len(polygons) else []
                        current_state = {
                            'action': 'delete_vertex',
                            'entity_id': entity_id,
                            'polygon_index': polygon_index,
                            'point_index': last_action['point_index'],
                            'old_polygon': current_polygon,
                            'new_polygon': copy.deepcopy(last_action['old_polygon']),
                        }
                        self.fine_tune_redo_stack.append(current_state)
                        if len(self.fine_tune_redo_stack) > self.max_stack_depth:
                            self.fine_tune_redo_stack.pop(0)

                        if polygon_index < len(polygons):
                            polygons[polygon_index] = copy.deepcopy(last_action['old_polygon'])
                            entity["polygons"] = normalize_polygons(entity["polygons"])
                            touch_instance(entity)
                            self.update_display()
                            main_win = self.get_main_window()
                            if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                                main_win.on_entity_geometry_modified()
                            return True
                elif last_action['action'] == 'add_hole':
                    entity_id = last_action['entity_id']
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        current_state = {
                            'action': 'add_hole',
                            'entity_id': entity_id,
                            'old_polygons': copy.deepcopy(entity.get("polygons", [])),
                            'new_polygons': copy.deepcopy(last_action['new_polygons']),
                            'removal_polygon': copy.deepcopy(last_action.get('removal_polygon', [])),
                        }
                        self.fine_tune_redo_stack.append(current_state)
                        if len(self.fine_tune_redo_stack) > self.max_stack_depth:
                            self.fine_tune_redo_stack.pop(0)

                        entity["polygons"] = normalize_polygons(copy.deepcopy(last_action['old_polygons']))
                        touch_instance(entity)
                        self.current_snap_point = None
                        self.update_display()
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                            main_win.on_entity_geometry_modified()
                        return True
                elif last_action['action'] == 'replace_entity_state':
                    entity = self._find_plant_by_id(last_action['entity_id'])
                    if entity:
                        current_state = {
                            'action': 'replace_entity_state',
                            'action_name': last_action.get('action_name'),
                            'entity_id': entity.get('id'),
                            'old_polygons': copy.deepcopy(entity.get('polygons', [])),
                            'old_labels': copy.deepcopy(entity.get('labels', [])),
                            'new_polygons': copy.deepcopy(last_action.get('new_polygons', [])),
                            'new_labels': copy.deepcopy(last_action.get('new_labels', [])),
                            'details': copy.deepcopy(last_action.get('details', {})),
                        }
                        self.fine_tune_redo_stack.append(current_state)
                        if len(self.fine_tune_redo_stack) > self.max_stack_depth:
                            self.fine_tune_redo_stack.pop(0)

                        entity["polygons"] = normalize_polygons(copy.deepcopy(last_action.get('old_polygons', [])))
                        entity["labels"] = self._ensure_label_slots(
                            copy.deepcopy(last_action.get('old_labels', [])),
                            len(entity["polygons"]),
                        )
                        touch_instance(entity)
                        self.current_snap_point = None
                        self.update_display()
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                            main_win.on_entity_geometry_modified()
                        return True
            
            # 弹出提示框提示无可undo的步骤
            from PyQt5.QtWidgets import QMessageBox
            main_win = self.get_main_window()
            parent = main_win if main_win else self
            QMessageBox.information(parent, "提示", "无可撤销的步骤")
            return False

        # 忽略区域模式
        if self.ignoring_region:
            if self.current_ignored_points:
                if self.ignore_stack:
                    last_action = self.ignore_stack.pop()
                    if last_action['action'] == 'add_point':
                        # 保存当前状态到redo栈
                        current_state = {
                            'action': 'add_point',
                            'points': self.current_ignored_points.copy()
                        }
                        self.redo_ignore_stack.append(current_state)
                        # 限制栈深度
                        if len(self.redo_ignore_stack) > self.max_stack_depth:
                            self.redo_ignore_stack.pop(0)
                        
                        self.current_ignored_points = last_action['points']
                else:
                    # 保存当前状态到redo栈
                    if self.current_ignored_points:
                        current_state = {
                            'action': 'add_point',
                            'points': self.current_ignored_points.copy()
                        }
                        self.redo_ignore_stack.append(current_state)
                        # 限制栈深度
                        if len(self.redo_ignore_stack) > self.max_stack_depth:
                            self.redo_ignore_stack.pop(0)
                        
                        self.current_ignored_points.pop()
                self.current_snap_point = None
                self.update_display()
                return True
            # 检查是否有已保存的忽略区域操作可以撤销
            elif self.ignore_stack:
                last_action = self.ignore_stack.pop()
                if last_action['action'] == 'save_region':
                    # 保存当前状态到redo栈
                    current_state = {
                        'action': 'save_region',
                        'regions': self.ignored_regions.copy()
                    }
                    self.redo_ignore_stack.append(current_state)
                    # 限制栈深度
                    if len(self.redo_ignore_stack) > self.max_stack_depth:
                        self.redo_ignore_stack.pop(0)
                    
                    self.ignored_regions = last_action['regions']
                    self.update_display()
                    return True
        else:
            # 标注区域和去除区域共用栈
            if self.main_stack:
                last_action = self.main_stack.pop()
                if last_action.get('action') == 'replace_preview_state':
                    current_state = {
                        'action': 'replace_preview_state',
                        'action_name': last_action.get('action_name'),
                        'old_polygons': copy.deepcopy(self.current_plant_polygons),
                        'old_labels': copy.deepcopy(self.current_plant_labels),
                        'new_polygons': copy.deepcopy(last_action.get('new_polygons', [])),
                        'new_labels': copy.deepcopy(last_action.get('new_labels', [])),
                        'details': copy.deepcopy(last_action.get('details', {})),
                    }
                    self.redo_main_stack.append(current_state)
                    if len(self.redo_main_stack) > self.max_stack_depth:
                        self.redo_main_stack.pop(0)

                    self.current_plant_polygons = copy.deepcopy(last_action.get('old_polygons', []))
                    self.current_plant_labels = self._ensure_label_slots(
                        copy.deepcopy(last_action.get('old_labels', [])),
                        len(self.current_plant_polygons),
                    )
                    self.current_snap_point = None
                    self.update_display()
                    return True
                
                # 处理暂存区域保存操作的撤销
                if last_action['action'] == 'save_polygon':
                    # 保存当前状态到redo栈
                    current_state = {
                        'action': 'save_polygon',
                        'current_points': self.current_points.copy(),
                        'current_plant_polygons': [poly.copy() for poly in self.current_plant_polygons],
                        'current_plant_labels': self.current_plant_labels.copy()
                    }
                    self.redo_main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.redo_main_stack) > self.max_stack_depth:
                        self.redo_main_stack.pop(0)
                    
                    # 恢复到保存前的状态，包括恢复current_points
                    self.current_points = last_action['current_points']
                    self.current_plant_polygons = last_action['current_plant_polygons']
                    self.current_plant_labels = last_action['current_plant_labels']
                    self.current_snap_point = None
                    self.update_display()
                    return True
                
                # 处理去除区域保存操作的撤销
                elif last_action['action'] == 'save_removal_region':
                    # 保存当前状态到redo栈
                    current_state = {
                        'action': 'save_removal_region',
                        'regions': [poly.copy() for poly in self.removal_regions],
                        'current_removal_points': self.current_removal_points.copy()
                    }
                    self.redo_main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.redo_main_stack) > self.max_stack_depth:
                        self.redo_main_stack.pop(0)
                    
                    self.removal_regions = last_action['regions']
                    self.current_removal_points = last_action['current_removal_points']
                    self.current_snap_point = None
                    self.update_display()
                    return True
                
                # 处理添加标注点操作的撤销
                elif last_action['action'] == 'add_point':
                    # 保存当前状态到redo栈
                    current_state = {
                        'action': 'add_point',
                        'points': self.current_points.copy()
                    }
                    self.redo_main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.redo_main_stack) > self.max_stack_depth:
                        self.redo_main_stack.pop(0)
                    
                    self.current_points = last_action['points']
                    self.current_snap_point = None
                    self.update_display()
                    return True
                
                # 处理添加去除区域点操作的撤销
                elif last_action['action'] == 'add_removal_point':
                    # 保存当前状态到redo栈
                    current_state = {
                        'action': 'add_removal_point',
                        'points': self.current_removal_points.copy()
                    }
                    self.redo_main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.redo_main_stack) > self.max_stack_depth:
                        self.redo_main_stack.pop(0)
                    
                    self.current_removal_points = last_action['points']
                    self.current_snap_point = None
                    self.update_display()
                    return True
                
                # 处理添加顶点操作的撤销
                elif last_action['action'] == 'add_vertex':
                    # 保存当前状态到redo栈
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    # 找到对应的实体
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        current_polygon = entity.get("polygons", [])[polygon_index].copy() if polygon_index < len(entity.get("polygons", [])) else []
                        current_state = {
                            'action': 'add_vertex',
                            'entity_id': entity_id,
                            'polygon_index': polygon_index,
                            'new_polygon': current_polygon
                        }
                        self.redo_main_stack.append(current_state)
                        # 限制栈深度
                        if len(self.redo_main_stack) > self.max_stack_depth:
                            self.redo_main_stack.pop(0)
                        
                        # 恢复到添加顶点前的状态
                        polygons = entity.get("polygons", [])
                        if polygon_index < len(polygons):
                            polygons[polygon_index] = last_action['old_polygon']
                            entity["polygons"] = normalize_polygons(entity["polygons"])
                            touch_instance(entity)
                            self.update_display()
                            return True
            
            # 现在所有点添加操作都通过栈记录，不需要直接操作当前点列表
            # 如果栈为空且当前有点，则直接清空（处理可能的边界情况）
            if self.removing_region and self.current_removal_points and not self.main_stack:
                # 保存当前状态到redo栈
                current_state = {
                    'action': 'add_removal_point',
                    'points': self.current_removal_points.copy()
                }
                self.redo_main_stack.append(current_state)
                # 限制栈深度
                if len(self.redo_main_stack) > self.max_stack_depth:
                    self.redo_main_stack.pop(0)
                
                self.current_removal_points = []
                self.current_snap_point = None
                self.update_display()
                return True
            elif not self.removing_region and self.current_points and not self.main_stack:
                # 保存当前状态到redo栈
                current_state = {
                    'action': 'add_point',
                    'points': self.current_points.copy()
                }
                self.redo_main_stack.append(current_state)
                # 限制栈深度
                if len(self.redo_main_stack) > self.max_stack_depth:
                    self.redo_main_stack.pop(0)
                
                self.current_points = []
                self.current_snap_point = None
                self.update_display()
                return True

        # 忽略区域的通用撤销
        if self.ignored_regions:
            # 保存当前状态到redo栈
            current_state = {
                'action': 'save_region',
                'regions': self.ignored_regions.copy()
            }
            self.redo_ignore_stack.append(current_state)
            # 限制栈深度
            if len(self.redo_ignore_stack) > self.max_stack_depth:
                self.redo_ignore_stack.pop(0)
            
            self.ignored_regions.pop()
            self.update_display()
            return True

        return False

    def redo_last_action(self):
        """重做上一步操作。"""
        if self.is_summary:
            return False

        # 微调模式
        if self.mode == "fine_tune":
            if self.fine_tune_redo_stack:
                last_action = self.fine_tune_redo_stack.pop()
                if last_action.get('action') == 'replace_entity_state':
                    entity = self._find_plant_by_id(last_action['entity_id'])
                    if entity:
                        current_state = {
                            'action': 'replace_entity_state',
                            'action_name': last_action.get('action_name'),
                            'entity_id': entity.get('id'),
                            'old_polygons': copy.deepcopy(entity.get('polygons', [])),
                            'old_labels': copy.deepcopy(entity.get('labels', [])),
                            'new_polygons': copy.deepcopy(last_action.get('new_polygons', [])),
                            'new_labels': copy.deepcopy(last_action.get('new_labels', [])),
                            'details': copy.deepcopy(last_action.get('details', {})),
                        }
                        self.fine_tune_stack.append(current_state)
                        if len(self.fine_tune_stack) > self.max_stack_depth:
                            self.fine_tune_stack.pop(0)

                        entity["polygons"] = normalize_polygons(copy.deepcopy(last_action.get('new_polygons', [])))
                        entity["labels"] = self._ensure_label_slots(
                            copy.deepcopy(last_action.get('new_labels', [])),
                            len(entity["polygons"]),
                        )
                        touch_instance(entity)
                        self.current_snap_point = None
                        self.update_display()
                        self._notify_annotation_changed()
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                            main_win.on_entity_geometry_modified()
                        return True
                
                # 处理添加顶点操作的重做
                if last_action['action'] == 'add_vertex':
                    # 保存当前状态到undo栈
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    # 找到对应的实体
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        current_polygon = entity.get("polygons", [])[polygon_index].copy() if polygon_index < len(entity.get("polygons", [])) else []
                        current_state = {
                            'action': 'add_vertex',
                            'entity_id': entity_id,
                            'polygon_index': polygon_index,
                            'old_polygon': current_polygon
                        }
                        self.fine_tune_stack.append(current_state)
                        # 限制栈深度
                        if len(self.fine_tune_stack) > self.max_stack_depth:
                            self.fine_tune_stack.pop(0)
                        
                        # 恢复到添加顶点后的状态
                        polygons = entity.get("polygons", [])
                        if polygon_index < len(polygons):
                            polygons[polygon_index] = last_action['new_polygon']
                            entity["polygons"] = normalize_polygons(entity["polygons"])
                            touch_instance(entity)
                            self.update_display()
                            return True
                
                # 处理拖拽顶点操作的重做
                elif last_action['action'] == 'drag_vertex':
                    # 保存当前状态到undo栈
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    point_index = last_action['point_index']
                    # 找到对应的实体
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        polygons = entity.get("polygons", [])
                        if polygon_index < len(polygons):
                            polygon = polygons[polygon_index]
                            if point_index < len(polygon):
                                current_position = polygon[point_index]
                                current_state = {
                                    'action': 'drag_vertex',
                                    'entity_id': entity_id,
                                    'polygon_index': polygon_index,
                                    'point_index': point_index,
                                    'old_position': current_position,
                                    'new_position': last_action['new_position']
                                }
                                self.fine_tune_stack.append(current_state)
                                # 限制栈深度
                                if len(self.fine_tune_stack) > self.max_stack_depth:
                                    self.fine_tune_stack.pop(0)
                                
                                # 恢复到拖拽后的状态
                                polygon[point_index] = last_action['new_position']
                                # 确保多边形仍然闭合
                                if polygon[0] == polygon[-1]:
                                    if point_index == 0:
                                        polygon[-1] = polygon[0]
                                    elif point_index == len(polygon) - 1:
                                        polygon[0] = polygon[-1]
                                entity["polygons"] = normalize_polygons(entity["polygons"])
                                touch_instance(entity)
                                self.update_display()
                                return True
                elif last_action['action'] == 'delete_vertex':
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        polygons = entity.get("polygons", [])
                        current_polygon = polygons[polygon_index].copy() if polygon_index < len(polygons) else []
                        current_state = {
                            'action': 'delete_vertex',
                            'entity_id': entity_id,
                            'polygon_index': polygon_index,
                            'point_index': last_action['point_index'],
                            'old_polygon': current_polygon,
                            'new_polygon': copy.deepcopy(last_action['new_polygon']),
                        }
                        self.fine_tune_stack.append(current_state)
                        if len(self.fine_tune_stack) > self.max_stack_depth:
                            self.fine_tune_stack.pop(0)

                        if polygon_index < len(polygons):
                            polygons[polygon_index] = copy.deepcopy(last_action['new_polygon'])
                            entity["polygons"] = normalize_polygons(entity["polygons"])
                            touch_instance(entity)
                            self.update_display()
                            main_win = self.get_main_window()
                            if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                                main_win.on_entity_geometry_modified()
                            return True
                elif last_action['action'] == 'add_hole':
                    entity_id = last_action['entity_id']
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        current_state = {
                            'action': 'add_hole',
                            'entity_id': entity_id,
                            'old_polygons': copy.deepcopy(entity.get("polygons", [])),
                            'new_polygons': copy.deepcopy(last_action['new_polygons']),
                            'removal_polygon': copy.deepcopy(last_action.get('removal_polygon', [])),
                        }
                        self.fine_tune_stack.append(current_state)
                        if len(self.fine_tune_stack) > self.max_stack_depth:
                            self.fine_tune_stack.pop(0)

                        entity["polygons"] = normalize_polygons(copy.deepcopy(last_action['new_polygons']))
                        touch_instance(entity)
                        self.current_snap_point = None
                        self.update_display()
                        self._notify_annotation_changed()
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
                            main_win.on_entity_geometry_modified()
                        return True
            
            # 弹出提示框提示无可redo的步骤
            from PyQt5.QtWidgets import QMessageBox
            main_win = self.get_main_window()
            parent = main_win if main_win else self
            QMessageBox.information(parent, "提示", "无可重做的步骤")
            return False

        # 忽略区域模式
        if self.ignoring_region:
            if self.redo_ignore_stack:
                last_action = self.redo_ignore_stack.pop()
                
                # 处理添加点操作的重做
                if last_action['action'] == 'add_point':
                    # 保存当前状态到undo栈
                    current_state = {
                        'action': 'add_point',
                        'points': self.current_ignored_points.copy()
                    }
                    self.ignore_stack.append(current_state)
                    # 限制栈深度
                    if len(self.ignore_stack) > self.max_stack_depth:
                        self.ignore_stack.pop(0)
                    
                    self.current_ignored_points = last_action['points']
                # 处理保存区域操作的重做
                elif last_action['action'] == 'save_region':
                    # 保存当前状态到undo栈
                    current_state = {
                        'action': 'save_region',
                        'regions': self.ignored_regions.copy()
                    }
                    self.ignore_stack.append(current_state)
                    # 限制栈深度
                    if len(self.ignore_stack) > self.max_stack_depth:
                        self.ignore_stack.pop(0)
                    
                    self.ignored_regions = last_action['regions']
                
                self.current_snap_point = None
                self.update_display()
                return True
            else:
                # 弹出提示框提示无可redo的步骤
                from PyQt5.QtWidgets import QMessageBox
                main_win = self.get_main_window()
                parent = main_win if main_win else self
                QMessageBox.information(parent, "提示", "无可重做的步骤")
                return False
        else:
            # 标注区域和去除区域共用栈
            if self.redo_main_stack:
                last_action = self.redo_main_stack.pop()
                if last_action.get('action') == 'replace_preview_state':
                    current_state = {
                        'action': 'replace_preview_state',
                        'action_name': last_action.get('action_name'),
                        'old_polygons': copy.deepcopy(self.current_plant_polygons),
                        'old_labels': copy.deepcopy(self.current_plant_labels),
                        'new_polygons': copy.deepcopy(last_action.get('new_polygons', [])),
                        'new_labels': copy.deepcopy(last_action.get('new_labels', [])),
                        'details': copy.deepcopy(last_action.get('details', {})),
                    }
                    self.main_stack.append(current_state)
                    if len(self.main_stack) > self.max_stack_depth:
                        self.main_stack.pop(0)

                    self.current_plant_polygons = copy.deepcopy(last_action.get('new_polygons', []))
                    self.current_plant_labels = self._ensure_label_slots(
                        copy.deepcopy(last_action.get('new_labels', [])),
                        len(self.current_plant_polygons),
                    )
                    self.current_snap_point = None
                    self.update_display()
                    return True
                
                # 处理暂存区域保存操作的重做
                if last_action['action'] == 'save_polygon':
                    # 保存当前状态到undo栈
                    current_state = {
                        'action': 'save_polygon',
                        'current_points': self.current_points.copy(),
                        'current_plant_polygons': [poly.copy() for poly in self.current_plant_polygons],
                        'current_plant_labels': self.current_plant_labels.copy()
                    }
                    self.main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.main_stack) > self.max_stack_depth:
                        self.main_stack.pop(0)
                    
                    self.current_points = last_action['current_points']
                    self.current_plant_polygons = last_action['current_plant_polygons']
                    self.current_plant_labels = last_action['current_plant_labels']
                # 处理去除区域保存操作的重做
                elif last_action['action'] == 'save_removal_region':
                    # 保存当前状态到undo栈
                    current_state = {
                        'action': 'save_removal_region',
                        'regions': [poly.copy() for poly in self.removal_regions],
                        'current_removal_points': self.current_removal_points.copy()
                    }
                    self.main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.main_stack) > self.max_stack_depth:
                        self.main_stack.pop(0)
                    
                    self.removal_regions = last_action['regions']
                    self.current_removal_points = last_action['current_removal_points']
                # 处理添加标注点操作的重做
                elif last_action['action'] == 'add_point':
                    # 保存当前状态到undo栈
                    current_state = {
                        'action': 'add_point',
                        'points': self.current_points.copy()
                    }
                    self.main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.main_stack) > self.max_stack_depth:
                        self.main_stack.pop(0)
                    
                    self.current_points = last_action['points']
                # 处理添加去除区域点操作的重做
                elif last_action['action'] == 'add_removal_point':
                    # 保存当前状态到undo栈
                    current_state = {
                        'action': 'add_removal_point',
                        'points': self.current_removal_points.copy()
                    }
                    self.main_stack.append(current_state)
                    # 限制栈深度
                    if len(self.main_stack) > self.max_stack_depth:
                        self.main_stack.pop(0)
                    
                    self.current_removal_points = last_action['points']
                
                # 处理添加顶点操作的重做
                elif last_action['action'] == 'add_vertex':
                    # 保存当前状态到undo栈
                    entity_id = last_action['entity_id']
                    polygon_index = last_action['polygon_index']
                    # 找到对应的实体
                    entity = None
                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            entity = plant
                            break
                    if entity:
                        current_polygon = entity.get("polygons", [])[polygon_index].copy() if polygon_index < len(entity.get("polygons", [])) else []
                        current_state = {
                            'action': 'add_vertex',
                            'entity_id': entity_id,
                            'polygon_index': polygon_index,
                            'old_polygon': current_polygon
                        }
                        self.main_stack.append(current_state)
                        # 限制栈深度
                        if len(self.main_stack) > self.max_stack_depth:
                            self.main_stack.pop(0)
                        
                        # 恢复到添加顶点后的状态
                        polygons = entity.get("polygons", [])
                        if polygon_index < len(polygons):
                            polygons[polygon_index] = last_action['new_polygon']
                            entity["polygons"] = normalize_polygons(entity["polygons"])
                            touch_instance(entity)
                
                self.current_snap_point = None
                self.update_display()
                return True
            else:
                # 弹出提示框提示无可redo的步骤
                from PyQt5.QtWidgets import QMessageBox
                main_win = self.get_main_window()
                parent = main_win if main_win else self
                QMessageBox.information(parent, "提示", "无可重做的步骤")
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

            image_pos = self.screen_to_image(event.pos())

            if self.preannotation_box_mode:
                if image_pos:
                    self.preannotation_box_dragging = True
                    self.preannotation_box_start = image_pos
                    self.preannotation_box_end = image_pos
                    self.preannotation_box_rect = None
                    self.update_display()
                return

            if self._is_fine_tune_interaction_active() and self.split_staging_mode:
                selected_kind, _ = self.get_selected_entity()
                if selected_kind == "staging" and image_pos:
                    self.split_line_dragging = True
                    self.split_line_start = image_pos
                    self.split_line_end = image_pos
                    self.update_display()
                return

            if self._is_fine_tune_interaction_active() and self.merge_staging_mode and image_pos:
                success, message = self._handle_merge_staging_click(image_pos)
                if message:
                    main_win = self.get_main_window()
                    if main_win and hasattr(main_win, "sam_info_text"):
                        prefix = "暂存区域合并完成" if success else "暂存区域合并提示"
                        main_win.sam_info_text.append(f"{prefix}: {message}")
                return

            # 检查图片状态是否为已完成
            main_win = self.get_main_window()
            if main_win and hasattr(main_win, 'current_image_state'):
                if main_win.current_image_state.get('annotation_completed', False):
                    from PyQt5.QtWidgets import QMessageBox
                    reply = QMessageBox.question(
                        main_win,
                        "图片已完成标注",
                        "此图片已标记为完成标注。是否取消已完成状态并继续标注？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                    else:
                        # 取消已完成状态
                        main_win.current_image_state['annotation_completed'] = False
                        main_win.mark_annotation_changed()
                        main_win.update_status_bar()

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

            if not image_pos:
                return

            # ??????????????????????????????????
            if self.candidate_instances and not self._is_fine_tune_interaction_active():
                hit_kind, hit_id = self._find_hit_entity(image_pos)
                if hit_kind == "candidate":
                    self.select_entity(hit_kind, hit_id)
                return

            # 优先处理微调模式
            if self._is_fine_tune_interaction_active():
                if self.removing_region:
                    self.main_stack.append({
                        'action': 'add_removal_point',
                        'points': self.current_removal_points.copy()
                    })
                    if self.current_removal_points:
                        self.current_removal_points.append(image_pos)
                    else:
                        self.current_removal_points = [image_pos]
                    self._notify_annotation_changed()
                    return
                if self.add_vertex_mode:
                    # 添加顶点模式：在当前微调实例最近的边上加点
                    entity = None
                    for plant in self.plants:
                        if int(plant.get("id", 0)) == int(self.fine_tune_instance_id):
                            entity = plant
                            break
                    if entity:
                        # 计算所有连线与点击位置的距离
                        min_distance = float('inf')
                        closest_edge = None

                        polygons = entity.get("polygons", [])

                        for polygon_index, polygon in enumerate(polygons):
                            point_count = len(polygon)
                            if point_count < 3:
                                continue

                            # 遍历每条边
                            limit = point_count if polygon[0] != polygon[-1] else point_count - 1
                            for i in range(limit):
                                p1 = polygon[i]
                                p2 = polygon[(i + 1) % point_count]

                                # 计算点到线段的距离
                                distance = self._point_to_line_distance(image_pos, p1, p2)

                                # 找到距离最近的线段
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_edge = {
                                        "entity": entity,
                                        "polygon_index": polygon_index,
                                        "edge_start": i,
                                        "edge_end": (i + 1) % point_count,
                                        "point": image_pos
                                    }

                        # 如果找到最近的连线
                        if closest_edge:
                            # 在连线上创建新顶点
                            self._add_vertex_on_edge(closest_edge)
                    return

                delete_vertex_hotkey = bool(event.modifiers() & Qt.AltModifier)
                if self.delete_vertex_mode or delete_vertex_hotkey:
                    vertex_hit = self._find_vertex_hit(image_pos, instance_id=self.fine_tune_instance_id)
                    if vertex_hit:
                        entity = None
                        for plant in self.plants:
                            if int(plant.get("id", 0)) == int(vertex_hit["entity_id"]):
                                entity = plant
                                break
                        if entity:
                            self._delete_vertex(entity, vertex_hit["polygon_index"], vertex_hit["point_index"])
                    return

                # 普通微调模式：
                # 1. 点到顶点时直接进入拖拽
                # 2. 点到多边形内部时选中该 polygon
                # 3. 点空白则不处理
                vertex_hit = self._find_vertex_hit(image_pos, instance_id=self.fine_tune_instance_id)
                if vertex_hit:
                    entity_id = vertex_hit["entity_id"]
                    polygon_index = vertex_hit["polygon_index"]
                    point_index = vertex_hit["point_index"]

                    for plant in self.plants:
                        if plant.get("id") == entity_id:
                            polygons = plant.get("polygons", [])
                            if polygon_index < len(polygons):
                                polygon = polygons[polygon_index]
                                if point_index < len(polygon):
                                    vertex_hit["original_pos"] = polygon[point_index]
                            break

                    self.vertex_drag_info = vertex_hit
                    return

                hit_kind, hit_id = self._find_hit_entity(image_pos)
                if hit_kind:
                    self.select_entity(hit_kind, hit_id)
                    return

                return

            # 非微调模式：执行其他操作
            if self.ignoring_region:
                if self.current_ignored_points:
                    # 记录操作前的状态
                    self.ignore_stack.append({
                        'action': 'add_point',
                        'points': self.current_ignored_points.copy()
                    })
                    self.current_ignored_points.append(image_pos)
                else:
                    self.current_ignored_points = [image_pos]
                self._notify_annotation_changed()
                return

            if self.removing_region:
                # 记录操作前的状态，包括添加第一个点的情况
                self.main_stack.append({
                    'action': 'add_removal_point',
                    'points': self.current_removal_points.copy()
                })
                if self.current_removal_points:
                    self.current_removal_points.append(image_pos)
                else:
                    self.current_removal_points = [image_pos]
                self._notify_annotation_changed()
                return

            # 只有在非微调模式下才处理current_points和current_plant_polygons
            if not self._is_fine_tune_interaction_active() and (self.current_points or self.current_plant_polygons):
                self._append_current_point(image_pos)
                self._notify_annotation_changed()
                return

            if self.region_growing_enabled:
                self.perform_region_growing(image_pos)
                self._notify_annotation_changed()
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
        """处理鼠标释放事件（修复版：确保闭合多边形一致性）。"""
        if event.button() == Qt.RightButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            return
        if event.button() == Qt.LeftButton and self.preannotation_box_dragging:
            self.preannotation_box_dragging = False
            rect = self._normalize_box(self.preannotation_box_start, self.preannotation_box_end)
            if rect:
                self.preannotation_box_rect = rect
                main_win = self.get_main_window()
                if main_win and hasattr(main_win, "on_preannotation_box_completed"):
                    main_win.on_preannotation_box_completed(rect)
            else:
                self.clear_preannotation_box()
            return
        if event.button() == Qt.LeftButton and self.split_line_dragging:
            if not self._is_fine_tune_interaction_active():
                self.set_split_staging_mode(False)
                return
            self.split_line_dragging = False
            image_pos = self.screen_to_image(event.pos())
            if image_pos:
                self.split_line_end = image_pos
            if self.split_line_start and self.split_line_end:
                success, message = self.split_selected_staging_polygon(self.split_line_start, self.split_line_end, gap=5)
                if not success:
                    main_win = self.get_main_window()
                    if main_win and hasattr(main_win, "sam_info_text"):
                        main_win.sam_info_text.append(f"切割暂存区域失败: {message}")
            else:
                self.set_split_staging_mode(False)
            return
        if event.button() == Qt.LeftButton and self.vertex_drag_info:
            if not self._is_fine_tune_interaction_active():
                self.vertex_drag_info = None
                return
            # 记录顶点拖拽操作到撤销栈
            if self.mode == "fine_tune":
                # 记录拖拽操作
                drag_info = self.vertex_drag_info
                entity_id = drag_info["entity_id"]
                polygon_index = drag_info["polygon_index"]
                point_index = drag_info["point_index"]

                # 找到对应的实体
                entity = None
                for plant in self.plants:
                    if plant.get("id") == entity_id:
                        entity = plant
                        break

                if entity:
                    polygons = entity.get("polygons", [])
                    if polygon_index < len(polygons):
                        polygon = polygons[polygon_index]

                        # ====================== 核心修复：计算有效顶点范围 ======================
                        is_closed = polygon[0] == polygon[-1]
                        valid_point_count = len(polygon) - 1 if is_closed else len(polygon)

                        # 确保拖拽的是有效顶点
                        if point_index >= valid_point_count:
                            self.vertex_drag_info = None
                            return

                        # 记录操作前的状态
                        old_position = drag_info.get('original_pos', polygon[point_index])
                        new_position = polygon[point_index]

                        # 只有当位置发生变化时才记录操作
                        if old_position != new_position:
                            self.fine_tune_stack.append({
                                'action': 'drag_vertex',
                                'entity_id': entity_id,
                                'polygon_index': polygon_index,
                                'point_index': point_index,
                                'old_position': old_position,
                                'new_position': new_position
                            })
                            # 限制栈深度
                            if len(self.fine_tune_stack) > self.max_stack_depth:
                                self.fine_tune_stack.pop(0)
                            # 清空重做栈
                            self.fine_tune_redo_stack = []

                            main_win = self.get_main_window()
                            if main_win and hasattr(main_win, "record_preannotation_adjustment_action"):
                                main_win.record_preannotation_adjustment_action(
                                    entity_id,
                                    "drag_vertex",
                                    {
                                        "polygon_index": polygon_index,
                                        "point_index": point_index,
                                        "old_position": old_position,
                                        "new_position": new_position,
                                    },
                                )

                        # ====================== 核心修复：规范化时确保闭合多边形一致性 ======================
                        entity_kind = "formal"
                        if entity_kind == "formal":
                            # 进行规范化处理
                            entity["polygons"] = normalize_polygons(entity["polygons"])

                            # 强制确保闭合多边形的最后一个点等于第一个点
                            updated_polygon = entity["polygons"][polygon_index]
                            if len(updated_polygon) > 1 and updated_polygon[0] == updated_polygon[-1]:
                                # 已经是闭合的，确保一致性
                                pass
                            elif len(updated_polygon) > 1:
                                # 如果规范化后丢失了闭合点，重新添加
                                updated_polygon.append(updated_polygon[0])

                            if entity.get("source") in ("ai_accepted", "ai_assisted"):
                                entity["source"] = "ai_modified"
                            touch_instance(entity)

                        # 更新显示
                        self.update_display()

                        # 通知主窗口更新撤销/重做状态
                        main_win = self.get_main_window()
                        if main_win and hasattr(main_win, 'update_undo_redo_state'):
                            main_win.update_undo_redo_state()

                # 清空拖拽信息
                self.vertex_drag_info = None

                # 通知标注已修改
                self._notify_annotation_changed()

                # 通知主窗口实体几何修改
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
            if not self._is_fine_tune_interaction_active():
                self.vertex_drag_info = None
                return
            image_pos = self.screen_to_image(event.pos())
            if image_pos:
                self._update_dragging_vertex(image_pos)
            return

        if self.preannotation_box_dragging:
            image_pos = self.screen_to_image(event.pos())
            if image_pos:
                self.preannotation_box_end = image_pos
                self.update_display()
            return

        if self.split_line_dragging:
            if not self._is_fine_tune_interaction_active():
                self.set_split_staging_mode(False)
                return
            image_pos = self.screen_to_image(event.pos())
            if image_pos:
                self.split_line_end = image_pos
                self.update_display()
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
            if (
                main_win
                and not self.is_summary
                and getattr(main_win, "projection_enabled", False)
            ):
                main_win.sync_summary_view()
        except Exception as error:
            print(f"wheelEvent error: {error}")
            traceback.print_exc()

    def _append_current_point(self, image_pos):
        """向当前手工 polygon 添加一个点。"""
        # 记录操作前的状态，包括添加第一个点的情况
        self.main_stack.append({
            'action': 'add_point',
            'points': self.current_points.copy()
        })
        if self.edge_snap_enabled and self.current_snap_point is not None:
            self.current_points.append(self.current_snap_point)
        else:
            self.current_points.append((float(image_pos[0]), float(image_pos[1])))
        self.current_snap_point = None
        self.update_display()

    def _iter_vertex_hit_candidates(self, instance_id=None, entity_candidates=None):
        if entity_candidates is not None:
            for candidate in entity_candidates:
                if candidate:
                    yield candidate.get("kind", "staging"), candidate
            return

        if instance_id:
            for plant in self.plants:
                if int(plant.get("id", 0)) == int(instance_id):
                    yield "formal", plant
                    return
            return

        entity_kind, entity = self.get_selected_entity()
        if entity:
            yield entity_kind, entity

    def _find_vertex_hit(self, image_pos, instance_id=None, entity_candidates=None):
        """查找当前选中对象是否有顶点命中（修复版：过滤闭合冗余点）。"""
        if entity_candidates is not None:
            hit_radius = self.vertex_hit_radius / max(self.scale_factor, 0.1)
            for entity_kind, entity in self._iter_vertex_hit_candidates(
                instance_id=instance_id,
                entity_candidates=entity_candidates,
            ):
                polygons = entity.get("polygons", [])
                for polygon_index, polygon in enumerate(polygons):
                    point_count = len(polygon)
                    if point_count < 3:
                        continue
                    is_closed = polygon[0] == polygon[-1]
                    valid_point_count = point_count - 1 if is_closed else point_count
                    for point_index in range(valid_point_count):
                        point = polygon[point_index]
                        distance = math.dist(
                            (float(point[0]), float(point[1])),
                            (float(image_pos[0]), float(image_pos[1])),
                        )
                        if distance <= hit_radius:
                            entity_id = entity.get("id")
                            if entity_kind == "candidate":
                                entity_id = entity.get("candidate_id")
                            return {
                                "kind": entity_kind,
                                "entity_id": entity_id,
                                "polygon_index": polygon_index,
                                "point_index": point_index,
                            }
            return None

        if instance_id:
            # 微调模式：查找指定实例
            for plant in self.plants:
                if int(plant.get("id", 0)) == int(instance_id):
                    entity_kind = "formal"
                    entity = plant
                    break
            else:
                return None
        else:
            # 其他模式：使用当前选中的实体
            entity_kind, entity = self.get_selected_entity()
            if not entity:
                return None
        hit_radius = self.vertex_hit_radius / max(self.scale_factor, 0.1)
        polygons = entity.get("polygons", [])

        for polygon_index, polygon in enumerate(polygons):
            point_count = len(polygon)
            if point_count < 3:
                continue

            # ====================== 核心修复：计算有效顶点范围 ======================
            is_closed = polygon[0] == polygon[-1]
            # 闭合多边形：只遍历到倒数第二个点（排除最后一个冗余点）
            # 非闭合多边形：遍历所有点
            valid_point_count = point_count - 1 if is_closed else point_count

            # 遍历所有有效顶点
            for point_index in range(valid_point_count):
                point = polygon[point_index]
                distance = math.dist((float(point[0]), float(point[1])), (float(image_pos[0]), float(image_pos[1])))
                if distance <= hit_radius:
                    return {
                        "kind": entity_kind,
                        "entity_id": entity.get("id") if entity_kind in ("formal", "staging") else entity.get(
                            "candidate_id"),
                        "polygon_index": polygon_index,
                        "point_index": point_index,
                    }
        return None
    def _find_edge_hit(self, image_pos, instance_id):
        """查找是否点击在多边形的连线上。"""
        # 查找指定实例
        for plant in self.plants:
            if int(plant.get("id", 0)) == int(instance_id):
                entity = plant
                break
        else:
            return None

        hit_radius = 5.0 / max(self.scale_factor, 0.1)  # 连线的命中半径
        polygons = entity.get("polygons", [])
        
        for polygon_index, polygon in enumerate(polygons):
            point_count = len(polygon)
            if point_count < 3:
                continue
            
            # 遍历每条边
            limit = point_count if polygon[0] != polygon[-1] else point_count - 1
            for i in range(limit):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % point_count]
                
                # 计算点到线段的距离
                distance = self._point_to_line_distance(image_pos, p1, p2)
                if distance <= hit_radius:
                    # 检查点是否在线段上
                    if self._point_on_segment(image_pos, p1, p2):
                        return {
                            "entity": entity,
                            "polygon_index": polygon_index,
                            "edge_start": i,
                            "edge_end": (i + 1) % point_count,
                            "point": image_pos
                        }
        return None

    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的距离。"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 计算向量
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        
        if len_sq != 0:  # 避免除零
            param = dot / len_sq
        
        xx, yy = 0, 0
        
        if param < 0:
            # 垂足在线段起点外
            xx = x1
            yy = y1
        elif param > 1:
            # 垂足在线段终点外
            xx = x2
            yy = y2
        else:
            # 垂足在线段上
            xx = x1 + param * C
            yy = y1 + param * D
        
        # 计算点到垂足的距离
        dx = x0 - xx
        dy = y0 - yy
        return math.sqrt(dx * dx + dy * dy)

    def _point_on_segment(self, point, segment_start, segment_end):
        """检查点是否在线段上。"""
        x, y = point
        x1, y1 = segment_start
        x2, y2 = segment_end
        
        # 检查点是否在由线段两端点定义的矩形内
        if not (min(x1, x2) - 1e-6 <= x <= max(x1, x2) + 1e-6 and 
                min(y1, y2) - 1e-6 <= y <= max(y1, y2) + 1e-6):
            return False
        
        # 检查点是否在直线上
        if abs((y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)) > 1e-6:
            return False
        
        return True

    def _add_vertex_on_edge(self, edge_hit):
        """在连线上添加新顶点。"""
        entity = edge_hit["entity"]
        polygon_index = edge_hit["polygon_index"]
        edge_start = edge_hit["edge_start"]
        point = edge_hit["point"]
        
        # 获取多边形
        polygons = entity.get("polygons", [])
        if polygon_index >= len(polygons):
            return
        
        polygon = polygons[polygon_index]
        
        # 记录操作前的状态，用于撤销
        if self.mode == "fine_tune":
            # 使用微调模式的栈
            self.fine_tune_stack.append({
                'action': 'add_vertex',
                'entity_id': entity.get("id"),
                'polygon_index': polygon_index,
                'edge_start': edge_start,
                'old_polygon': polygon.copy()
            })
            # 限制栈深度
            if len(self.fine_tune_stack) > self.max_stack_depth:
                self.fine_tune_stack.pop(0)
            # 清空重做栈
            self.fine_tune_redo_stack = []
        else:
            # 使用主栈
            self.main_stack.append({
                'action': 'add_vertex',
                'entity_id': entity.get("id"),
                'polygon_index': polygon_index,
                'edge_start': edge_start,
                'old_polygon': polygon.copy()
            })
        
        # 计算插入位置
        point_count = len(polygon)
        edge_end = (edge_start + 1) % point_count
        
        # 确保插入位置正确，避免在闭合点后插入
        insert_index = edge_start + 1
        if polygon[0] == polygon[-1] and insert_index >= len(polygon) - 1:
            insert_index = len(polygon) - 1
        
        # 在边上插入新顶点
        new_vertex = (float(point[0]), float(point[1]))
        polygon.insert(insert_index, new_vertex)
        
        # 确保多边形仍然闭合
        if polygon[0] == polygon[-1] and insert_index < len(polygon) - 1:
            polygon[-1] = polygon[0]
        
        # 更新实体的多边形数据
        entity["polygons"] = normalize_polygons(entity["polygons"])
        
        # 更新COCO数据
        touch_instance(entity)

        # 通知主窗口更新
        self._notify_annotation_changed()

        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "record_preannotation_adjustment_action") and self.mode == "fine_tune":
            main_win.record_preannotation_adjustment_action(
                entity.get("id"),
                "add_vertex",
                {
                    "polygon_index": polygon_index,
                    "insert_index": insert_index,
                    "new_vertex": new_vertex,
                },
            )
        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
            main_win.on_entity_geometry_modified()

        # 刷新画布
        self.update_display()

    def _delete_vertex(self, entity, polygon_index, point_index):
        """删除指定顶点，保留至少 3 个有效顶点。"""
        polygons = entity.get("polygons", [])
        if polygon_index >= len(polygons):
            return False

        polygon = polygons[polygon_index]
        is_closed = bool(polygon) and polygon[0] == polygon[-1]
        valid_point_count = len(polygon) - 1 if is_closed else len(polygon)
        if valid_point_count <= 3 or point_index >= valid_point_count:
            return False

        old_polygon = polygon.copy()
        if is_closed:
            base_polygon = polygon[:-1]
            del base_polygon[point_index]
            polygon = base_polygon + [base_polygon[0]]
        else:
            del polygon[point_index]

        if self.mode == "fine_tune":
            self.fine_tune_stack.append(
                {
                    "action": "delete_vertex",
                    "entity_id": entity.get("id"),
                    "polygon_index": polygon_index,
                    "point_index": point_index,
                    "old_polygon": old_polygon,
                    "new_polygon": polygon.copy(),
                }
            )
            if len(self.fine_tune_stack) > self.max_stack_depth:
                self.fine_tune_stack.pop(0)
            self.fine_tune_redo_stack = []

        polygons[polygon_index] = polygon
        entity["polygons"] = normalize_polygons(polygons)
        touch_instance(entity)
        self._notify_annotation_changed()

        main_win = self.get_main_window()
        if main_win and hasattr(main_win, "record_preannotation_adjustment_action") and self.mode == "fine_tune":
            main_win.record_preannotation_adjustment_action(
                entity.get("id"),
                "delete_vertex",
                {
                    "polygon_index": polygon_index,
                    "point_index": point_index,
                    "deleted_position": old_polygon[point_index],
                },
            )
        if main_win and hasattr(main_win, "on_entity_geometry_modified"):
            main_win.on_entity_geometry_modified()

        self.update_display()
        return True

    def _normalize_box(self, start_point, end_point):
        """规范化框选矩形。"""
        if not start_point or not end_point:
            return None
        x1 = min(float(start_point[0]), float(end_point[0]))
        y1 = min(float(start_point[1]), float(end_point[1]))
        x2 = max(float(start_point[0]), float(end_point[0]))
        y2 = max(float(start_point[1]), float(end_point[1]))
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            return None
        return (x1, y1, x2, y2)

    def _update_dragging_vertex(self, image_pos):
        """实时更新拖拽顶点（修复版：拖拽首点自动同步尾点）。"""
        drag = self.vertex_drag_info
        if not drag:
            return
        if not self._is_fine_tune_interaction_active():
            self.vertex_drag_info = None
            return
        # 获取目标实体
        if self.mode == "fine_tune" and self.fine_tune_instance_id:
            # 微调模式：使用指定的实例
            entity_kind = "formal"
            entity = None
            for plant in self.plants:
                if int(plant.get("id", 0)) == int(self.fine_tune_instance_id):
                    entity = plant
                    break
            if not entity:
                return
        else:
            # 其他模式：使用当前选中的实体
            entity_kind, entity = self.get_selected_entity()
            if not entity:
                return
        # 获取当前多边形
        polygon = entity["polygons"][drag["polygon_index"]]
        point_index = drag["point_index"]

        # ====================== 核心修复：计算有效顶点范围 ======================
        is_closed = polygon[0] == polygon[-1]
        valid_point_count = len(polygon) - 1 if is_closed else len(polygon)

        # 确保拖拽的是有效顶点（不是尾点）
        if point_index >= valid_point_count:
            return

        # 计算新位置
        new_position = (float(image_pos[0]), float(image_pos[1]))

        # 检查是否与相邻顶点重复
        if len(polygon) > 1:
            # 检查与前一个点是否重复
            prev_index = (point_index - 1) % valid_point_count
            if polygon[prev_index] == new_position:
                return

            # 检查与后一个点是否重复
            next_index = (point_index + 1) % valid_point_count
            if polygon[next_index] == new_position:
                return

        # 更新顶点位置
        polygon[point_index] = new_position

        # ====================== 核心修复：拖拽首点时自动同步尾点 ======================
        if is_closed and point_index == 0:
            # 如果是闭合多边形且拖拽的是第一个点，同步更新最后一个冗余点
            polygon[-1] = new_position

        # 更新显示
        self.update_display()

        # 同步摘要视图
        main_win = self.get_main_window()
        if main_win and entity_kind == "formal":
            main_win.sync_summary_view()
    def _iter_preview_staging_areas(self):
        labels = self._ensure_label_slots(self.current_plant_labels, len(self.current_plant_polygons))
        self.current_plant_labels = labels
        for index, polygon in enumerate(self.current_plant_polygons):
            yield {
                "id": self._make_staging_entity_id("preview", None, index),
                "owner_kind": "preview",
                "owner_id": None,
                "polygon_index": index,
                "polygon": polygon,
                "polygons": [polygon],
                "label": self._label_for_index(labels, index),
            }

    def _legacy_iter_formal_staging_areas_unused(self, plant):
        polygons = plant.get("polygons", [])
        labels = self._ensure_label_slots(plant.get("labels", []), len(polygons))
        plant["labels"] = labels
        for index, polygon in enumerate(polygons):
            yield {
                "id": self._make_staging_entity_id("formal", plant.get("id"), index),
                "owner_kind": "formal",
                "owner_id": plant.get("id"),
                "polygon_index": index,
                "polygon": polygon,
                "polygons": [polygon],
                "label": self._label_for_index(labels, index),
                "plant": plant,
            }

    @staticmethod
    def _get_rightmost_point(polygon):
        valid_points = list(polygon[:-1]) if polygon and polygon[0] == polygon[-1] else list(polygon or [])
        if not valid_points:
            return None
        return max(valid_points, key=lambda point: (point[0], -point[1]))

    def _find_hit_entity(self, image_pos):
        """?????????????????????"""
        for candidate in reversed(self.candidate_instances):
            if self._point_hits_polygons(image_pos, candidate.get("polygons", [])):
                return "candidate", candidate.get("candidate_id")

        for area in reversed(list(self._iter_preview_removal_areas())):
            if self._point_hits_polygons(image_pos, area.get("polygons", [])):
                return "removal", area.get("id")

        for area in reversed(list(self._iter_preview_staging_areas())):
            if self._point_hits_polygons(image_pos, area.get("polygons", [])):
                return "staging", area.get("id")

        for plant in reversed(self.plants):
            if self.mode == "fine_tune" and int(plant.get("id", 0)) == int(self.fine_tune_instance_id or 0):
                for area in reversed(list(self._iter_formal_removal_areas(plant))):
                    if self._point_hits_polygons(image_pos, area.get("polygons", [])):
                        return "removal", area.get("id")
                for area in reversed(list(self._iter_staging_areas_for_plant(plant))):
                    if self._point_hits_polygons(image_pos, area.get("polygons", [])):
                        return "staging", area.get("id")
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

    def _point_in_polygon(self, point, polygon):
        """判断点是否在多边形内部。"""
        try:
            import cv2
            import numpy as np
            
            # 确保多边形是闭合的
            if polygon[0] != polygon[-1]:
                polygon = polygon + [polygon[0]]
            
            # 转换为numpy数组
            polygon_np = np.array(polygon, dtype=np.int32)
            point_np = np.array(point, dtype=np.float32)
            
            # 使用cv2.pointPolygonTest判断点是否在多边形内部
            result = cv2.pointPolygonTest(polygon_np, point_np, False)
            
            # result > 0 表示点在多边形内部，result == 0 表示点在多边形边上
            return result >= 0
        except Exception as e:
            print(f"_point_in_polygon error: {e}")
            # 如果出错，使用射线法判断
            x, y = point
            inside = False
            n = len(polygon)
            for i in range(n):
                j = (i + 1) % n
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
            return inside

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
            
            # 转换回原始坐标系并控制点的密度
            intersection_poly = []
            for i, point in enumerate(largest_contour):
                # 每五个像素取一个点
                if i % 5 == 0:
                    x = point[0][0] + min_x
                    y = point[0][1] + min_y
                    intersection_poly.append((float(x), float(y)))
            
            # 确保至少有3个点
            if len(intersection_poly) < 3:
                # 如果点太少，使用所有点
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
            widget_width = self.width()
            widget_height = self.height()
            if widget_width <= 0 or widget_height <= 0:
                return

            img_width = self.raw_pixmap.width()
            img_height = self.raw_pixmap.height()
            offset_x = widget_width / 2 - self.view_center_x * self.scale_factor
            offset_y = widget_height / 2 - self.view_center_y * self.scale_factor

            view_half_width = widget_width / (2 * self.scale_factor)
            view_half_height = widget_height / (2 * self.scale_factor)
            src_left = max(0.0, self.view_center_x - view_half_width)
            src_top = max(0.0, self.view_center_y - view_half_height)
            src_right = min(float(img_width), self.view_center_x + view_half_width)
            src_bottom = min(float(img_height), self.view_center_y + view_half_height)

            src_width = max(0.0, src_right - src_left)
            src_height = max(0.0, src_bottom - src_top)

            final_pixmap = QPixmap(self.size())
            final_pixmap.fill(QColor(240, 240, 240))
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            if src_width > 0 and src_height > 0:
                target_rect = QRectF(
                    src_left * self.scale_factor + offset_x,
                    src_top * self.scale_factor + offset_y,
                    src_width * self.scale_factor,
                    src_height * self.scale_factor,
                )
                source_rect = QRectF(src_left, src_top, src_width, src_height)
                painter.drawPixmap(target_rect, self.raw_pixmap, source_rect)

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

            active_box = None
            if self.preannotation_box_dragging:
                active_box = self._normalize_box(self.preannotation_box_start, self.preannotation_box_end)
            elif self.preannotation_box_rect:
                active_box = self.preannotation_box_rect
            if active_box:
                x1, y1, x2, y2 = active_box
                qpt1 = img_to_screen((x1, y1))
                qpt2 = img_to_screen((x2, y2))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(0, 170, 255), 2, Qt.DashLine))
                painter.drawRect(qpt1.x(), qpt1.y(), qpt2.x() - qpt1.x(), qpt2.y() - qpt1.y())

            if self.split_staging_mode and self.split_line_start and self.split_line_end:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(255, 140, 0), 3, Qt.DashLine))
                painter.drawLine(img_to_screen(self.split_line_start), img_to_screen(self.split_line_end))

            if self.merge_staging_clicks:
                painter.setPen(QPen(QColor(0, 180, 140), 2))
                painter.setBrush(QBrush(QColor(0, 180, 140)))
                # 绘制所有点击点
                for click in self.merge_staging_clicks:
                    point = click.get("point")
                    if not point:
                        continue
                    qpoint = img_to_screen(point)
                    painter.drawEllipse(qpoint, 5, 5)
                # 绘制连接线
                if len(self.merge_staging_clicks) >= 2:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(QColor(0, 180, 140), 2, Qt.DashLine))
                    # 连接第一个区域的两个点
                    if len(self.merge_staging_clicks) >= 2:
                        painter.drawLine(
                            img_to_screen(self.merge_staging_clicks[0]["point"]),
                            img_to_screen(self.merge_staging_clicks[1]["point"]),
                        )
                    # 连接第二个区域的两个点
                    if len(self.merge_staging_clicks) >= 4:
                        painter.drawLine(
                            img_to_screen(self.merge_staging_clicks[2]["point"]),
                            img_to_screen(self.merge_staging_clicks[3]["point"]),
                        )
                    # 连接两个区域的对应点
                    if len(self.merge_staging_clicks) >= 3:
                        painter.drawLine(
                            img_to_screen(self.merge_staging_clicks[0]["point"]),
                            img_to_screen(self.merge_staging_clicks[2]["point"]),
                        )
                    if len(self.merge_staging_clicks) >= 4:
                        painter.drawLine(
                            img_to_screen(self.merge_staging_clicks[1]["point"]),
                            img_to_screen(self.merge_staging_clicks[3]["point"]),
                        )

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
            is_fine_tune = self.mode == "fine_tune" and int(plant.get("id", 0)) == int(self.fine_tune_instance_id or 0)

            polygons = plant.get("polygons", [])
            if polygons:
                # 创建临时QPixmap用于绘制填充和去除区域
                temp_pixmap = QPixmap(self.size())
                temp_pixmap.fill(Qt.transparent)
                temp_painter = QPainter(temp_pixmap)
                temp_painter.setRenderHint(QPainter.Antialiasing)
                
                # 绘制所有外轮廓（面积为负的多边形）
                for polygon in polygons:
                    if len(polygon) >= 3 and self._get_polygon_area(polygon) <= 0:
                        qpts = [img_to_screen(point) for point in polygon]
                        if summary_mode:
                            temp_painter.setBrush(QBrush(plant_color))
                            temp_painter.setPen(QPen(QColor(255, 0, 0), 3) if is_selected else QPen(QColor(0, 0, 0), 1))
                        else:
                            weak_alpha = 70 if is_fine_tune else 55
                            weak_color = QColor(plant_color.red(), plant_color.green(), plant_color.blue(), weak_alpha)
                            line_color = QColor(255, 140, 0) if (is_selected or is_fine_tune) else QColor(100, 100, 100)
                            temp_painter.setBrush(QBrush(weak_color))
                            temp_painter.setPen(QPen(line_color, 3 if (is_selected or is_fine_tune) else 1))
                        temp_painter.drawPolygon(*qpts)
                
                # 绘制所有内孔（面积为正的多边形）
                for polygon in polygons:
                    if len(polygon) >= 3 and self._get_polygon_area(polygon) > 0:
                        qpts = [img_to_screen(point) for point in polygon]
                        # 使用组合模式实现挖空
                        temp_painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
                        temp_painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
                        temp_painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
                        temp_painter.drawPolygon(*qpts)
                
                temp_painter.end()
                # 将临时pixmap绘制到主画布
                painter.drawPixmap(0, 0, temp_pixmap)

            if not summary_mode and is_fine_tune:
                self._draw_polygon_vertices(painter, img_to_screen, plant.get("polygons", []), QColor(255, 0, 0))
            
            # 绘制暂存区域
            if not summary_mode:
                staging_areas = list(self._iter_staging_areas_for_plant(plant))
                for area in staging_areas:
                    area_id = area.get("id")
                    is_staging_selected = self.selected_entity_kind == "staging" and self.selected_entity_id == str(area_id)
                    show_staging_highlight = is_staging_selected and is_fine_tune
                    polygon = area.get("polygon", [])
                    if len(polygon) >= 3:
                        qpts = [img_to_screen(point) for point in polygon]
                        fill_color = QColor(0, 180, 255, 35 if not show_staging_highlight else 70)
                        line_color = QColor(0, 150, 255) if not show_staging_highlight else QColor(255, 165, 0)
                        painter.setBrush(QBrush(fill_color))
                        painter.setPen(QPen(line_color, 2, Qt.DashLine))
                        painter.drawPolygon(*qpts)
                        if show_staging_highlight:
                            self._draw_polygon_vertices(painter, img_to_screen, [polygon], QColor(0, 150, 255))
                for area in self._iter_formal_removal_areas(plant):
                    area_id = area.get("id")
                    is_removal_selected = self.selected_entity_kind == "removal" and self.selected_entity_id == str(area_id)
                    show_removal_highlight = is_removal_selected and is_fine_tune
                    polygon = area.get("polygon", [])
                    if len(polygon) >= 3:
                        qpts = [img_to_screen(point) for point in polygon]
                        if show_removal_highlight:
                            painter.setBrush(QBrush(QColor(255, 80, 80, 65)))
                            painter.setPen(QPen(QColor(255, 60, 60), 2, Qt.DashLine))
                            painter.drawPolygon(*qpts)
                            self._draw_polygon_vertices(painter, img_to_screen, [polygon], QColor(255, 60, 60))
            
            # 在实例的最左端的点用红色文字标出 plantid
            # 只在摘要模式（右侧画布）下显示
            if summary_mode:
                plant_id = plant.get("id", 0)
                # 找到所有多边形的所有点
                all_points = []
                for polygon in polygons:
                    all_points.extend(polygon)
                if all_points:
                    # 找到最左端的点（x坐标最小的点）
                    leftmost_point = min(all_points, key=lambda p: p[0])
                    # 转换为屏幕坐标
                    label_point = img_to_screen(leftmost_point)
                    # 绘制文字
                    painter.setPen(QPen(QColor(255, 0, 0), 1))
                    painter.drawText(label_point, f"plant {plant_id}")
                for area in self._iter_staging_areas_for_plant(plant):
                    rightmost_point = self._get_rightmost_point(area.get("polygon", []))
                    if not rightmost_point:
                        continue
                    label_point = img_to_screen(rightmost_point)
                    painter.setPen(QPen(QColor(20, 20, 20), 1))
                    painter.drawText(label_point, area.get("label", "stem"))

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
                for i, polygon in enumerate(self.current_plant_polygons):
                    if len(polygon) >= 3:
                        qpts = [img_to_screen(point) for point in polygon]
                        is_preview_staging_selected = (
                            self.selected_entity_kind == "staging"
                            and self.selected_entity_id == self._make_staging_entity_id("preview", None, i)
                        )
                        show_preview_staging_highlight = is_preview_staging_selected and self._is_fine_tune_interaction_active()
                        if show_preview_staging_highlight:
                            temp_painter.setBrush(QBrush(QColor(255, 196, 0, 140)))
                            temp_painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))
                        else:
                            temp_painter.setBrush(QBrush(temp_color))
                            temp_painter.setPen(QPen(QColor(0, 150, 0), 2))
                        temp_painter.drawPolygon(*qpts)
                        
                        # 绘制区域 label
                        if i < len(self.current_plant_labels):
                            label = self.current_plant_labels[i]
                            # 找到多边形最靠左边的点
                            # 找到 x 坐标最小的点
                            leftmost_point = min(polygon, key=lambda p: p[0])
                            # 转换为屏幕坐标
                            label_point = img_to_screen(leftmost_point)
                            # 绘制文字
                            temp_painter.setPen(QPen(QColor(0, 0, 0), 1))
                            temp_painter.drawText(label_point, label)
                        if show_preview_staging_highlight:
                            self._draw_polygon_vertices(temp_painter, img_to_screen, [polygon], QColor(255, 140, 0))
            
            # 绘制去除区域（挖空效果）
            if self.removal_regions:
                for index, region in enumerate(self.removal_regions):
                    if len(region) >= 3:
                        qpts = [img_to_screen(point) for point in region]
                        # 使用组合模式实现挖空
                        temp_painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
                        temp_painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
                        temp_painter.setPen(QPen(QColor(255, 255, 255, 255), 2))
                        temp_painter.drawPolygon(*qpts)
                        if (
                            self.selected_entity_kind == "removal"
                            and self.selected_entity_id == self._make_removal_entity_id("preview", None, index)
                        ):
                            if self._is_fine_tune_interaction_active():
                                temp_painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                                temp_painter.setBrush(QBrush(QColor(255, 80, 80, 65)))
                                temp_painter.setPen(QPen(QColor(255, 60, 60), 2, Qt.DashLine))
                                temp_painter.drawPolygon(*qpts)
                                self._draw_polygon_vertices(temp_painter, img_to_screen, [region], QColor(255, 60, 60))
            
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
            if hasattr(main_win, "_update_staging_controls"):
                main_win._update_staging_controls()

    def _iter_active_fine_tune_staging_areas(self):
        if self.mode != "fine_tune" or not self.fine_tune_instance_id:
            return

        plant = self._find_plant_by_id(self.fine_tune_instance_id)
        if not plant:
            return

        polygons = plant.get("polygons", [])
        outer_indices = self._get_outer_polygon_indices(polygons)
        if not outer_indices and len(polygons) == 1:
            outer_indices = [0]

        labels = list(plant.get("labels", []) or [])[: len(outer_indices)]
        while len(labels) < len(outer_indices):
            labels.append("stem")
        plant["labels"] = labels

        for label_index, actual_polygon_index in enumerate(outer_indices):
            polygon = polygons[actual_polygon_index]
            yield {
                "id": self._make_staging_entity_id("formal", plant.get("id"), label_index),
                "owner_kind": "formal",
                "owner_id": plant.get("id"),
                "polygon_index": label_index,
                "actual_polygon_index": actual_polygon_index,
                "polygon": polygon,
                "polygons": [polygon],
                "label": self._label_for_index(labels, label_index),
                "label_storage": "fine_tune_outer",
                "plant": plant,
            }

    def _iter_staging_areas_for_plant(self, plant):
        if not plant:
            return

        if (
            self.mode == "fine_tune"
            and int(plant.get("id", 0)) == int(self.fine_tune_instance_id or 0)
        ):
            yield from self._iter_active_fine_tune_staging_areas()
            return

        yield from self._iter_formal_staging_areas(plant)

    def _iter_formal_staging_areas(self, plant):
        polygons = plant.get("polygons", [])
        raw_labels = list(plant.get("labels", []) or [])
        outer_indices = self._get_outer_polygon_indices(polygons)

        use_outer_mapping = bool(outer_indices) and len(outer_indices) >= len(raw_labels)
        if use_outer_mapping:
            labels = raw_labels[:len(outer_indices)]
            while len(labels) < len(outer_indices):
                labels.append("stem")
            mapping = [(label_index, polygon_index, "outer_only") for label_index, polygon_index in enumerate(outer_indices)]
        elif raw_labels:
            fallback_count = min(len(raw_labels), len(polygons))
            labels = self._ensure_label_slots(raw_labels[:fallback_count], fallback_count)
            mapping = [(index, index, "per_polygon") for index in range(fallback_count)]
        elif len(polygons) == 1:
            labels = ["stem"]
            mapping = [(0, 0, "per_polygon")]
        else:
            labels = []
            mapping = []

        plant["labels"] = labels
        for label_index, actual_polygon_index, label_storage in mapping:
            polygon = polygons[actual_polygon_index]
            yield {
                "id": self._make_staging_entity_id("formal", plant.get("id"), label_index),
                "owner_kind": "formal",
                "owner_id": plant.get("id"),
                "polygon_index": label_index,
                "actual_polygon_index": actual_polygon_index,
                "polygon": polygon,
                "polygons": [polygon],
                "label": self._label_for_index(labels, label_index),
                "label_storage": label_storage,
                "plant": plant,
            }

    def _iter_preview_removal_areas(self):
        for index, polygon in enumerate(self.removal_regions):
            yield {
                "id": self._make_removal_entity_id("preview", None, index),
                "owner_kind": "preview",
                "owner_id": None,
                "polygon_index": index,
                "polygon": polygon,
                "polygons": [polygon],
            }

    def _iter_formal_removal_areas(self, plant):
        polygons = plant.get("polygons", [])
        for hole_index, polygon_index in enumerate(self._get_inner_polygon_indices(polygons)):
            polygon = polygons[polygon_index]
            yield {
                "id": self._make_removal_entity_id("formal", plant.get("id"), hole_index),
                "owner_kind": "formal",
                "owner_id": plant.get("id"),
                "polygon_index": hole_index,
                "actual_polygon_index": polygon_index,
                "polygon": polygon,
                "polygons": [polygon],
                "plant": plant,
            }

    def _resolve_removal_entity(self, entity_id=None):
        parsed = self._parse_removal_entity_id(entity_id or self.selected_entity_id)
        if not parsed:
            return None

        polygon_index = parsed["polygon_index"]
        if parsed["owner_kind"] == "preview":
            if polygon_index < 0 or polygon_index >= len(self.removal_regions):
                return None
            polygon = self.removal_regions[polygon_index]
            return {
                "id": self._make_removal_entity_id("preview", None, polygon_index),
                "owner_kind": "preview",
                "owner_id": None,
                "polygon_index": polygon_index,
                "polygon": polygon,
                "polygons": [polygon],
            }

        plant = self._find_plant_by_id(parsed["owner_id"])
        if not plant:
            return None
        inner_indices = self._get_inner_polygon_indices(plant.get("polygons", []))
        if polygon_index < 0 or polygon_index >= len(inner_indices):
            return None
        actual_polygon_index = inner_indices[polygon_index]
        polygon = plant.get("polygons", [])[actual_polygon_index]
        return {
            "id": self._make_removal_entity_id("formal", plant.get("id"), polygon_index),
            "owner_kind": "formal",
            "owner_id": plant.get("id"),
            "polygon_index": polygon_index,
            "actual_polygon_index": actual_polygon_index,
            "polygon": polygon,
            "polygons": [polygon],
            "plant": plant,
        }

    def update_selected_staging_label(self, new_label):
        selected_kind, staging = self.get_selected_entity()
        if selected_kind != "staging" or not staging:
            return False, "请先选择一个暂存区域"

        polygon_index = staging["polygon_index"]
        if staging["owner_kind"] == "preview":
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            self.current_plant_labels = self._ensure_label_slots(self.current_plant_labels, len(self.current_plant_polygons))
            self.current_plant_labels[polygon_index] = new_label
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "update_label",
                {"polygon_index": polygon_index, "label": new_label},
            )
        else:
            plant = staging["plant"]
            old_polygons = copy.deepcopy(plant.get("polygons", []))
            old_labels = copy.deepcopy(plant.get("labels", []))
            actual_polygon_index = staging.get("actual_polygon_index", polygon_index)
            if (
                self.mode == "fine_tune"
                and int(plant.get("id", 0)) == int(self.fine_tune_instance_id or 0)
            ) or staging.get("label_storage") == "fine_tune_outer":
                labels = list(plant.get("labels", []) or [])
                while len(labels) <= polygon_index:
                    labels.append("stem")
                labels[polygon_index] = new_label
            elif staging.get("label_storage") == "per_polygon":
                labels = self._ensure_label_slots(
                    list(plant.get("labels", []) or []),
                    max(len(plant.get("polygons", [])), actual_polygon_index + 1),
                )
                labels[actual_polygon_index] = new_label
            else:
                labels = list(plant.get("labels", []) or [])
                while len(labels) <= polygon_index:
                    labels.append("stem")
                labels[polygon_index] = new_label
            plant["labels"] = labels
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "update_label",
                {"polygon_index": polygon_index, "label": new_label},
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "update_staging_label",
                {"polygon_index": polygon_index, "label": new_label},
            )

        self._notify_annotation_changed()
        self.update_display()
        return True, "暂存区域标签已更新"

    def delete_selected_staging_polygon(self):
        selected_kind, staging = self.get_selected_entity()
        if selected_kind == "removal" and staging:
            if staging["owner_kind"] == "preview":
                old_regions = copy.deepcopy(self.removal_regions)
                polygon_index = staging["polygon_index"]
                if polygon_index < 0 or polygon_index >= len(self.removal_regions):
                    return False, "请先选择一个去除区域"
                del self.removal_regions[polygon_index]
                self.selected_entity_kind = None
                self.selected_entity_id = None
                self.main_stack.append(
                    {
                        "action": "save_removal_region",
                        "regions": old_regions,
                        "current_removal_points": self.current_removal_points.copy(),
                    }
                )
                if len(self.main_stack) > self.max_stack_depth:
                    self.main_stack.pop(0)
                self.redo_main_stack = []
            else:
                plant = staging["plant"]
                polygons = copy.deepcopy(plant.get("polygons", []))
                actual_polygon_index = staging.get("actual_polygon_index")
                if actual_polygon_index is None or actual_polygon_index < 0 or actual_polygon_index >= len(polygons):
                    return False, "请先选择一个去除区域"
                old_polygons = copy.deepcopy(polygons)
                old_labels = copy.deepcopy(plant.get("labels", []))
                del polygons[actual_polygon_index]
                plant["polygons"] = normalize_polygons(polygons)
                labels, _ = self._normalize_labels_for_polygons(plant.get("labels", []), plant["polygons"])
                plant["labels"] = labels
                touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
                self.selected_entity_kind = "formal"
                self.selected_entity_id = str(plant.get("id"))
                self._record_fine_tune_state_change(
                    plant,
                    old_polygons,
                    old_labels,
                    "delete_removal_region",
                    {"polygon_index": staging["polygon_index"], "actual_polygon_index": actual_polygon_index},
                )
                self._notify_preannotation_adjustment(
                    plant.get("id"),
                    "delete_removal_region",
                    {"polygon_index": staging["polygon_index"], "actual_polygon_index": actual_polygon_index},
                )

            self.set_merge_staging_mode(False)
            self.set_split_staging_mode(False)
            self._notify_annotation_changed()
            self.update_display()
            return True, "去除区域已删除"

        if selected_kind != "staging" or not staging:
            return False, "请先选择一个暂存区域或去除区域"

        polygon_index = staging["polygon_index"]
        if staging["owner_kind"] == "preview":
            if len(self.current_plant_polygons) <= 1:
                return False, "至少保留一个暂存区域后再删除"
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            del self.current_plant_polygons[polygon_index]
            labels = self._ensure_label_slots(self.current_plant_labels, len(old_polygons))
            del labels[polygon_index]
            self.current_plant_labels = labels
            self.selected_entity_kind = None
            self.selected_entity_id = None
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "delete_polygon",
                {"polygon_index": polygon_index},
            )
        else:
            plant = staging["plant"]
            polygons = plant.get("polygons", [])
            staging_areas = list(self._iter_staging_areas_for_plant(plant))
            if len(staging_areas) <= 1:
                return False, "至少保留一个暂存区域后再删除"
            old_polygons = copy.deepcopy(polygons)
            old_labels = copy.deepcopy(plant.get("labels", []))
            actual_polygon_index = staging.get("actual_polygon_index", polygon_index)
            del polygons[actual_polygon_index]
            if (
                self.mode == "fine_tune"
                and int(plant.get("id", 0)) == int(self.fine_tune_instance_id or 0)
            ) or staging.get("label_storage") == "fine_tune_outer":
                labels = list(plant.get("labels", []) or [])
                while len(labels) <= polygon_index:
                    labels.append("stem")
                del labels[polygon_index]
            else:
                labels, _ = self._normalize_labels_for_polygons(plant.get("labels", []), old_polygons)
                del labels[polygon_index]
            plant["polygons"] = normalize_polygons(polygons)
            plant["labels"] = labels
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self.selected_entity_kind = "formal"
            self.selected_entity_id = str(plant.get("id"))
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "delete_polygon",
                {"polygon_index": polygon_index},
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "delete_staging_polygon",
                {"polygon_index": polygon_index, "actual_polygon_index": actual_polygon_index},
            )

        self.set_merge_staging_mode(False)
        self.set_split_staging_mode(False)
        self._notify_annotation_changed()
        self.update_display()
        return True, "暂存区域已删除"

    def split_selected_staging_polygon(self, line_start, line_end, gap=5):
        selected_kind, staging = self.get_selected_entity()
        if selected_kind != "staging" or not staging:
            return False, "请先选择一个暂存区域"

        polygon = staging.get("polygon", [])
        if len(polygon) < 3:
            return False, "当前暂存区域无法切割"

        if math.dist((float(line_start[0]), float(line_start[1])), (float(line_end[0]), float(line_end[1]))) < 3:
            return False, "切割线太短"

        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        padding = max(8, int(gap) + 6)
        min_x = math.floor(min(x_coords)) - padding
        min_y = math.floor(min(y_coords)) - padding
        max_x = math.ceil(max(x_coords)) + padding
        max_y = math.ceil(max(y_coords)) + padding
        width = max(2, int(max_x - min_x + 1))
        height = max(2, int(max_y - min_y + 1))
        mask = np.zeros((height, width), dtype=np.uint8)

        local_polygon = np.array(
            [[int(round(point[0] - min_x)), int(round(point[1] - min_y))] for point in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [local_polygon], 255)

        local_start = (int(round(line_start[0] - min_x)), int(round(line_start[1] - min_y)))
        local_end = (int(round(line_end[0] - min_x)), int(round(line_end[1] - min_y)))
        cv2.line(mask, local_start, local_end, 0, thickness=max(5, int(gap)))

        split_polygons = self._extract_split_polygons_from_mask(mask, min_x, min_y, polygon)
        if len(split_polygons) < 2:
            return False, "切割后没有得到两个有效区域"

        polygon_index = staging["polygon_index"]
        label = staging.get("label", "stem")
        if staging["owner_kind"] == "preview":
            old_polygons = copy.deepcopy(self.current_plant_polygons)
            old_labels = copy.deepcopy(self.current_plant_labels)
            self.current_plant_polygons = (
                self.current_plant_polygons[:polygon_index]
                + split_polygons
                + self.current_plant_polygons[polygon_index + 1:]
            )
            labels = self._ensure_label_slots(self.current_plant_labels, len(old_polygons))
            self.current_plant_labels = labels[:polygon_index] + [label, label] + labels[polygon_index + 1:]
            self._record_preview_state_change(
                old_polygons,
                old_labels,
                "split_polygon",
                {"polygon_index": polygon_index, "gap": gap},
            )
            self.select_entity("staging", self._make_staging_entity_id("preview", None, polygon_index))
        else:
            plant = staging["plant"]
            old_polygons = copy.deepcopy(plant.get("polygons", []))
            old_labels = copy.deepcopy(plant.get("labels", []))
            actual_polygon_index = staging.get("actual_polygon_index", polygon_index)
            polygons = plant.get("polygons", [])
            plant["polygons"] = polygons[:actual_polygon_index] + split_polygons + polygons[actual_polygon_index + 1:]
            if (
                self.mode == "fine_tune"
                and int(plant.get("id", 0)) == int(self.fine_tune_instance_id or 0)
            ) or staging.get("label_storage") == "fine_tune_outer":
                labels = list(plant.get("labels", []) or [])
                while len(labels) <= polygon_index:
                    labels.append("stem")
            else:
                labels, _ = self._normalize_labels_for_polygons(plant.get("labels", []), old_polygons)
            plant["labels"] = labels[:polygon_index] + [label, label] + labels[polygon_index + 1:]
            touch_instance(plant, "ai_modified" if plant.get("source") in ("ai_accepted", "ai_assisted") else None)
            self._record_fine_tune_state_change(
                plant,
                old_polygons,
                old_labels,
                "split_polygon",
                {"polygon_index": polygon_index, "actual_polygon_index": actual_polygon_index, "gap": gap},
            )
            self._notify_preannotation_adjustment(
                plant.get("id"),
                "split_staging_polygon",
                {"polygon_index": polygon_index, "actual_polygon_index": actual_polygon_index, "gap": gap},
            )
            self.select_entity("staging", self._make_staging_entity_id("formal", plant.get("id"), polygon_index))

        self.set_merge_staging_mode(False)
        self.set_split_staging_mode(False)
        self._notify_annotation_changed()
        self.update_display()
        return True, "暂存区域已切割为两个区域"

    def confirm_preview_and_save(self):
        """覆盖旧逻辑：去除区域只作为内洞，不产生 label。"""
        if self.is_summary:
            return False
        if self.current_points:
            self.save_current_polygon()
        if self.current_removal_points:
            self.save_current_removal_region()
        final_polygons, final_labels = self._compose_preview_instance_data()

        if not final_polygons:
            return False

        if hasattr(self, "_original_plant_id") and self._original_plant_id:
            instance_id = self._original_plant_id
            delattr(self, "_original_plant_id")
        else:
            instance_id = self.current_plant_id
            self.current_plant_id += 1

        new_instance = make_formal_instance(instance_id=instance_id, polygons=final_polygons, source="manual")
        new_instance["labels"] = final_labels
        self.plants.append(new_instance)
        self.plants.sort(key=lambda item: int(item.get("id", 0)))
        self.main_stack.append({"action": "confirm_preview", "instance": copy.deepcopy(new_instance)})
        if len(self.main_stack) > self.max_stack_depth:
            self.main_stack.pop(0)
        self.redo_main_stack = []

        self.current_plant_polygons = []
        self.current_plant_labels = []
        self.removal_regions = []
        self.current_points = []
        self.current_removal_points = []
        self.current_snap_point = None
        self.selected_entity_kind = "formal"
        self.selected_entity_id = str(instance_id)
        self.update_display()
        self._notify_selection_changed()
        return instance_id

    def get_main_window(self):
        """获取主窗口。"""
        parent = self.parent()
        while parent and not hasattr(parent, "toggle_edge_snap"):
            parent = parent.parent()
        return parent
