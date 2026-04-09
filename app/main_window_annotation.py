import copy
import os

from PyQt5.QtWidgets import QInputDialog, QMessageBox

from components.help_dialog import HelpDialog
from config import SHORTCUTS
from utils.annotation_schema import format_elapsed_seconds


class MainWindowAnnotationMixin:
    def _update_staging_controls(self):
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        in_fine_tune = self.left_label.mode == "fine_tune"
        selected_kind, _ = self.left_label.get_selected_entity()
        vertex_mode_active = self.left_label.add_vertex_mode or getattr(self.left_label, "delete_vertex_mode", False)
        has_staging = in_fine_tune and (not vertex_mode_active) and selected_kind == "staging"
        has_removal = (not vertex_mode_active) and selected_kind == "removal"
        can_delete_area = has_staging or has_removal
        can_split_staging = in_fine_tune and has_staging
        staging_count = 0
        if in_fine_tune and hasattr(self.left_label, "_iter_active_fine_tune_staging_areas"):
            staging_count = sum(1 for _ in self.left_label._iter_active_fine_tune_staging_areas())
        can_merge_staging = in_fine_tune and (not vertex_mode_active) and staging_count >= 2

        if hasattr(self, "btn_apply_staging_label"):
            self.btn_apply_staging_label.setEnabled(has_staging)
        if hasattr(self, "btn_delete_staging_polygon"):
            self.btn_delete_staging_polygon.setEnabled(can_delete_area)
        if hasattr(self, "btn_split_staging_polygon"):
            self.btn_split_staging_polygon.setEnabled(can_split_staging)
            self.btn_split_staging_polygon.setText(
                "退出切割暂存区域" if self.left_label.split_staging_mode else "切割选中暂存区域"
            )
        if hasattr(self, "btn_merge_staging_polygon"):
            self.btn_merge_staging_polygon.setEnabled(can_merge_staging)
            self.btn_merge_staging_polygon.setText(
                "退出合并暂存区域" if getattr(self.left_label, "merge_staging_mode", False) else "合并暂存区域"
            )

    def _can_activate_fine_tune_tool(self, tool_name):
        if self.left_label.mode == "fine_tune":
            return True
        QMessageBox.warning(self, "警告", f"请先进入微调模式，再使用{tool_name}")
        return False

    def _stop_removal_region_drawing(self):
        if not self.left_label.removing_region:
            return
        self.left_label.removing_region = False
        self.left_label.current_removal_points = []
        self.left_label.current_snap_point = None
        if hasattr(self, "btn_removal_region"):
            self.btn_removal_region.setText("去除区域 (R)")

    def sync_label_combo_with_selection(self):
        if not hasattr(self, "combo_label"):
            return
        selected_kind, selected_entity = self.left_label.get_selected_entity()
        if selected_kind != "staging" or not selected_entity:
            return

        target_label = selected_entity.get("label")
        if not target_label:
            return

        index = self.combo_label.findText(target_label)
        if index < 0 or index == self.combo_label.currentIndex():
            return

        blocked = self.combo_label.blockSignals(True)
        try:
            self.combo_label.setCurrentIndex(index)
        finally:
            self.combo_label.blockSignals(blocked)

    def _has_active_preview_session(self):
        return bool(
            getattr(self, "_is_continuing_annotation", False)
            or self.left_label.current_plant_polygons
            or self.left_label.removal_regions
            or self.left_label.current_points
            or self.left_label.current_removal_points
        )

    def _reset_preview_annotation_buffers(self, clear_tracking=True):
        self.left_label.current_plant_polygons = []
        self.left_label.current_plant_labels = []
        self.left_label.removal_regions = []
        self.left_label.current_points = []
        self.left_label.current_removal_points = []
        self.left_label.current_snap_point = None
        self.left_label.selected_plant_id = None
        self.left_label.selected_entity_kind = None
        self.left_label.selected_entity_id = None
        if clear_tracking:
            for attr_name in ("_original_plant_id", "_original_plant_snapshot"):
                if hasattr(self.left_label, attr_name):
                    delattr(self.left_label, attr_name)

    def _restore_continue_annotation_snapshot(self):
        snapshot = getattr(self.left_label, "_original_plant_snapshot", None)
        plant_id = getattr(self.left_label, "_original_plant_id", None)
        if snapshot:
            self.left_label.plants = [
                plant for plant in self.left_label.plants if int(plant.get("id", 0)) != int(snapshot.get("id", 0))
            ]
            self.left_label.plants.append(copy.deepcopy(snapshot))
            self.left_label.plants.sort(key=lambda item: int(item.get("id", 0)))
            plant_id = snapshot.get("id", plant_id)

        self._is_continuing_annotation = False
        if hasattr(self, "btn_continue_annotation"):
            self.btn_continue_annotation.setText("继续标注选中植株")
        self._reset_preview_annotation_buffers(clear_tracking=True)
        if plant_id is not None:
            self.left_label.select_entity("formal", plant_id)
        else:
            self.left_label.update_display()
        self.update_plant_list()
        self.sync_summary_view()
        if hasattr(self, "refresh_properties_panel"):
            self.refresh_properties_panel()
        self.update_status_bar()

    def apply_selected_staging_label(self):
        if self.left_label.mode != "fine_tune":
            QMessageBox.warning(self, "警告", "修改暂存区域标签仅可在微调模式下使用")
            return
        selected_kind, staging = self.left_label.get_selected_entity()
        if selected_kind != "staging" or not staging:
            QMessageBox.warning(self, "警告", "请先选中一个暂存区域")
            return

        label_options = []
        if hasattr(self, "combo_label"):
            label_options = [self.combo_label.itemText(index).strip() for index in range(self.combo_label.count())]
            label_options = [label for label in label_options if label]
        current_label = (staging.get("label") or "").strip() or (label_options[0] if label_options else "stem")
        if current_label not in label_options:
            label_options.insert(0, current_label)

        selected_label, ok = QInputDialog.getItem(
            self,
            "修改暂存区域标签",
            "请选择该暂存区域的新标签:",
            label_options or [current_label],
            max(0, (label_options or [current_label]).index(current_label)),
            False,
        )
        if not ok:
            return
        selected_label = selected_label.strip()
        if not selected_label:
            QMessageBox.warning(self, "警告", "当前没有可用标签")
            return

        success, message = self.left_label.update_selected_staging_label(selected_label)
        if not success:
            QMessageBox.warning(self, "警告", message)
            return
        self.mark_annotation_changed()
        self.sync_summary_view()
        self.update_plant_list()
        self.update_undo_redo_state()
        if hasattr(self, "refresh_properties_panel"):
            self.refresh_properties_panel()
        if hasattr(self, "combo_label"):
            label_index = self.combo_label.findText(selected_label)
            if label_index >= 0:
                self.combo_label.setCurrentIndex(label_index)
        self.sync_label_combo_with_selection()
        self._update_staging_controls()
        self.update_status_bar()
    def delete_selected_staging_polygon(self):
        success, message = self.left_label.delete_selected_staging_polygon()
        if not success:
            QMessageBox.warning(self, "警告", message)
            return
        self.mark_annotation_changed()
        self.sync_summary_view()
        self.update_plant_list()
        self.update_undo_redo_state()
        if hasattr(self, "refresh_properties_panel"):
            self.refresh_properties_panel()
        self._update_staging_controls()
        self.update_status_bar()

    def delete_selected_staging_polygon_shortcut(self):
        selected_kind, _ = self.left_label.get_selected_entity()
        if selected_kind not in ("staging", "removal"):
            return
        self.delete_selected_staging_polygon()
    def toggle_split_staging_polygon(self):
        selected_kind, _ = self.left_label.get_selected_entity()
        if not self.left_label.split_staging_mode and (
            self.left_label.mode != "fine_tune" or selected_kind != "staging"
        ):
            QMessageBox.warning(self, "警告", "请先接受或拒绝当前预标注候选，再进行微调")
            return
        if not self.left_label.split_staging_mode:
            self._stop_removal_region_drawing()
        if self.left_label.add_vertex_mode:
            self.left_label.exit_add_vertex_mode()
            if hasattr(self, "btn_add_vertex"):
                self.btn_add_vertex.setText("添加顶点")
        if getattr(self.left_label, "delete_vertex_mode", False):
            self.left_label.exit_delete_vertex_mode()
            if hasattr(self, "btn_delete_vertex"):
                self.btn_delete_vertex.setText("删除顶点")
        if getattr(self.left_label, "merge_staging_mode", False):
            self.left_label.set_merge_staging_mode(False)
        self.left_label.set_split_staging_mode(not self.left_label.split_staging_mode)
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self._update_staging_controls()
        self.update_status_bar()

    def toggle_merge_staging_polygon(self):
        staging_count = 0
        if hasattr(self.left_label, "_iter_active_fine_tune_staging_areas"):
            staging_count = sum(1 for _ in self.left_label._iter_active_fine_tune_staging_areas())
        if not getattr(self.left_label, "merge_staging_mode", False) and (
            self.left_label.mode != "fine_tune" or staging_count < 2
        ):
            QMessageBox.warning(self, "警告", "请先进入微调模式，并确保当前至少有两个暂存区域")
            return
        if not getattr(self.left_label, "merge_staging_mode", False):
            self._stop_removal_region_drawing()
        if self.left_label.add_vertex_mode:
            self.left_label.exit_add_vertex_mode()
            if hasattr(self, "btn_add_vertex"):
                self.btn_add_vertex.setText("添加顶点")
        if getattr(self.left_label, "delete_vertex_mode", False):
            self.left_label.exit_delete_vertex_mode()
            if hasattr(self, "btn_delete_vertex"):
                self.btn_delete_vertex.setText("删除顶点")
        if self.left_label.split_staging_mode:
            self.left_label.set_split_staging_mode(False)
        new_state = not getattr(self.left_label, "merge_staging_mode", False)
        self.left_label.set_merge_staging_mode(new_state)
        if new_state and hasattr(self, "sam_info_text"):
            self.sam_info_text.append("请分别点击两个暂存区域的两个顶点，系统会保留远离另一区域的边界并自动完成合并")
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self._update_staging_controls()
        self.update_status_bar()
    def toggle_ignore_region(self):
        """切换忽略区域绘制模式。"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        has_unsaved_points = False
        message = ""

        if not self.ignoring_region:
            if self.left_label.current_points:
                has_unsaved_points = True
                message = "当前有未保存的标注点"
            elif self.left_label.current_removal_points:
                has_unsaved_points = True
                message = "当前有未保存的去除区域点"

            if has_unsaved_points:
                reply = QMessageBox.question(
                    self,
                    "未保存的点",
                    f"{message}，是否忽略这些点并切换到忽略区域模式？",
                    QMessageBox.Ignore | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if reply == QMessageBox.Cancel:
                    return
                if reply == QMessageBox.Ignore:
                    if self.left_label.current_points:
                        self.left_label.current_points = []
                    elif self.left_label.current_removal_points:
                        self.left_label.current_removal_points = []
                    self.left_label.current_snap_point = None
                    self.left_label.update_display()

        self.ignoring_region = not self.ignoring_region
        if self.ignoring_region:
            self.btn_ignore_region.setText(f"退出忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            self.left_label.ignoring_region = True
            self.left_label.removing_region = False
            self.left_label.current_ignored_points = []
            self.btn_removal_region.setText("去除区域 (R)")
        else:
            self.btn_ignore_region.setText(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            self.left_label.ignoring_region = False

    def toggle_removal_region(self):
        """切换去除区域绘制模式。"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        if self.left_label.preannotation_box_mode:
            QMessageBox.warning(self, "警告", "请先完成或取消当前框选预标注，再切换去除区域模式")
            return

        if getattr(self.left_label, "candidate_instances", None):
            if self.left_label.candidate_instances:
                QMessageBox.warning(self, "警告", "请先接受或拒绝当前预标注候选，再切换去除区域模式")
                return

        entering_mode = not self.left_label.removing_region

        if entering_mode:
            has_unsaved_points = False
            message = ""
            if self.left_label.current_points:
                has_unsaved_points = True
                message = "当前有未保存的标注点"
            elif self.left_label.current_ignored_points:
                has_unsaved_points = True
                message = "当前有未保存的忽略区域点"

            if has_unsaved_points:
                reply = QMessageBox.question(
                    self,
                    "未保存的点",
                    f"{message}，是否忽略这些点并切换到去除区域模式？",
                    QMessageBox.Ignore | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if reply == QMessageBox.Cancel:
                    return
                if reply == QMessageBox.Ignore:
                    if self.left_label.current_points:
                        self.left_label.current_points = []
                    elif self.left_label.current_ignored_points:
                        self.left_label.current_ignored_points = []
                    self.left_label.current_snap_point = None
        elif self.left_label.current_removal_points:
            reply = QMessageBox.question(
                self,
                "未保存的去除区域",
                "当前有未保存的去除区域点，是否丢弃并退出去除区域模式？",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            self.left_label.current_removal_points = []
            self.left_label.current_snap_point = None
            if self.left_label.selected_entity_kind == "removal" and not self.left_label.removal_regions:
                self.left_label.selected_entity_kind = None
                self.left_label.selected_entity_id = None

        self.left_label.removing_region = entering_mode
        if self.left_label.removing_region:
            self.btn_removal_region.setText("退出去除区域 (R)")
            self.left_label.ignoring_region = False
            self.left_label.current_removal_points = []
            if self.left_label.mode == "fine_tune" and self.left_label.add_vertex_mode:
                self.left_label.exit_add_vertex_mode()
                if hasattr(self, "btn_add_vertex"):
                    self.btn_add_vertex.setText("添加顶点")
            if self.left_label.mode == "fine_tune" and getattr(self.left_label, "delete_vertex_mode", False):
                self.left_label.exit_delete_vertex_mode()
                if hasattr(self, "btn_delete_vertex"):
                    self.btn_delete_vertex.setText("删除顶点")
            if self.left_label.mode == "fine_tune" and self.left_label.split_staging_mode:
                self.left_label.set_split_staging_mode(False)
            if self.left_label.mode == "fine_tune" and getattr(self.left_label, "merge_staging_mode", False):
                self.left_label.set_merge_staging_mode(False)
            self.btn_ignore_region.setText(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            self.ignoring_region = False
        else:
            self.btn_removal_region.setText("去除区域 (R)")
            self.left_label.removing_region = False
            self.left_label.current_removal_points = []
            self.left_label.current_snap_point = None

        self.left_label.update_display()
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self._update_staging_controls()
        self.update_status_bar()

    def toggle_projection(self):
        """切换投影框显示。"""
        self.projection_enabled = not getattr(self, "projection_enabled", False)
        if self.projection_enabled:
            self.btn_toggle_projection.setText("投影框: 开启")
        else:
            self.btn_toggle_projection.setText("投影框: 关闭")
        self.sync_summary_view()
        self.update_status_bar()

    def clear_all_ignore_regions(self):
        """清除所有忽略区域。"""
        self.left_label.ignored_regions = []
        self.left_label.update_display()
        self.sync_summary_view()
        self.mark_annotation_changed()
        self.update_status_bar()

    def show_help(self):
        dialog = HelpDialog(self)
        dialog.exec_()

    def save_current_polygon(self):
        changed = False
        if self.ignoring_region:
            if self.left_label.save_current_ignored_region():
                changed = True
        elif self.left_label.removing_region:
            if self.left_label.save_current_removal_region():
                changed = True
        else:
            selected_label = self.combo_label.currentText()
            if self.left_label.save_current_polygon(label=selected_label):
                changed = True
        if changed:
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_undo_redo_state()
            if hasattr(self, "refresh_properties_panel"):
                self.refresh_properties_panel()
            self.update_status_bar()
        self._update_staging_controls()

    def save_plant(self):
        """兼容旧快捷键：保存当前手动实例。"""
        in_fine_tune_mode = self.left_label.mode == "fine_tune"

        if in_fine_tune_mode:
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            if hasattr(self, "refresh_properties_panel"):
                self.refresh_properties_panel()

            exited = self.left_label.exit_fine_tune_mode(save_changes=True)
            if not exited:
                self._update_staging_controls()
                self.update_status_bar()
                return
            if hasattr(self, "btn_fine_tune"):
                self.btn_fine_tune.setText("微调模式")
            if hasattr(self, "btn_add_vertex"):
                self.btn_add_vertex.setText("添加顶点")
                self.btn_add_vertex.setEnabled(False)
            if hasattr(self, "btn_delete_vertex"):
                self.btn_delete_vertex.setText("删除顶点")
                self.btn_delete_vertex.setEnabled(False)
            self.left_label.set_split_staging_mode(False)
        else:
            saved_id = self.left_label.confirm_preview_and_save()
            if saved_id:
                saved_instance = next((plant for plant in self.left_label.plants if plant["id"] == saved_id), None)

                if hasattr(self, "_is_continuing_annotation") and self._is_continuing_annotation:
                    self._is_continuing_annotation = False
                    if hasattr(self, "btn_continue_annotation"):
                        self.btn_continue_annotation.setText("继续标注选中植株")

                    self._reset_preview_annotation_buffers(clear_tracking=True)

                    self.left_label.update_display()
                    self.update_plant_list()
                    self.sync_summary_view()
                if saved_instance:
                    self.clear_undo_stack()
                self.sync_summary_view()
                self.update_plant_list()
                self.update_undo_redo_state()
                self.mark_annotation_changed()
                if hasattr(self, "refresh_properties_panel"):
                    self.refresh_properties_panel()
            else:
                QMessageBox.warning(self, "警告", "请先暂存至少一个区域")

        self._update_staging_controls()
        self.update_status_bar()

    def undo(self):
        """撤销临时绘制或实例级增减。"""
        if self.left_label.undo_last_action():
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            if hasattr(self, "refresh_properties_panel"):
                self.refresh_properties_panel()
            self.update_status_bar()
            return
        if not self.undo_stack:
            return

        self.is_undo_redo = True
        try:
            action = self.undo_stack.pop()
            if action["type"] == "add_instance":
                instance = action["data"]
                self.left_label.delete_plant(instance["id"])
                self.redo_stack.append(action)
            elif action["type"] == "delete_instance":
                instance = copy.deepcopy(action["data"])
                self.left_label.plants.append(instance)
                self.left_label.plants.sort(key=lambda item: item["id"])
                self.redo_stack.append(action)

            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()
            if hasattr(self, "refresh_properties_panel"):
                self.refresh_properties_panel()
        finally:
            self.is_undo_redo = False

    def redo(self):
        """重做操作。"""
        if self.left_label.redo_last_action():
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            if hasattr(self, "refresh_properties_panel"):
                self.refresh_properties_panel()
            self.update_status_bar()
            return
        if not self.redo_stack:
            return

        self.is_undo_redo = True
        try:
            action = self.redo_stack.pop()
            if action["type"] == "add_instance":
                instance = copy.deepcopy(action["data"])
                self.left_label.plants.append(instance)
                self.left_label.plants.sort(key=lambda item: item["id"])
                self.undo_stack.append(action)
            elif action["type"] == "delete_instance":
                instance = action["data"]
                self.left_label.delete_plant(instance["id"])
                self.undo_stack.append(action)

            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()
            if hasattr(self, "refresh_properties_panel"):
                self.refresh_properties_panel()
        finally:
            self.is_undo_redo = False

    def delete_plant(self):
        """删除当前选中正式实例或候选实例。"""
        selected_kind, selected_entity = self.left_label.get_selected_entity()
        if not selected_entity:
            text = self.combo_plants.currentText()
            try:
                parts = text.split()
                if len(parts) >= 2:
                    instance_id = int(parts[1])
                    self.left_label.select_entity("formal", instance_id)
                    selected_kind, selected_entity = self.left_label.get_selected_entity()
            except (TypeError, ValueError):
                selected_kind, selected_entity = None, None
        if selected_kind == "formal" and selected_entity:
            self.push_undo_action("delete_instance", selected_entity)
            if hasattr(self, "record_preannotation_instance_deleted"):
                self.record_preannotation_instance_deleted(selected_entity)
            deleted = self.left_label.delete_plant(selected_entity.get("id"))
            if deleted:
                self.mark_annotation_changed()
                self.sync_summary_view()
                self.update_plant_list()
                self.update_undo_redo_state()
                if hasattr(self, "refresh_properties_panel"):
                    self.refresh_properties_panel()
                self.update_status_bar()

    def undo_delete_plant(self):
        """撤销删除植株操作。"""
        undone = self.left_label.undo_delete_plant()
        if undone:
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_plant_list()
            self.refresh_properties_panel()
            self.update_status_bar()

    def continue_annotation(self):
        """继续标注选中的植株，或结束当前继续标注会话。"""
        if self.left_label.mode == "fine_tune":
            QMessageBox.warning(self, "警告", "请先退出当前微调模式，再继续标注植株")
            return

        if getattr(self, "_is_continuing_annotation", False):
            reply = QMessageBox.question(
                self,
                "结束继续标注",
                "当前植株仍在继续标注中。是否保存当前修改？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self.save_plant()
                return

            self._restore_continue_annotation_snapshot()
            return

        if hasattr(self, "current_image_state") and self.current_image_state.get("annotation_completed", False):
            reply = QMessageBox.question(
                self,
                "图片已完成标注",
                "此图片已标记为完成标注。是否取消完成状态并继续标注？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
            self.current_image_state["annotation_completed"] = False
            self.mark_annotation_changed()
            self.update_status_bar()

        if not hasattr(self, "combo_plants") or self.combo_plants.currentIndex() == -1:
            return

        selected_plant_id = self.combo_plants.currentData()
        if selected_plant_id is None:
            return

        selected_plant = None
        for plant in self.left_label.plants:
            if plant.get("id") == selected_plant_id:
                selected_plant = plant
                break

        if not selected_plant:
            return

        self._reset_preview_annotation_buffers(clear_tracking=False)
        self.left_label.load_preview_from_formal_instance(selected_plant)
        self.left_label.plants = [plant for plant in self.left_label.plants if plant.get("id") != selected_plant_id]
        self.left_label._original_plant_id = selected_plant_id
        self.left_label._original_plant_snapshot = copy.deepcopy(selected_plant)

        self._is_continuing_annotation = True
        self.btn_continue_annotation.setText("结束标注该植株")

        self.left_label.update_display()
        self.sync_summary_view()
        self.mark_annotation_changed()

    def toggle_add_vertex_mode(self):
        """切换添加顶点模式。"""
        if self.left_label.add_vertex_mode:
            self.left_label.exit_add_vertex_mode()
            self.btn_add_vertex.setText("添加顶点")
        else:
            if not self._can_activate_fine_tune_tool("添加顶点"):
                return
            self._stop_removal_region_drawing()
            if getattr(self.left_label, "delete_vertex_mode", False):
                self.left_label.exit_delete_vertex_mode()
                if hasattr(self, "btn_delete_vertex"):
                    self.btn_delete_vertex.setText("删除顶点")
            if self.left_label.split_staging_mode:
                self.left_label.set_split_staging_mode(False)
            if getattr(self.left_label, "merge_staging_mode", False):
                self.left_label.set_merge_staging_mode(False)
            self.left_label.enter_add_vertex_mode()
            if self.left_label.add_vertex_mode:
                self.btn_add_vertex.setText("退出添加顶点")
        self._update_staging_controls()
        self.update_status_bar()

    def toggle_delete_vertex_mode(self):
        """切换删除顶点模式。"""
        if getattr(self.left_label, "delete_vertex_mode", False):
            self.left_label.exit_delete_vertex_mode()
            self.btn_delete_vertex.setText("删除顶点")
        else:
            if not self._can_activate_fine_tune_tool("删除顶点"):
                return
            self._stop_removal_region_drawing()
            if self.left_label.add_vertex_mode:
                self.left_label.exit_add_vertex_mode()
                if hasattr(self, "btn_add_vertex"):
                    self.btn_add_vertex.setText("添加顶点")
            if self.left_label.split_staging_mode:
                self.left_label.set_split_staging_mode(False)
            if getattr(self.left_label, "merge_staging_mode", False):
                self.left_label.set_merge_staging_mode(False)
            self.left_label.enter_delete_vertex_mode()
            if getattr(self.left_label, "delete_vertex_mode", False):
                self.btn_delete_vertex.setText("退出删除顶点")
        self._update_staging_controls()
        self.update_status_bar()

    def toggle_fine_tune_mode(self):
        """切换微调模式。"""
        if self.left_label.mode != "fine_tune" and getattr(self.left_label, "candidate_instances", None):
            if self.left_label.candidate_instances:
                QMessageBox.warning(self, "警告", "请先接受或拒绝当前预标注候选，再进行微调")
                return

        if self.left_label.mode != "fine_tune" and self._has_active_preview_session():
            QMessageBox.warning(self, "警告", "请先完成或取消当前继续标注/暂存编辑，再进入微调模式")
            return

        if self.left_label.mode == "fine_tune":
            exited = self.left_label.exit_fine_tune_mode()
            if not exited:
                self._update_staging_controls()
                self.update_status_bar()
                return
            self.left_label.set_split_staging_mode(False)
            self.btn_fine_tune.setText("微调模式")
            self.left_label.ignoring_region = False
            self.left_label.removing_region = False
            self.left_label.current_removal_points = []
            if hasattr(self, "btn_removal_region"):
                self.btn_removal_region.setText("去除区域 (R)")
            if hasattr(self, "btn_add_vertex"):
                self.btn_add_vertex.setEnabled(False)
                self.btn_add_vertex.setText("添加顶点")
            if hasattr(self, "btn_delete_vertex"):
                self.btn_delete_vertex.setEnabled(False)
                self.btn_delete_vertex.setText("删除顶点")
        else:
            if not hasattr(self, "combo_plants") or self.combo_plants.currentIndex() == -1:
                return

            selected_plant_id = self.combo_plants.currentData()
            if selected_plant_id is None:
                return

            selected_plant = None
            for plant in self.left_label.plants:
                if plant.get("id") == selected_plant_id:
                    selected_plant = plant
                    break

            if not selected_plant:
                return

            self.left_label.enter_fine_tune_mode(selected_plant_id)
            self.btn_fine_tune.setText("退出微调模式")
            self.left_label.ignoring_region = False
            self.left_label.removing_region = False
            self.left_label.current_removal_points = []
            if hasattr(self, "btn_removal_region"):
                self.btn_removal_region.setText("去除区域 (R)")
            if hasattr(self, "btn_add_vertex"):
                self.btn_add_vertex.setEnabled(True)
                self.btn_add_vertex.setText("添加顶点")
            if hasattr(self, "btn_delete_vertex"):
                self.btn_delete_vertex.setEnabled(True)
                self.btn_delete_vertex.setText("删除顶点")

        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self._update_staging_controls()
        self.update_status_bar()
    def update_plant_list(self):
        """更新植株列表。"""
        if not hasattr(self, "combo_plants"):
            return
        self.combo_plants.clear()
        sorted_plants = sorted(self.left_label.plants, key=lambda item: int(item["id"]))
        for plant in sorted_plants:
            class_name = "plant"
            self.combo_plants.addItem(f"{class_name} {plant['id']}", plant["id"])

    def sync_summary_view(self):
        """同步右侧总览视图。"""
        if not self.current_image:
            return

        plants = copy.deepcopy(self.left_label.plants)
        ignored_regions = copy.deepcopy(self.left_label.ignored_regions)
        preview_instance_id = getattr(self.left_label, "_original_plant_id", None) or self.left_label.current_plant_id
        preview_instance = self.left_label.build_preview_formal_instance(instance_id=preview_instance_id)
        if preview_instance:
            plants = [plant for plant in plants if int(plant.get("id", 0)) != int(preview_instance.get("id", 0))]
            plants.append(preview_instance)
            plants.sort(key=lambda item: int(item.get("id", 0)))

        self.right_label.set_annotation_state(
            plants,
            current_plant_id=self.left_label.current_plant_id,
        )
        self.right_label.ignored_regions = ignored_regions

        if hasattr(self, "projection_enabled") and self.projection_enabled:
            left_view_rect = getattr(self.left_label, "get_view_rect", lambda: None)()
            if left_view_rect:
                self.right_label.projection_rect = left_view_rect
            else:
                self.right_label.projection_rect = None
        else:
            self.right_label.projection_rect = None

        self.right_label.update_display()

    def update_status_bar(self):
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        """更新状态栏信息。"""
        status_parts = []

        if self.current_image_path:
            image_name = os.path.basename(self.current_image_path)
            status_parts.append(f"图片: {image_name}")
            if self.current_image_index >= 0 and self.image_paths:
                status_parts.append(f"({self.current_image_index + 1}/{len(self.image_paths)})")

        if self.current_image_state:
            completed = self.current_image_state.get("annotation_completed", False)
            status_parts.append(f"状态: {'已完成' if completed else '未完成'}")

        if self.annotation_changed:
            status_parts.append("有未保存修改")

        if self.ignoring_region:
            status_parts.append("忽略区域模式")

        if self.left_label.removing_region:
            status_parts.append("去除区域模式")

        if self.left_label.mode == "fine_tune":
            status_parts.append("微调模式")

        if self.left_label.add_vertex_mode:
            status_parts.append("添加顶点模式")

        if getattr(self.left_label, "delete_vertex_mode", False):
            status_parts.append("删除顶点模式")

        if self.left_label.split_staging_mode:
            status_parts.append("切割暂存区域模式")

        if self.left_label.edge_snap_enabled:
            status_parts.append("边缘吸附: 开启")
        else:
            status_parts.append("边缘吸附: 关闭")

        if getattr(self, "projection_enabled", False):
            status_parts.append("投影框: 开启")
        else:
            status_parts.append("投影框: 关闭")

        if self.left_label.preannotation_box_mode:
            status_parts.append("框选预标注模式")
        elif self.left_label.candidate_instances:
            status_parts.append("存在待处理预标注候选")

        if hasattr(self, "_get_live_timing_totals") and self.current_image_path:
            total_seconds, _, _ = self._get_live_timing_totals()
            status_parts.append(f"计时:{format_elapsed_seconds(total_seconds)}")

        state_name = getattr(self.interaction_state_machine, "state", None)
        if state_name == "preannotation_box":
            status_parts.append("预标注框选")
        elif state_name == "preannotation_candidate":
            status_parts.append("预标注候选")
        elif state_name == "fine_tune_add_vertex":
            status_parts.append("微调-添加顶点")
        elif state_name == "fine_tune_delete_vertex":
            status_parts.append("微调-删除顶点")
        elif state_name == "fine_tune_split_staging":
            status_parts.append("微调-切割暂存")
        elif state_name == "fine_tune_merge_staging":
            status_parts.append("微调-合并暂存")
        elif state_name == "fine_tune":
            status_parts.append("微调模式")
        elif state_name == "ignore_region":
            status_parts.append("忽略区域")
        elif state_name == "removal_region":
            status_parts.append("去除区域")

        self.statusBar().showMessage(" | ".join(status_parts))

    def push_undo_action(self, action_type, data):
        """压入撤销操作。"""
        if self.is_undo_redo:
            return
        self.undo_stack.append({"type": action_type, "data": data})
        self.redo_stack.clear()
        self.update_undo_redo_state()

    def clear_undo_stack(self):
        """清空撤销栈。"""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_state()

    def update_undo_redo_state(self):
        """更新撤销/重做按钮状态。"""
        if hasattr(self, "btn_undo"):
            has_undo = bool(self.undo_stack) or bool(self.left_label.main_stack) or bool(self.left_label.ignore_stack)
            self.btn_undo.setEnabled(has_undo)

        if hasattr(self, "btn_redo"):
            has_redo = bool(self.redo_stack) or bool(self.left_label.redo_main_stack) or bool(self.left_label.redo_ignore_stack)
            self.btn_redo.setEnabled(has_redo)
