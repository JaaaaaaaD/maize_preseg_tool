import os
import traceback

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from config import SHORTCUTS
from utils.annotation_schema import current_timestamp, make_image_state, normalize_image_state
from utils.data_manager import load_annotation_from_coco
from utils.helpers import load_image
from utils.image_processor import preprocess_image


class MainWindowProjectMixin:
    def refresh_project_status(self):
        """刷新项目状态，实时读取最新的已完成和未完成状态。"""
        self.load_annotation_from_coco_container()
        if hasattr(self, "refresh_properties_panel"):
            self.refresh_properties_panel()
        self.left_label.update_display()
        self.update_plant_list()
        self.sync_summary_view()
        self.update_status_bar()
        QMessageBox.information(self, "刷新成功", "项目状态已更新为最新")

    def mark_annotation_changed(self):
        """标记当前图片有未保存修改。"""
        self.annotation_changed = True
        if self.current_image_state:
            self.current_image_state["last_modified_at"] = current_timestamp()
            timing_state = self.current_image_state.get("annotation_timing", {})
            total_seconds = float(timing_state.get("total_seconds", 0.0) or 0.0)
            if (
                self.current_image_path
                and not self.current_image_state.get("annotation_completed", False)
                and total_seconds <= 0.0
                and hasattr(self, "start_annotation_timer")
            ):
                self.start_annotation_timer()
        if self.current_image_path:
            if hasattr(self, "_save_preannotation_adjustment_records"):
                self._save_preannotation_adjustment_records(self.current_image_path)
            annotation_state = self.left_label.get_annotation_state()
            annotation = {
                "plants": annotation_state["plants"],
                "current_plant_id": annotation_state["current_plant_id"],
                "ignored_regions": self.left_label.ignored_regions,
                "image_state": self.current_image_state,
            }
            self.coco_container[self.current_image_path] = annotation
        self.update_status_bar()

    def clear_annotation_changed(self):
        """清除未保存修改标记。"""
        self.annotation_changed = False
        self.update_status_bar()

    def load_batch_images(self):
        """批量加载图片。"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "批量选择图片（可多选）",
            self.import_path or "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_paths:
            return

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = [path for path in file_paths if os.path.splitext(path.lower())[1] in valid_extensions]
        if not image_paths:
            QMessageBox.warning(self, "警告", "未选择有效的图片文件")
            return

        if image_paths:
            self.import_path = os.path.dirname(image_paths[0])

        self.image_paths = image_paths
        self.image_sequence_map = {path: index for index, path in enumerate(self.image_paths, start=1)}
        self.preprocess_cache.clear()
        self.coco_container.clear()
        if hasattr(self, "pause_annotation_timer"):
            self.pause_annotation_timer()
        self.preannotation_adjustment_records = []
        self.preannotation_records_by_image = {}
        self.preannotation_record_counter = 1
        self.preannotation_fine_tune_sessions = {}
        self._clear_preannotation_candidate()
        self.current_image_index = -1
        self.btn_prev.setEnabled(len(self.image_paths) > 1)
        self.btn_next.setEnabled(len(self.image_paths) > 1)
        self.goto_image(0)

        QMessageBox.information(
            self,
            "加载成功",
            f"成功加载 {len(self.image_paths)} 张图片",
        )

    def goto_image(self, index):
        """跳转到指定图片。"""
        if index < 0 or index >= len(self.image_paths):
            return

        if self.current_image_path:
            if hasattr(self, "_commit_annotation_timer_segment"):
                self._commit_annotation_timer_segment(reason="image_switch")
                self.annotation_timer.stop()
            if hasattr(self, "_save_preannotation_adjustment_records"):
                self._save_preannotation_adjustment_records(self.current_image_path)
            annotation_state = self.left_label.get_annotation_state()
            annotation = {
                "plants": annotation_state["plants"],
                "current_plant_id": annotation_state["current_plant_id"],
                "ignored_regions": self.left_label.ignored_regions,
                "image_state": self.current_image_state,
            }
            self.coco_container[self.current_image_path] = annotation

        image_path = self.image_paths[index]
        try:
            image = load_image(image_path)
            if not image:
                QMessageBox.warning(self, "警告", f"无法加载图片: {image_path}")
                return

            self.current_image = image
            self.current_image_path = image_path
            self.current_image_index = index

            preprocessed_data = self.preprocess_cache.get(image_path)
            if not preprocessed_data:
                preprocessed_data = preprocess_image(self.current_image)
                self.preprocess_cache[image_path] = preprocessed_data

            self.left_label.set_image(self.current_image, preprocessed_data)
            self.right_label.set_image(self.current_image, preprocessed_data)
            self.current_preannotation_candidate = None

            self.load_annotation_from_coco_container()
            if hasattr(self, "_load_preannotation_adjustment_records"):
                self._load_preannotation_adjustment_records(image_path)

            self.update_snap_button_state()
            self.clear_undo_stack()
            self.clear_annotation_changed()
            self.update_plant_list()
            self.sync_summary_view()
            if hasattr(self, "update_timing_panel"):
                self.update_timing_panel()
            self.update_status_bar()
            if hasattr(self, "sync_interaction_state"):
                self.sync_interaction_state()
            self._update_preannotation_controls()
        except Exception as error:
            QMessageBox.critical(self, "错误", f"加载图片失败: {error}")
            traceback.print_exc()

    def load_annotation_from_coco_container(self):
        """从COCO容器加载标注。"""
        if not self.current_image_path:
            return

        annotation = None
        if self.current_image_path in self.coco_container:
            annotation = self.coco_container[self.current_image_path]
        else:
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            coco_path = os.path.join(self.save_path or os.getcwd(), f"{base_name}_coco.json")
            annotation = load_annotation_from_coco(coco_path)
            if annotation:
                self.coco_container[self.current_image_path] = annotation

        if annotation:
            self.left_label.set_annotation_state(
                annotation["plants"],
                current_plant_id=annotation.get("current_plant_id", 1),
            )
            self.left_label.ignored_regions = annotation.get("ignored_regions", [])
            self.current_image_state = normalize_image_state(
                self.current_image_path,
                annotation.get("image_state", make_image_state(self.current_image_path)),
            )
            self.current_annotation_hash = annotation.get("annotation_hash")
        else:
            self.left_label.set_annotation_state([], current_plant_id=1)
            self.left_label.ignored_regions = []
            self.current_image_state = normalize_image_state(
                self.current_image_path,
                make_image_state(self.current_image_path, annotation_completed=False),
            )
            self.current_annotation_hash = None

    def prev_image(self):
        if self.current_image_index > 0:
            self.goto_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.goto_image(self.current_image_index + 1)

    def mark_current_image_completed(self):
        """标记当前图片为已完成。"""
        if not self.current_image_path:
            return
        self.current_image_state["annotation_completed"] = True
        self.current_image_state["dirty_since_last_train"] = True
        self.mark_annotation_changed()
        if hasattr(self, "pause_annotation_timer"):
            self.pause_annotation_timer()
        if hasattr(self, "_save_preannotation_adjustment_records"):
            self._save_preannotation_adjustment_records(self.current_image_path)
        QMessageBox.information(self, "状态更新", "当前图片已标记为已完成")

    def mark_current_image_incomplete(self):
        """取消当前图片已完成状态。"""
        if not self.current_image_path:
            return
        self.current_image_state["annotation_completed"] = False
        self.current_image_state["dirty_since_last_train"] = False
        self.mark_annotation_changed()
        if hasattr(self, "_save_preannotation_adjustment_records"):
            self._save_preannotation_adjustment_records(self.current_image_path)
        QMessageBox.information(self, "状态更新", "当前图片已取消已完成状态")

    def toggle_annotation_status(self):
        """兼容旧按钮：在已完成 / 未完成间切换。"""
        if self.current_image_state.get("annotation_completed"):
            self.mark_current_image_incomplete()
        else:
            self.mark_current_image_completed()

        if hasattr(self, "refresh_properties_panel"):
            self.refresh_properties_panel()

    def toggle_edge_snap(self):
        """切换边缘吸附。"""
        self.left_label.edge_snap_enabled = not self.left_label.edge_snap_enabled
        self.update_snap_button_state()
        if not self.left_label.edge_snap_enabled:
            self.left_label.current_snap_point = None
            self.left_label.update_display()
        self.update_status_bar()

    def update_snap_button_state(self):
        """更新边缘吸附按钮状态。"""
        if self.left_label.edge_snap_enabled:
            self.btn_toggle_snap.setText(f"边缘吸附: 开启 ({SHORTCUTS['TOGGLE_EDGE_SNAP']})")
        else:
            self.btn_toggle_snap.setText(f"边缘吸附: 关闭 ({SHORTCUTS['TOGGLE_EDGE_SNAP']})")
