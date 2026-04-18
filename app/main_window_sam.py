import copy
import json
import os
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from utils.annotation_schema import compute_annotation_hash, current_timestamp, make_formal_instance
from utils.helpers import calculate_signed_polygon_area
from utils.preannotation_records import (
    REASON_CODE_LABELS,
    append_event,
    append_reasoned_event,
    close_active_reason_segment,
    load_records_from_file,
    make_annotation_state,
    next_record_counter,
    normalize_record,
    save_records_to_file,
    set_active_reason,
    set_annotation_state,
    set_status,
    sync_active_reason_segment,
)
from utils.project_context import (
    ensure_project_for_images,
    mark_training_failed,
    mark_training_started,
    mark_training_success,
)

from .workers import SamTrainingWorker


class MainWindowSamMixin:
    @staticmethod
    def _sanitize_correction_image_name(image_path):
        image_name = Path(image_path or "").name or "unnamed"
        invalid_chars = '<>:"/\\|?*'
        safe_name = "".join("_" if char in invalid_chars else char for char in image_name).strip()
        return safe_name or "unnamed"
    def _prompt_load_sam_model(self, prompt_message="请先加载 SAM 模型"):
        reply = QMessageBox.question(
            self,
            "加载 SAM 模型",
            f"{prompt_message}\n\n是否现在选择并加载模型？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return False

        file_path, _ = QFileDialog.getOpenFileName(self, "选择 SAM 模型文件", "", "模型文件 (*.pth)")
        if not file_path:
            self.sam_info_text.append("已取消选择 SAM 模型文件")
            return False

        model_type, ok = QInputDialog.getItem(
            self,
            "选择模型类型",
            "请选择 SAM 模型类型:",
            ["vit_b", "vit_l", "vit_h"],
            0,
            False,
        )
        if not ok:
            self.sam_info_text.append("已取消选择 SAM 模型类型")
            return False

        self.sam_manager.load_model(file_path, model_type=model_type)
        self.sam_info_text.append(f"SAM 模型加载成功 (类型: {model_type})")
        return True

    def debug_print_coco_container(self):
        from utils.data_manager import debug_print_coco_container

        debug_print_coco_container(self.coco_container)

    def _cleanup_sam_training_worker(self):
        if self.sam_training_worker:
            self.sam_training_worker.deleteLater()
            self.sam_training_worker = None

    def _find_plant_by_id(self, instance_id):
        for plant in self.left_label.plants:
            if int(plant.get("id", 0)) == int(instance_id):
                return plant
        return None

    def _find_preannotation_record(self, record_id):
        for record in self.preannotation_adjustment_records:
            if record.get("record_id") == record_id:
                return record
        return None

    def _remove_preannotation_record(self, record_id):
        if not record_id:
            return False
        original_count = len(self.preannotation_adjustment_records)
        self.preannotation_adjustment_records = [
            record for record in self.preannotation_adjustment_records if record.get("record_id") != record_id
        ]
        return len(self.preannotation_adjustment_records) != original_count

    @staticmethod
    def _record_state_snapshot(polygons, labels=None):
        return make_annotation_state(polygons, labels)

    def _get_plant_state_snapshot(self, plant):
        if not plant:
            return self._record_state_snapshot([], [])
        return self._record_state_snapshot(plant.get("polygons", []), plant.get("labels", []))

    def _get_record_final_state_snapshot(self, record):
        if not record:
            return self._record_state_snapshot([], [])
        return self._record_state_snapshot(record.get("final_polygons", []), record.get("final_labels", []))

    def _set_record_final_state(self, record, polygons, labels=None):
        state = set_annotation_state(record, "final", polygons, labels)
        return {"polygons": state["polygons"], "labels": state["labels"]}

    def _close_reason_segment(self, record):
        if not record:
            return
        final_state = self._get_record_final_state_snapshot(record)
        close_active_reason_segment(
            record,
            end_polygons=final_state["polygons"],
            end_labels=final_state["labels"],
        )

    def _append_reasoned_adjustment(
        self,
        record,
        event_type,
        details=None,
        before_state=None,
        after_state=None,
        reason_code=None,
    ):
        before_state = before_state or self._get_record_final_state_snapshot(record)
        after_state = after_state or before_state
        append_reasoned_event(
            record,
            event_type,
            details=details,
            reason_code=reason_code if reason_code is not None else record.get("active_reason_code"),
            before_polygons=before_state["polygons"],
            before_labels=before_state["labels"],
            after_polygons=after_state["polygons"],
            after_labels=after_state["labels"],
        )

    def _get_correction_filename(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return None

        safe_name = self._sanitize_correction_image_name(target_image)
        if self.image_paths:
            target_name = Path(target_image).name
            duplicate_count = sum(1 for path in self.image_paths if Path(path).name == target_name)
            if duplicate_count > 1:
                index = self._resolve_image_sequence(target_image) or 1
                safe_name = f"{safe_name}_{index}"
        return f"correction_{safe_name}.json"

    def _get_legacy_correction_path(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return None
        index = self._resolve_image_sequence(target_image) or 1
        directory = self.save_path or os.getcwd()
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"image_{index}_correction.json")

    def _get_current_image_correction_path(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return None
        directory = self.save_path or os.getcwd()
        os.makedirs(directory, exist_ok=True)
        filename = self._get_correction_filename(target_image)
        if not filename:
            return None
        return os.path.join(directory, filename)

    def _get_existing_correction_path(self, image_path=None):
        current_path = self._get_current_image_correction_path(image_path)
        if current_path and os.path.exists(current_path):
            return current_path

        legacy_path = self._get_legacy_correction_path(image_path)
        if legacy_path and os.path.exists(legacy_path):
            return legacy_path
        return current_path

    def _get_cached_preannotation_records(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return []
        return copy.deepcopy((self.preannotation_records_by_image or {}).get(target_image, []))

    @staticmethod
    def _extract_image_name_from_correction_filename(file_name):
        base_name = Path(file_name or "").name
        if base_name.lower().endswith(".json"):
            base_name = base_name[:-5]
        if base_name.startswith("correction_"):
            base_name = base_name[len("correction_") :]
        return base_name

    def _resolve_image_path_by_name(self, image_name):
        image_name = Path(image_name or "").name
        if not image_name:
            return None
        for image_path in self.image_paths or []:
            if Path(image_path).name == image_name:
                return image_path
        return None

    @staticmethod
    def _build_preannotation_record_option_text(record):
        record_id = str(record.get("record_id") or "unknown")
        status = str(record.get("status") or "unknown")
        reason_count = len(record.get("reason_codes") or [])
        return f"{record_id} | {status} | reasons={reason_count}"

    def _refresh_imported_preannotation_record_selector(self):
        combo = getattr(self, "combo_imported_preannotation_record", None)
        if combo is None:
            return
        current_record_id = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if not self.current_image_path:
            combo.addItem("请先加载图片", "")
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        if not self.preannotation_adjustment_records:
            combo.addItem("当前图片暂无已导入记录", "")
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        for record in self.preannotation_adjustment_records:
            combo.addItem(self._build_preannotation_record_option_text(record), record.get("record_id"))
        combo.setEnabled(True)
        if current_record_id:
            self._set_combobox_data(combo, current_record_id, default_index=0)
        combo.blockSignals(False)
        self._refresh_imported_reason_segment_selector()

    def _get_selected_imported_preannotation_record(self):
        combo = getattr(self, "combo_imported_preannotation_record", None)
        if combo is None:
            return None
        record_id = combo.currentData() or None
        if not record_id:
            return None
        return self._find_preannotation_record(record_id)

    @staticmethod
    def _build_reason_segment_option_text(index, segment):
        reason_code = str(segment.get("reason_code") or "unspecified")
        reason_label = REASON_CODE_LABELS.get(reason_code, reason_code)
        event_count = len(segment.get("event_log") or [])
        return f"#{index + 1} | {reason_label} | events={event_count}"

    def _refresh_imported_reason_segment_selector(self):
        combo = getattr(self, "combo_imported_reason_segment", None)
        if combo is None:
            return
        selected_segment_id = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        record = self._get_selected_imported_preannotation_record()
        reason_segments = list((record or {}).get("reason_segments") or [])
        if not reason_segments and record:
            fallback_reason = record.get("active_reason_code")
            if not fallback_reason:
                fallback_reason = (record.get("reason_codes") or [None])[0]
            reason_segments = [
                {
                    "segment_id": "record_overall",
                    "reason_code": fallback_reason,
                    "start_polygons": copy.deepcopy(record.get("original_polygons") or []),
                    "start_labels": copy.deepcopy(record.get("original_labels") or []),
                    "end_polygons": copy.deepcopy(record.get("final_polygons") or []),
                    "end_labels": copy.deepcopy(record.get("final_labels") or []),
                    "event_log": copy.deepcopy(record.get("event_log") or []),
                }
            ]
        self._imported_reason_segments_cache = reason_segments
        if not reason_segments:
            combo.addItem("当前记录暂无理由片段", "")
            combo.setEnabled(False)
            combo.blockSignals(False)
            self._set_reason_segment_summary_label(None)
            return
        for index, segment in enumerate(reason_segments):
            segment_id = segment.get("segment_id") or f"segment_{index + 1:04d}"
            combo.addItem(self._build_reason_segment_option_text(index, segment), segment_id)
        combo.setEnabled(True)
        if selected_segment_id:
            self._set_combobox_data(combo, selected_segment_id, default_index=0)
        combo.blockSignals(False)
        self._set_reason_segment_summary_label(self._get_selected_reason_segment())

    def _get_selected_reason_segment(self):
        combo = getattr(self, "combo_imported_reason_segment", None)
        if combo is None:
            return None
        segment_id = combo.currentData() or None
        for segment in getattr(self, "_imported_reason_segments_cache", []) or []:
            current_segment_id = segment.get("segment_id")
            if current_segment_id == segment_id:
                return segment
        return None

    @staticmethod
    def _count_vertices(polygons):
        total = 0
        for polygon in polygons or []:
            total += len(polygon or [])
        return total

    def _append_reason_segment_summary(self, segment):
        if not segment:
            return
        reason_code = str(segment.get("reason_code") or "unspecified")
        reason_label = REASON_CODE_LABELS.get(reason_code, reason_code)
        start_polygons = segment.get("start_polygons") or []
        end_polygons = segment.get("end_polygons") or []
        event_count = len(segment.get("event_log") or [])
        self.sam_info_text.append(
            "理由片段: "
            f"{reason_label} | events={event_count} | "
            f"polygons {len(start_polygons)}->{len(end_polygons)} | "
            f"vertices {self._count_vertices(start_polygons)}->{self._count_vertices(end_polygons)}"
        )

    def _set_reason_segment_summary_label(self, segment):
        label = getattr(self, "label_imported_reason_segment_summary", None)
        if label is None:
            return
        if not segment:
            label.setText("理由改动说明: 暂无")
            return
        reason_code = str(segment.get("reason_code") or "unspecified")
        reason_label = REASON_CODE_LABELS.get(reason_code, reason_code)
        start_polygons = segment.get("start_polygons") or []
        end_polygons = segment.get("end_polygons") or []
        event_count = len(segment.get("event_log") or [])
        label.setText(
            "理由改动说明: "
            f"{reason_label} | 事件={event_count} | "
            f"区域 {len(start_polygons)}->{len(end_polygons)} | "
            f"顶点 {self._count_vertices(start_polygons)}->{self._count_vertices(end_polygons)}"
        )

    def _show_reason_segment_preview(self, polygons, reason_code, stage_name):
        existing_candidate = self._get_selected_candidate()
        if existing_candidate and self.current_preannotation_candidate:
            QMessageBox.warning(self, "警告", "当前有未处理的预标注候选，请先接受/拒绝后再预览理由片段")
            return False
        preview_candidate_id = f"preview_{stage_name}_{current_timestamp().replace(' ', '_').replace(':', '')}"
        preview_candidate = {
            "candidate_id": preview_candidate_id,
            "polygons": copy.deepcopy(polygons or []),
            "confidence": None,
            "model_version": f"imported_reason_{stage_name}",
            "roi_box": [],
        }
        self.left_label.candidate_instances = [preview_candidate]
        self.left_label.select_entity("candidate", preview_candidate_id)
        self.left_label.update_display()
        reason_label = REASON_CODE_LABELS.get(reason_code, reason_code or "unspecified")
        self.sam_info_text.append(
            f"已预览理由片段({stage_name}): {reason_label}, polygons={len(preview_candidate['polygons'])}"
        )
        return True

    def _has_imported_reason_preview(self):
        if not self.left_label.candidate_instances:
            return False
        candidate = self.left_label.candidate_instances[0]
        model_version = str(candidate.get("model_version") or "")
        return model_version.startswith("imported_reason_")

    def _load_preannotation_adjustment_records(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            self.preannotation_adjustment_records = []
            self.preannotation_record_counter = 1
            self._sync_preannotation_reason_ui()
            self._refresh_imported_preannotation_record_selector()
            return []

        if target_image in self.preannotation_records_by_image:
            self.preannotation_adjustment_records = self._get_cached_preannotation_records(target_image)
        else:
            correction_path = self._get_existing_correction_path(target_image)
            if correction_path and os.path.exists(correction_path):
                self.preannotation_adjustment_records = load_records_from_file(correction_path, image_path=target_image)
                self.preannotation_records_by_image[target_image] = copy.deepcopy(self.preannotation_adjustment_records)
            else:
                self.preannotation_adjustment_records = []

        self.preannotation_record_counter = next_record_counter(self.preannotation_adjustment_records, default_value=1)
        self._sync_preannotation_reason_ui()
        self._refresh_imported_preannotation_record_selector()
        return self.preannotation_adjustment_records

    def _save_preannotation_adjustment_records(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return None
        self.preannotation_records_by_image[target_image] = copy.deepcopy(self.preannotation_adjustment_records)
        return None

    def _persist_preannotation_adjustment_records(self):
        if self.current_image_path:
            self._save_preannotation_adjustment_records(self.current_image_path)

    def _next_preannotation_record_id(self):
        record_id = f"pre_{self.preannotation_record_counter:04d}"
        self.preannotation_record_counter += 1
        return record_id

    def _get_current_preannotation_reason_code(self):
        if hasattr(self, "combo_preannotation_reason"):
            return self.combo_preannotation_reason.currentData() or None
        return self.preannotation_default_reason_code

    @staticmethod
    def _set_combobox_data(combo_box, value, default_index=0):
        if combo_box is None:
            return
        target_value = value
        if target_value is None:
            target_value = combo_box.itemData(default_index)
        for index in range(combo_box.count()):
            if combo_box.itemData(index) == target_value:
                combo_box.setCurrentIndex(index)
                return
        combo_box.setCurrentIndex(default_index)

    def _get_selected_preannotation_formal_plant(self):
        if self.left_label.mode == "fine_tune" and self.left_label.fine_tune_instance_id:
            plant = self._find_plant_by_id(self.left_label.fine_tune_instance_id)
            if plant and plant.get("preannotation_record_id"):
                return plant

        selected_kind, selected_entity = self.left_label.get_selected_entity()
        if selected_kind == "formal" and selected_entity and selected_entity.get("preannotation_record_id"):
            return selected_entity

        if hasattr(self, "combo_plants") and self.combo_plants.currentIndex() >= 0:
            plant = self._find_plant_by_id(self.combo_plants.currentData())
            if plant and plant.get("preannotation_record_id"):
                return plant
        return None

    def _get_active_preannotation_context(self):
        candidate = self._get_selected_candidate()
        if candidate:
            return {"kind": "candidate", "candidate": candidate, "plant": None, "record": None}

        plant = self._get_selected_preannotation_formal_plant()
        if not plant:
            return {"kind": None, "candidate": None, "plant": None, "record": None}

        record = self._find_preannotation_record(plant.get("preannotation_record_id"))
        return {"kind": "formal", "candidate": None, "plant": plant, "record": record}

    def _get_active_preannotation_record(self):
        context = self._get_active_preannotation_context()
        plant = context.get("plant")
        if not plant:
            return None
        instance_id = int(plant.get("id", 0))
        if instance_id not in self.preannotation_fine_tune_sessions:
            return None
        return context.get("record")

    def _sync_preannotation_reason_ui(self):
        if getattr(self, "_updating_preannotation_controls", False):
            return
        self._updating_preannotation_controls = True
        try:
            record = self._get_active_preannotation_record()
            reason_code = self.preannotation_default_reason_code
            if record:
                reason_code = record.get("active_reason_code")
            if hasattr(self, "combo_preannotation_reason"):
                self._set_combobox_data(self.combo_preannotation_reason, reason_code, default_index=0)
        finally:
            self._updating_preannotation_controls = False

    def on_preannotation_reason_changed(self, _index):
        if getattr(self, "_updating_preannotation_controls", False):
            return
        reason_code = self._get_current_preannotation_reason_code()
        self.preannotation_default_reason_code = reason_code
        record = self._get_active_preannotation_record()
        if not record:
            return
        previous_reason = record.get("active_reason_code")
        if not reason_code:
            QMessageBox.warning(self, "警告", "预标注微调阶段必须先选择一个理由")
            self._updating_preannotation_controls = True
            try:
                self._set_combobox_data(self.combo_preannotation_reason, previous_reason, default_index=0)
            finally:
                self._updating_preannotation_controls = False
            return
        if previous_reason != reason_code:
            self._close_reason_segment(record)
        set_active_reason(record, reason_code)
        if previous_reason != record.get("active_reason_code"):
            append_event(
                record,
                "reason_selected",
                {
                    "previous_reason_code": previous_reason,
                    "reason_code": record.get("active_reason_code"),
                },
                reason_code=record.get("active_reason_code"),
            )
            self._persist_preannotation_adjustment_records()

    @staticmethod
    def _extract_outer_polygons(polygons):
        outer_polygons = []
        for polygon in polygons or []:
            if calculate_signed_polygon_area(polygon) < 0:
                outer_polygons.append(copy.deepcopy(polygon))
        if outer_polygons:
            return outer_polygons
        return copy.deepcopy(polygons or [])

    def _append_preannotation_event(self, record, event_type, details=None):
        append_event(
            record,
            event_type,
            details=details,
            reason_code=record.get("active_reason_code"),
        )

    def _finalize_preannotation_record(self, record):
        event_types = {entry.get("event_type") for entry in record.get("event_log", []) if entry.get("event_type")}
        if event_types & {"candidate_ignored", "instance_ignored"}:
            set_status(record, "ignored")
        elif event_types & {"proposal_merged"}:
            set_status(record, "merged")
        elif event_types & {"candidate_rejected", "delete_instance"} and not record.get("final_polygons"):
            set_status(record, "rejected")
        else:
            set_status(record, "accepted")

    def _build_preannotation_record(self, candidate, formal_instance_id=None, status="accepted"):
        candidate_state = self._record_state_snapshot(candidate.get("polygons", []), candidate.get("labels", []))
        record = normalize_record(
            {
                "record_id": self._next_preannotation_record_id(),
                "image_path": self.current_image_path,
                "created_at": current_timestamp(),
                "updated_at": current_timestamp(),
                "model_path": self.sam_manager.model_path,
                "model_type": self.sam_manager.model_type,
                "roi_box": copy.deepcopy(candidate.get("roi_box", [])),
                "candidate_id": candidate.get("candidate_id"),
                "confidence": candidate.get("confidence"),
                "original_polygons": candidate_state["polygons"],
                "original_labels": candidate_state["labels"],
                "final_polygons": candidate_state["polygons"],
                "final_labels": candidate_state["labels"],
                "formal_instance_id": formal_instance_id,
                "status": status,
                "event_log": [],
                "reason_segments": [],
            }
        )
        set_active_reason(record, self.preannotation_default_reason_code)
        if record.get("active_reason_code"):
            append_event(
                record,
                "reason_selected",
                {
                    "previous_reason_code": None,
                    "reason_code": record.get("active_reason_code"),
                },
                reason_code=record.get("active_reason_code"),
            )
        return record

    def _remove_preannotation_formal_instance(self, instance_id):
        self.left_label.plants = [
            plant for plant in self.left_label.plants if int(plant.get("id", 0)) != int(instance_id)
        ]
        if self.left_label.selected_plant_id == instance_id:
            self.left_label.selected_plant_id = None
        if self.left_label.selected_entity_kind == "formal" and int(self.left_label.selected_entity_id or 0) == int(instance_id):
            self.left_label.selected_entity_kind = None
            self.left_label.selected_entity_id = None
        self.left_label.update_display()

    def ignore_selected_preannotation(self):
        context = self._get_active_preannotation_context()
        if context.get("kind") == "candidate":
            candidate = context.get("candidate")
            self.left_label.ignored_regions.extend(self._extract_outer_polygons(candidate.get("polygons", [])))
            self._clear_preannotation_candidate()
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_undo_redo_state()
            self.sam_info_text.append(f"已忽略 proposal: {candidate.get('candidate_id')}")
            return

        if context.get("kind") != "formal":
            QMessageBox.warning(self, "警告", "当前没有可忽略的 proposal")
            return

        plant = context.get("plant")
        if not plant:
            return
        instance_id = int(plant.get("id", 0))
        if self.left_label.mode == "fine_tune" and int(self.left_label.fine_tune_instance_id or 0) == instance_id:
            exited = self.left_label.exit_fine_tune_mode(save_changes=True)
            if not exited:
                return
            plant = self._find_plant_by_id(instance_id)
        if not plant:
            return
        self._remove_preannotation_record(plant.get("preannotation_record_id"))
        self.left_label.ignored_regions.extend(self._extract_outer_polygons(plant.get("polygons", [])))
        self._remove_preannotation_formal_instance(instance_id)
        self.mark_annotation_changed()
        self.sync_summary_view()
        self.update_plant_list()
        self.update_undo_redo_state()
        self._persist_preannotation_adjustment_records()
        self._update_preannotation_controls()
        self.sam_info_text.append(f"已忽略 proposal instance={instance_id}")

    def record_preannotation_instance_deleted(self, plant):
        if not plant or not plant.get("preannotation_record_id"):
            return False
        removed = self._remove_preannotation_record(plant.get("preannotation_record_id"))
        if removed:
            self._persist_preannotation_adjustment_records()
        self._update_preannotation_controls()
        return removed

    def _update_preannotation_controls(self):
        context = self._get_active_preannotation_context()
        has_candidate = context.get("kind") == "candidate"
        has_active_record = bool(context.get("kind"))
        has_image = bool(self.current_image_path)
        box_mode = bool(self.left_label.preannotation_box_mode)
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        if hasattr(self, "btn_sam_preannotate"):
            self.btn_sam_preannotate.setText("取消框选预标注" if box_mode else "框选预标注")
        if hasattr(self, "btn_sam_select_mode"):
            self.btn_sam_select_mode.setEnabled(has_candidate)
        if hasattr(self, "btn_save_staging_areas"):
            self.btn_save_staging_areas.setEnabled(has_candidate)
        if hasattr(self, "btn_ignore_preannotation"):
            self.btn_ignore_preannotation.setEnabled(has_active_record)
        if hasattr(self, "combo_preannotation_reason"):
            self.combo_preannotation_reason.setEnabled(has_image)
        self._sync_preannotation_reason_ui()
        self._refresh_imported_preannotation_record_selector()
        has_imported_record = bool(self.preannotation_adjustment_records)
        if hasattr(self, "btn_import_preannotation_records"):
            self.btn_import_preannotation_records.setEnabled(bool(self.image_paths))
        if hasattr(self, "btn_apply_imported_preannotation_record"):
            self.btn_apply_imported_preannotation_record.setEnabled(has_image and has_imported_record)
        reason_segment = self._get_selected_reason_segment()
        has_reason_segment = bool(reason_segment)
        if hasattr(self, "btn_preview_reason_segment_start"):
            self.btn_preview_reason_segment_start.setEnabled(has_image and has_reason_segment)
        if hasattr(self, "btn_preview_reason_segment_end"):
            self.btn_preview_reason_segment_end.setEnabled(has_image and has_reason_segment)
        if hasattr(self, "btn_clear_reason_segment_preview"):
            self.btn_clear_reason_segment_preview.setEnabled(self._has_imported_reason_preview())

    def _clear_preannotation_candidate(self, clear_box=True):
        self.current_preannotation_candidate = None
        self.left_label.candidate_instances = []
        if self.left_label.selected_entity_kind == "candidate":
            self.left_label.selected_entity_kind = None
            self.left_label.selected_entity_id = None
        if clear_box:
            self.left_label.set_preannotation_box_mode(False)
            self.left_label.clear_preannotation_box()
        else:
            self.left_label.update_display()
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self._update_preannotation_controls()

    def _get_selected_candidate(self):
        selected_kind, selected_entity = self.left_label.get_selected_entity()
        if selected_kind == "candidate" and selected_entity:
            return selected_entity
        if self.left_label.candidate_instances:
            return self.left_label.candidate_instances[0]
        return None

    def on_canvas_entity_selected(self, entity_kind, entity_id):
        self._update_preannotation_controls()
        if hasattr(self, "sync_label_combo_with_selection"):
            self.sync_label_combo_with_selection()
        if hasattr(self, "_update_staging_controls"):
            self._update_staging_controls()
        if hasattr(self, "refresh_properties_panel"):
            self.refresh_properties_panel()

    def on_preannotation_box_completed(self, rect):
        self.left_label.set_preannotation_box_mode(False)
        self._update_preannotation_controls()

        if not self.current_image or not self.sam_manager.has_model_loaded():
            self.left_label.clear_preannotation_box()
            return

        try:
            import numpy as np

            image = np.array(self.current_image.convert("RGB"))
            img_h, img_w = image.shape[:2]
            x1 = max(0, min(int(rect[0]), img_w - 1))
            y1 = max(0, min(int(rect[1]), img_h - 1))
            x2 = max(x1 + 1, min(int(rect[2]), img_w))
            y2 = max(y1 + 1, min(int(rect[3]), img_h))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise RuntimeError("Selected ROI is empty")

            predictor = self.sam_manager.get_predictor()
            predictor.set_image(crop)

            box = np.array([0, 0, crop.shape[1] - 1, crop.shape[0] - 1], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )

            if masks is None or len(masks) == 0:
                raise RuntimeError("SAM did not return a valid mask")

            scored_masks = []
            for mask, score in zip(masks, scores):
                area = int(np.sum(mask))
                if area <= 16:
                    continue
                scored_masks.append((float(score), area, mask))
            if not scored_masks:
                raise RuntimeError("Candidate mask area is too small")

            scored_masks.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_score, _, best_mask = scored_masks[0]

            from utils.sam_utils import mask_to_polygons, process_sam_polygons

            polygons = process_sam_polygons(mask_to_polygons(best_mask.astype(np.uint8) * 255, pixel_interval=50))
            if not polygons:
                raise RuntimeError("Failed to convert mask into a valid polygon")

            mapped_polygons = []
            for polygon in polygons:
                mapped_polygons.append([(float(point[0] + x1), float(point[1] + y1)) for point in polygon])

            candidate_id = f"candidate_{current_timestamp().replace(' ', '_').replace(':', '')}"
            candidate = {
                "candidate_id": candidate_id,
                "polygons": mapped_polygons,
                "confidence": float(best_score),
                "model_version": self.sam_manager.model_type,
                "roi_box": [x1, y1, x2, y2],
                "model_path": self.sam_manager.model_path,
            }

            self.current_preannotation_candidate = copy.deepcopy(candidate)
            self.left_label.candidate_instances = [candidate]
            self.left_label.select_entity("candidate", candidate_id)
            self.left_label.preannotation_box_rect = (x1, y1, x2, y2)
            self.left_label.update_display()
            if hasattr(self, "sync_interaction_state"):
                self.sync_interaction_state()
            self.sam_info_text.append(
                f"预标注完成: ROI=({x1},{y1})-({x2},{y2}), score={best_score:.4f}, polygons={len(mapped_polygons)}"
            )
            self._update_preannotation_controls()
        except Exception as error:
            self.sam_info_text.append(f"预标注失败: {str(error)}")
            self.left_label.clear_preannotation_box()
            QMessageBox.warning(self, "失败", f"预标注失败: {str(error)}")

    def run_sam_preannotation(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        if not self.sam_manager.has_model_loaded():
            QMessageBox.warning(self, "警告", "请先加载 SAM 模型")
            return
        if self.left_label.preannotation_box_mode:
            self.left_label.set_preannotation_box_mode(False)
            self.left_label.clear_preannotation_box()
            self.sam_info_text.append("已取消框选预标注")
            self._update_preannotation_controls()
            return

        if hasattr(self, "mark_sam_timing_used"):
            self.mark_sam_timing_used(auto_start=True)

        if self.left_label.candidate_instances:
            reply = QMessageBox.question(
                self,
                "覆盖当前候选",
                "当前还有未处理的预标注候选，是否丢弃并重新框选？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self._clear_preannotation_candidate()

        self.left_label.set_preannotation_box_mode(True)
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self.sam_info_text.append("请在左侧画布拖拽一个虚线框进行预标注")
        self._update_preannotation_controls()
    def enter_sam_select_mode(self):
        candidate = self._get_selected_candidate()
        if not candidate:
            QMessageBox.warning(self, "警告", "请先选择一个预标注候选")
            return
        if not self._get_current_preannotation_reason_code():
            QMessageBox.warning(self, "警告", "接受候选前请先在“当前理由”中选择一个理由")
            return

        if hasattr(self, "mark_sam_timing_used"):
            self.mark_sam_timing_used(auto_start=True)

        instance_id = self.left_label.current_plant_id
        self.left_label.current_plant_id += 1

        new_instance = make_formal_instance(
            instance_id=instance_id,
            polygons=copy.deepcopy(candidate.get("polygons", [])),
            source="ai_accepted",
            origin_model_version=self.sam_manager.model_type,
            origin_confidence=candidate.get("confidence"),
        )
        record = self._build_preannotation_record(candidate, formal_instance_id=instance_id, status="accepted")
        self._append_preannotation_event(record, "candidate_accepted", {"formal_instance_id": instance_id})
        self._finalize_preannotation_record(record)
        new_instance["preannotation_record_id"] = record["record_id"]
        self.left_label.plants.append(new_instance)
        self.preannotation_adjustment_records.append(record)

        self._clear_preannotation_candidate()
        self.left_label.select_entity("formal", instance_id)
        self.preannotation_pending_fine_tune_entries.add(int(instance_id))
        self.left_label.enter_fine_tune_mode(instance_id)
        self.btn_fine_tune.setText("退出微调模式")
        self.btn_add_vertex.setEnabled(True)
        if hasattr(self, "btn_delete_vertex"):
            self.btn_delete_vertex.setEnabled(True)
        if hasattr(self, "btn_brush_vertex"):
            self.btn_brush_vertex.setEnabled(True)
        if hasattr(self, "btn_brush_delete"):
            self.btn_brush_delete.setEnabled(True)
        self.left_label.removing_region = False
        if hasattr(self, "sync_interaction_state"):
            self.sync_interaction_state()
        self.left_label.current_removal_points = []

        self.mark_annotation_changed()
        self.sync_summary_view()
        self.update_plant_list()
        self.update_undo_redo_state()
        self._persist_preannotation_adjustment_records()
        self.sam_info_text.append(f"已接受预标注: instance={instance_id}")
        self._update_preannotation_controls()

    def save_selected_staging_areas(self):
        candidate = self._get_selected_candidate()
        if not candidate:
            QMessageBox.warning(self, "警告", "当前没有可拒绝的预标注候选")
            return

        candidate_id = candidate.get("candidate_id")
        self._clear_preannotation_candidate()
        self.sam_info_text.append(f"已拒绝候选: {candidate_id}")
        self._update_preannotation_controls()

    def on_fine_tune_session_started(self, instance_id):
        instance_id = int(instance_id or 0)
        if instance_id not in self.preannotation_pending_fine_tune_entries:
            return
        self.preannotation_pending_fine_tune_entries.discard(instance_id)
        plant = self._find_plant_by_id(instance_id)
        if not plant:
            return
        record_id = plant.get("preannotation_record_id")
        record = self._find_preannotation_record(record_id)
        if not record:
            return
        self.preannotation_fine_tune_sessions[instance_id] = {
            "record_id": record_id,
            "event_log_len": len(record.get("event_log", [])),
            "final_polygons": copy.deepcopy(record.get("final_polygons", [])),
            "final_labels": copy.deepcopy(record.get("final_labels", [])),
            "status": record.get("status"),
            "reason_codes": copy.deepcopy(record.get("reason_codes", [])),
            "active_reason_code": record.get("active_reason_code"),
            "reason_segments": copy.deepcopy(record.get("reason_segments", [])),
            "active_reason_segment_index": record.get("active_reason_segment_index"),
        }

    def on_fine_tune_session_finished(self, instance_id, saved):
        instance_id = int(instance_id or 0)
        self.preannotation_pending_fine_tune_entries.discard(instance_id)
        session = self.preannotation_fine_tune_sessions.pop(instance_id, None)
        if not session:
            return
        record = self._find_preannotation_record(session.get("record_id"))
        if not record:
            return
        if saved:
            plant = self._find_plant_by_id(instance_id)
            if plant:
                self._set_record_final_state(record, plant.get("polygons", []), plant.get("labels", []))
            self._close_reason_segment(record)
            self._finalize_preannotation_record(record)
        else:
            record["event_log"] = record.get("event_log", [])[:session["event_log_len"]]
            record["operations"] = copy.deepcopy(record["event_log"])
            record["final_polygons"] = copy.deepcopy(session["final_polygons"])
            record["final_labels"] = copy.deepcopy(session["final_labels"])
            record["status"] = session.get("status")
            record["reason_codes"] = copy.deepcopy(session.get("reason_codes", []))
            record["active_reason_code"] = session.get("active_reason_code")
            record["reason_segments"] = copy.deepcopy(session.get("reason_segments", []))
            record["active_reason_segment_index"] = session.get("active_reason_segment_index")
        normalized_record = normalize_record(record)
        record.clear()
        record.update(normalized_record)
        self._persist_preannotation_adjustment_records()

    def record_preannotation_adjustment_action(self, instance_id, action_type, details):
        instance_id = int(instance_id or 0)
        if instance_id not in self.preannotation_fine_tune_sessions:
            return
        plant = self._find_plant_by_id(instance_id)
        if not plant:
            return
        record_id = plant.get("preannotation_record_id")
        record = self._find_preannotation_record(record_id)
        if not record:
            return
        active_reason = record.get("active_reason_code") or self._get_current_preannotation_reason_code()
        if not active_reason:
            QMessageBox.warning(self, "警告", "请先选择理由，再进行预标注微调操作")
            return
        if record.get("active_reason_code") != active_reason:
            previous_reason = record.get("active_reason_code")
            if previous_reason != active_reason:
                self._close_reason_segment(record)
            set_active_reason(record, active_reason)
            append_event(
                record,
                "reason_selected",
                {
                    "previous_reason_code": previous_reason,
                    "reason_code": record.get("active_reason_code"),
                },
                reason_code=record.get("active_reason_code"),
            )
        before_state = self._get_record_final_state_snapshot(record)
        after_state = self._get_plant_state_snapshot(plant)
        self._append_reasoned_adjustment(
            record,
            action_type,
            copy.deepcopy(details),
            before_state=before_state,
            after_state=after_state,
        )
        self._finalize_preannotation_record(record)
        self._persist_preannotation_adjustment_records()

    def on_entity_geometry_modified(self):
        if self.left_label.mode != "fine_tune":
            return
        instance_id = int(self.left_label.fine_tune_instance_id or 0)
        if instance_id not in self.preannotation_fine_tune_sessions:
            return
        plant = self._find_plant_by_id(instance_id)
        if not plant:
            return
        record_id = plant.get("preannotation_record_id")
        record = self._find_preannotation_record(record_id)
        if record:
            self._set_record_final_state(record, plant.get("polygons", []), plant.get("labels", []))
            sync_active_reason_segment(
                record,
                polygons=record.get("final_polygons", []),
                labels=record.get("final_labels", []),
            )
            self._finalize_preannotation_record(record)

    def on_imported_preannotation_record_changed(self, _index):
        self._refresh_imported_reason_segment_selector()
        reason_segment = self._get_selected_reason_segment()
        if reason_segment:
            self._append_reason_segment_summary(reason_segment)
            self._set_reason_segment_summary_label(reason_segment)
            self.preview_selected_reason_segment_end()
        else:
            self._set_reason_segment_summary_label(None)
        self._update_preannotation_controls()

    def on_imported_reason_segment_changed(self, _index):
        reason_segment = self._get_selected_reason_segment()
        if reason_segment:
            self._append_reason_segment_summary(reason_segment)
            self._set_reason_segment_summary_label(reason_segment)
            self.preview_selected_reason_segment_end()
        else:
            self._set_reason_segment_summary_label(None)
        self._update_preannotation_controls()

    def preview_selected_reason_segment_start(self):
        reason_segment = self._get_selected_reason_segment()
        if not reason_segment:
            QMessageBox.warning(self, "警告", "请先选择一个理由片段")
            return
        self._show_reason_segment_preview(
            reason_segment.get("start_polygons") or [],
            reason_segment.get("reason_code"),
            "before",
        )
        self._update_preannotation_controls()

    def preview_selected_reason_segment_end(self):
        reason_segment = self._get_selected_reason_segment()
        if not reason_segment:
            QMessageBox.warning(self, "警告", "请先选择一个理由片段")
            return
        self._show_reason_segment_preview(
            reason_segment.get("end_polygons") or [],
            reason_segment.get("reason_code"),
            "after",
        )
        self._update_preannotation_controls()

    def clear_imported_reason_segment_preview(self):
        if not self._has_imported_reason_preview():
            return
        self.left_label.candidate_instances = []
        if self.left_label.selected_entity_kind == "candidate":
            self.left_label.selected_entity_kind = None
            self.left_label.selected_entity_id = None
        self.left_label.update_display()
        self.sam_info_text.append("已清除理由片段预览")
        self._update_preannotation_controls()

    def import_preannotation_adjustments(self):
        directory = QFileDialog.getExistingDirectory(self, "选择预标注记录目录", self.import_path or self.export_path or "")
        if not directory:
            return
        self.import_path = directory
        imported_files = 0
        imported_records = 0
        skipped_files = 0
        for entry in sorted(os.listdir(directory)):
            if not entry.startswith("correction_") or not entry.lower().endswith(".json"):
                continue
            path = os.path.join(directory, entry)
            if not os.path.isfile(path):
                continue
            fallback_name = self._extract_image_name_from_correction_filename(entry)
            target_image = self._resolve_image_path_by_name(fallback_name)
            payload_image_path = None
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    if isinstance(payload, dict):
                        payload_image_path = payload.get("image_path")
            except Exception:
                payload_image_path = None
            if not target_image and payload_image_path:
                target_image = self._resolve_image_path_by_name(payload_image_path)
            if not target_image:
                skipped_files += 1
                continue
            records = load_records_from_file(path, image_path=target_image)
            if not records:
                skipped_files += 1
                continue
            cached_records = self.preannotation_records_by_image.get(target_image, [])
            merged_records = {str(item.get("record_id") or ""): copy.deepcopy(item) for item in cached_records}
            for record in records:
                merged_records[str(record.get("record_id") or "")] = copy.deepcopy(record)
            self.preannotation_records_by_image[target_image] = list(merged_records.values())
            imported_files += 1
            imported_records += len(records)
        if self.current_image_path:
            self._load_preannotation_adjustment_records(self.current_image_path)
        self._update_preannotation_controls()
        QMessageBox.information(
            self,
            "导入完成",
            f"成功导入 {imported_files} 个文件，共 {imported_records} 条记录，跳过 {skipped_files} 个文件",
        )

    def apply_selected_imported_preannotation_record(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        if self._has_imported_reason_preview():
            self.clear_imported_reason_segment_preview()
        record = self._get_selected_imported_preannotation_record()
        if not record:
            QMessageBox.warning(self, "警告", "请先选择一条已导入记录")
            return
        final_state = self._get_record_final_state_snapshot(record)
        if not final_state["polygons"]:
            QMessageBox.warning(self, "警告", "该记录没有可恢复的 final 多边形")
            return
        record_id = record.get("record_id")
        formal_instance_id = record.get("formal_instance_id")
        target_plant = None
        if formal_instance_id is not None:
            target_plant = self._find_plant_by_id(formal_instance_id)
        if not target_plant:
            for plant in self.left_label.plants:
                if plant.get("preannotation_record_id") == record_id:
                    target_plant = plant
                    break
        if target_plant:
            target_plant["polygons"] = copy.deepcopy(final_state["polygons"])
            target_plant["labels"] = copy.deepcopy(final_state["labels"])
            target_plant["source"] = "ai_imported"
            target_plant["origin_model_version"] = record.get("model_type")
            target_plant["origin_confidence"] = record.get("confidence")
            target_plant["preannotation_record_id"] = record_id
            instance_id = int(target_plant.get("id", 0))
        else:
            instance_id = None
            if formal_instance_id is not None and not self._find_plant_by_id(formal_instance_id):
                instance_id = int(formal_instance_id)
            if instance_id is None or instance_id <= 0:
                instance_id = int(self.left_label.current_plant_id)
            new_instance = make_formal_instance(
                instance_id=instance_id,
                polygons=copy.deepcopy(final_state["polygons"]),
                source="ai_imported",
                origin_model_version=record.get("model_type"),
                origin_confidence=record.get("confidence"),
            )
            new_instance["labels"] = copy.deepcopy(final_state["labels"])
            new_instance["preannotation_record_id"] = record_id
            self.left_label.plants.append(new_instance)
        self.left_label.current_plant_id = max(int(self.left_label.current_plant_id), int(instance_id) + 1)
        append_event(
            record,
            "imported_correction_applied",
            {
                "instance_id": int(instance_id),
                "image_path": self.current_image_path,
            },
            reason_code=record.get("active_reason_code"),
        )
        self._persist_preannotation_adjustment_records()
        self.left_label.select_entity("formal", int(instance_id))
        self.mark_annotation_changed()
        self.sync_summary_view()
        self.update_plant_list()
        self.update_undo_redo_state()
        self._update_preannotation_controls()
        self.sam_info_text.append(f"已恢复导入记录: {record_id} -> instance={instance_id}")
        QMessageBox.information(self, "恢复完成", f"已将记录 {record_id} 的 final 结果恢复到实例 {instance_id}")

    def export_preannotation_adjustments(self):
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录", self.export_path or "")
        if not directory:
            return

        self.export_path = directory
        self._persist_preannotation_adjustment_records()
        exported_count = 0
        skipped_count = 0

        for image_path in self.image_paths:
            annotation = self.coco_container.get(image_path)
            completed = False
            if annotation:
                completed = bool(annotation.get("image_state", {}).get("annotation_completed", False))
            if image_path == self.current_image_path:
                completed = bool((self.current_image_state or {}).get("annotation_completed", False))
            if not completed:
                skipped_count += 1
                continue

            records = self._get_cached_preannotation_records(image_path)
            if not records:
                correction_path = self._get_existing_correction_path(image_path)
                if correction_path and os.path.exists(correction_path):
                    records = load_records_from_file(correction_path, image_path=image_path)
                else:
                    skipped_count += 1
                    continue

            export_filename = self._get_correction_filename(image_path)
            export_path = os.path.join(directory, export_filename)
            save_records_to_file(export_path, image_path, records)
            exported_count += 1

        QMessageBox.information(self, "导出完成", f"成功导出 {exported_count} 个文件，跳过 {skipped_count} 个文件")

    def _ensure_sam_model_loaded_interactive(self, prompt_message):
        if self.sam_manager.has_model_loaded():
            return True
        try:
            return self._prompt_load_sam_model(prompt_message=prompt_message)
        except Exception as error:
            self.sam_info_text.append(f"加载模型失败: {str(error)}")
            QMessageBox.warning(self, "加载失败", f"加载 SAM 模型失败: {str(error)}")
            return False

    def _ensure_training_project_context(self):
        if not self.image_paths:
            return None
        class_names = []
        if self.project_metadata:
            class_names = list(self.project_metadata.get("class_names", []) or [])
        elif self.current_image_path in self.coco_container:
            class_names = list(self.coco_container[self.current_image_path].get("class_names", []) or [])
        project_id, metadata, paths = ensure_project_for_images(self.image_paths, class_names=class_names or None)
        self.project_id = project_id
        self.project_metadata = metadata
        self.project_paths = paths
        return paths

    def _get_training_output_root(self):
        if self.project_paths and self.project_paths.get("models_root"):
            return str((Path(self.project_paths["models_root"]) / "sam_training").resolve())
        if self.save_path:
            return str((Path(self.save_path).expanduser().resolve() / "sam_training"))
        if self.image_paths:
            try:
                common_path = os.path.commonpath(self.image_paths)
            except ValueError:
                common_path = os.path.dirname(self.image_paths[0])
            base_dir = Path(common_path)
            if base_dir.is_file():
                base_dir = base_dir.parent
            return str((base_dir / "maize_preseg_artifacts" / "sam_training").resolve())
        return str((Path.cwd() / "maize_preseg_artifacts" / "sam_training").resolve())

    def _get_training_blocker(self):
        if self.left_label.mode == "fine_tune":
            return "请先退出当前微调模式，再开始训练"
        if hasattr(self, "_has_active_preview_session") and self._has_active_preview_session():
            return "请先完成或取消当前继续标注/暂存编辑，再开始训练"
        if self.left_label.preannotation_box_mode:
            return "请先完成或取消当前框选预标注，再开始训练"
        if self.left_label.candidate_instances:
            return "请先接受或拒绝当前预标注候选，再开始训练"
        return None

    def _mark_training_snapshot_clean(self, snapshot_hashes):
        for image_path, trained_hash in (snapshot_hashes or {}).items():
            annotation = self.coco_container.get(image_path)
            if not annotation:
                continue
            image_state = annotation.setdefault("image_state", {})
            current_hash = annotation.get("annotation_hash") or compute_annotation_hash(
                annotation.get("plants", []),
                image_state,
            )
            annotation["annotation_hash"] = current_hash
            if current_hash != trained_hash:
                continue
            image_state["last_trained_seen_hash"] = trained_hash
            image_state["dirty_since_last_train"] = False
            if image_path == self.current_image_path and self.current_image_state is not image_state:
                self.current_image_state["last_trained_seen_hash"] = trained_hash
                self.current_image_state["dirty_since_last_train"] = False
        if hasattr(self, "update_status_bar"):
            self.update_status_bar()

    def _handle_sam_training_finished(self, success, message, best_model_path):
        self.btn_load_sam.setEnabled(True)
        self.btn_sam_train.setEnabled(True)
        self.btn_sam_train.setText("开始训练")
        self.btn_sam_preannotate.setEnabled(True)

        run_info = dict(getattr(self.sam_training_manager, "last_run_info", {}) or {})
        best_model_path = best_model_path or run_info.get("best_model_path", "")

        if success:
            snapshot_hashes = run_info.get("snapshot_hashes", {})
            if self.project_id and best_model_path:
                version_name = Path(best_model_path).parent.name
                try:
                    self.project_metadata = mark_training_success(self.project_id, version_name, snapshot_hashes)
                except Exception as error:
                    self.sam_info_text.append(f"训练状态同步失败: {error}")
            self._mark_training_snapshot_clean(snapshot_hashes)

            validation_output_dir = run_info.get("validation_output_dir", "")
            run_dir = run_info.get("run_dir", "")
            if run_dir:
                self.sam_info_text.append(f"训练输出目录: {run_dir}")
            if best_model_path:
                self.sam_info_text.append(f"最佳模型: {best_model_path}")
            if validation_output_dir:
                self.sam_info_text.append(f"验证可视化: {validation_output_dir}")
            self.sam_info_text.append("训练完成")

            detail_lines = [message]
            if best_model_path:
                detail_lines.append(f"最佳模型已保存到:\n{best_model_path}")
            if validation_output_dir:
                detail_lines.append(f"验证可视化已保存到:\n{validation_output_dir}")
            QMessageBox.information(self, "训练完成", "\n\n".join(detail_lines))
            return

        if self.project_id:
            try:
                self.project_metadata = mark_training_failed(self.project_id, message)
            except Exception as error:
                self.sam_info_text.append(f"训练失败状态同步失败: {error}")
        self.sam_info_text.append(f"训练失败: {message}")
        QMessageBox.warning(self, "训练失败", message)

    def load_sam_model(self):
        loaded = self._ensure_sam_model_loaded_interactive("请选择一个 SAM 模型文件")
        if loaded:
            QMessageBox.information(self, "成功", f"SAM 模型加载成功 (类型: {self.sam_manager.model_type})")

    def start_sam_training(self):
        if self.sam_training_worker and self.sam_training_worker.isRunning():
            QMessageBox.information(self, "提示", "SAM 训练正在进行中")
            return

        if not self._ensure_sam_model_loaded_interactive("开始训练前需要先加载 SAM 模型"):
            return

        blocker = self._get_training_blocker()
        if blocker:
            QMessageBox.warning(self, "警告", blocker)
            return

        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        self._ensure_training_project_context()

        completed_count = sum(
            1
            for ann in self.coco_container.values()
            if ann.get("image_state", {}).get("annotation_completed", False)
        )
        self.sam_info_text.append(f"已完成图片数量: {completed_count}")
        if completed_count <= 0:
            QMessageBox.warning(self, "警告", "当前项目还没有已完成图片，无法开始训练")
            return

        output_dir = self._get_training_output_root()
        checkpoint_path = self.sam_manager.model_path if self.sam_manager.has_model_loaded() else None
        self.sam_info_text.append(f"训练产物将保存到: {output_dir}")
        if self.project_id:
            try:
                self.project_metadata = mark_training_started(self.project_id, f"训练目录: {output_dir}")
            except Exception as error:
                self.sam_info_text.append(f"训练开始状态同步失败: {error}")

        self.btn_load_sam.setEnabled(False)
        self.btn_sam_train.setEnabled(False)
        self.btn_sam_train.setText("训练中...")
        self.btn_sam_preannotate.setEnabled(False)

        self.sam_training_worker = SamTrainingWorker(
            self.sam_training_manager,
            self.coco_container,
            self.image_paths,
            train_kwargs={
                "output_dir": output_dir,
                "checkpoint_path": checkpoint_path,
            },
        )
        self.sam_training_worker.finished_signal.connect(self._handle_sam_training_finished)
        self.sam_training_worker.finished.connect(self._cleanup_sam_training_worker)
        self.sam_training_worker.start()
