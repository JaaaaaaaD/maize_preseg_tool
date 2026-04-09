import copy
import os
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from utils.annotation_schema import compute_annotation_hash, current_timestamp, make_formal_instance
from utils.helpers import calculate_polygon_area
from utils.preannotation_records import (
    SEMANTIC_ACTION_AUTO,
    append_event,
    load_records_from_file,
    next_record_counter,
    normalize_record,
    save_records_to_file,
    set_active_reason,
    set_semantic_action,
    set_status,
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

    def _load_preannotation_adjustment_records(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            self.preannotation_adjustment_records = []
            self.preannotation_record_counter = 1
            self._sync_preannotation_judgement_ui()
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
        self._sync_preannotation_judgement_ui()
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

    def _get_current_preannotation_semantic_action(self):
        if hasattr(self, "combo_preannotation_semantic_action"):
            return self.combo_preannotation_semantic_action.currentData() or SEMANTIC_ACTION_AUTO
        return self.preannotation_default_semantic_action or SEMANTIC_ACTION_AUTO

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
        return self._get_active_preannotation_context().get("record")

    def _sync_preannotation_judgement_ui(self):
        if getattr(self, "_updating_preannotation_controls", False):
            return
        self._updating_preannotation_controls = True
        try:
            record = self._get_active_preannotation_record()
            reason_code = self.preannotation_default_reason_code
            semantic_action = self.preannotation_default_semantic_action or SEMANTIC_ACTION_AUTO
            if record:
                reason_code = record.get("active_reason_code")
                if record.get("semantic_action_source") == "manual":
                    semantic_action = record.get("semantic_action") or SEMANTIC_ACTION_AUTO
                else:
                    semantic_action = SEMANTIC_ACTION_AUTO
            if hasattr(self, "combo_preannotation_reason"):
                self._set_combobox_data(self.combo_preannotation_reason, reason_code, default_index=0)
            if hasattr(self, "combo_preannotation_semantic_action"):
                self._set_combobox_data(self.combo_preannotation_semantic_action, semantic_action, default_index=0)
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
        set_active_reason(record, reason_code)
        if previous_reason != record.get("active_reason_code"):
            append_event(
                record,
                "reason_selected",
                {"reason_code": record.get("active_reason_code")},
                reason_code=record.get("active_reason_code"),
            )
            self._persist_preannotation_adjustment_records()

    def on_preannotation_semantic_action_changed(self, _index):
        if getattr(self, "_updating_preannotation_controls", False):
            return
        semantic_action = self._get_current_preannotation_semantic_action()
        self.preannotation_default_semantic_action = semantic_action
        record = self._get_active_preannotation_record()
        if not record:
            return
        previous_action = record.get("semantic_action")
        previous_source = record.get("semantic_action_source")
        if semantic_action == SEMANTIC_ACTION_AUTO:
            set_semantic_action(record, None, source="auto")
        else:
            set_semantic_action(record, semantic_action, source="manual")
        if (
            previous_action != record.get("semantic_action")
            or previous_source != record.get("semantic_action_source")
        ):
            append_event(
                record,
                "semantic_action_selected",
                {"semantic_action": semantic_action},
                reason_code=record.get("active_reason_code"),
                semantic_action=None if semantic_action == SEMANTIC_ACTION_AUTO else semantic_action,
            )
            self._persist_preannotation_adjustment_records()

    @staticmethod
    def _extract_outer_polygons(polygons):
        outer_polygons = []
        for polygon in polygons or []:
            if calculate_polygon_area(polygon) < 0:
                outer_polygons.append(copy.deepcopy(polygon))
        if outer_polygons:
            return outer_polygons
        return copy.deepcopy(polygons or [])

    def _append_preannotation_event(self, record, event_type, details=None, semantic_action=None):
        append_event(
            record,
            event_type,
            details=details,
            reason_code=record.get("active_reason_code"),
            semantic_action=semantic_action,
        )

    def _finalize_preannotation_record(self, record):
        if record.get("semantic_action_source") != "manual":
            set_semantic_action(record, None, source="auto")
        semantic_action = record.get("semantic_action")
        if semantic_action == "ignore":
            set_status(record, "ignored")
        elif semantic_action == "reject":
            set_status(record, "rejected")
        elif semantic_action == "merge":
            set_status(record, "merged")
        elif record.get("event_log"):
            set_status(record, "modified")
        else:
            set_status(record, "accepted")

    def _build_preannotation_record(self, candidate, formal_instance_id=None, status="accepted"):
        semantic_action = self.preannotation_default_semantic_action or SEMANTIC_ACTION_AUTO
        if status == "accepted" and semantic_action in ("ignore", "reject"):
            semantic_action = SEMANTIC_ACTION_AUTO
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
                "original_polygons": copy.deepcopy(candidate.get("polygons", [])),
                "final_polygons": copy.deepcopy(candidate.get("polygons", [])),
                "formal_instance_id": formal_instance_id,
                "status": status,
                "event_log": [],
            }
        )
        set_active_reason(record, self.preannotation_default_reason_code)
        if semantic_action == SEMANTIC_ACTION_AUTO:
            set_semantic_action(record, None, source="auto")
        else:
            set_semantic_action(record, semantic_action, source="manual")
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
            record = self._build_preannotation_record(candidate, formal_instance_id=None, status="ignored")
            set_semantic_action(record, "ignore", source="manual")
            set_status(record, "ignored")
            self._append_preannotation_event(
                record,
                "candidate_ignored",
                {"candidate_id": candidate.get("candidate_id")},
                semantic_action="ignore",
            )
            self.preannotation_adjustment_records.append(record)
            self.left_label.ignored_regions.extend(self._extract_outer_polygons(candidate.get("polygons", [])))
            self._clear_preannotation_candidate()
            self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_undo_redo_state()
            self._persist_preannotation_adjustment_records()
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
        record = self._find_preannotation_record(plant.get("preannotation_record_id"))
        if not record:
            return
        record["final_polygons"] = copy.deepcopy(plant.get("polygons", []))
        set_semantic_action(record, "ignore", source="manual")
        set_status(record, "ignored")
        self._append_preannotation_event(
            record,
            "instance_ignored",
            {"formal_instance_id": instance_id},
            semantic_action="ignore",
        )
        self.left_label.ignored_regions.extend(self._extract_outer_polygons(plant.get("polygons", [])))
        self._remove_preannotation_formal_instance(instance_id)
        self.mark_annotation_changed()
        self.sync_summary_view()
        self.update_plant_list()
        self.update_undo_redo_state()
        self._persist_preannotation_adjustment_records()
        self._update_preannotation_controls()
        self.sam_info_text.append(f"已忽略 proposal instance={instance_id}")

    def record_preannotation_instance_deleted(self, plant, semantic_action=None):
        if not plant or not plant.get("preannotation_record_id"):
            return False
        record = self._find_preannotation_record(plant.get("preannotation_record_id"))
        if not record:
            return False

        chosen_action = semantic_action or self._get_current_preannotation_semantic_action()
        if chosen_action != "merge":
            chosen_action = "reject"
        record["formal_instance_id"] = int(plant.get("id", 0))
        record["final_polygons"] = []
        set_semantic_action(record, chosen_action, source="manual")
        set_status(record, "merged" if chosen_action == "merge" else "rejected")
        self._append_preannotation_event(
            record,
            "proposal_merged" if chosen_action == "merge" else "delete_instance",
            {"formal_instance_id": int(plant.get("id", 0))},
            semantic_action=chosen_action,
        )
        self._persist_preannotation_adjustment_records()
        self._update_preannotation_controls()
        return True

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
        if hasattr(self, "combo_preannotation_semantic_action"):
            self.combo_preannotation_semantic_action.setEnabled(has_image)
        self._sync_preannotation_judgement_ui()

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
        self.left_label.enter_fine_tune_mode(instance_id)
        self.btn_fine_tune.setText("退出微调模式")
        self.btn_add_vertex.setEnabled(True)
        if hasattr(self, "btn_delete_vertex"):
            self.btn_delete_vertex.setEnabled(True)
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

        record = self._build_preannotation_record(candidate, formal_instance_id=None, status="rejected")
        set_semantic_action(record, "reject", source="manual")
        set_status(record, "rejected")
        self._append_preannotation_event(
            record,
            "candidate_rejected",
            {"candidate_id": candidate.get("candidate_id")},
            semantic_action="reject",
        )
        self.preannotation_adjustment_records.append(record)

        candidate_id = candidate.get("candidate_id")
        self._clear_preannotation_candidate()
        self._persist_preannotation_adjustment_records()
        self.sam_info_text.append(f"已拒绝候选: {candidate_id}")
        self._update_preannotation_controls()

    def on_fine_tune_session_started(self, instance_id):
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
            "status": record.get("status"),
            "semantic_action": record.get("semantic_action"),
            "semantic_action_source": record.get("semantic_action_source"),
            "reason_codes": copy.deepcopy(record.get("reason_codes", [])),
            "active_reason_code": record.get("active_reason_code"),
        }

    def on_fine_tune_session_finished(self, instance_id, saved):
        session = self.preannotation_fine_tune_sessions.pop(instance_id, None)
        if not session:
            return
        record = self._find_preannotation_record(session.get("record_id"))
        if not record:
            return
        if saved:
            plant = self._find_plant_by_id(instance_id)
            if plant:
                record["final_polygons"] = copy.deepcopy(plant.get("polygons", []))
            self._finalize_preannotation_record(record)
        else:
            record["event_log"] = record.get("event_log", [])[:session["event_log_len"]]
            record["operations"] = copy.deepcopy(record["event_log"])
            record["final_polygons"] = copy.deepcopy(session["final_polygons"])
            record["status"] = session.get("status")
            record["semantic_action"] = session.get("semantic_action")
            record["semantic_action_source"] = session.get("semantic_action_source")
            record["reason_codes"] = copy.deepcopy(session.get("reason_codes", []))
            record["active_reason_code"] = session.get("active_reason_code")
        self._persist_preannotation_adjustment_records()

    def record_preannotation_adjustment_action(self, instance_id, action_type, details):
        plant = self._find_plant_by_id(instance_id)
        if not plant:
            return
        record_id = plant.get("preannotation_record_id")
        record = self._find_preannotation_record(record_id)
        if not record:
            return
        self._append_preannotation_event(record, action_type, copy.deepcopy(details))
        record["final_polygons"] = copy.deepcopy(plant.get("polygons", []))
        self._finalize_preannotation_record(record)

    def on_entity_geometry_modified(self):
        if self.left_label.mode != "fine_tune":
            return
        instance_id = self.left_label.fine_tune_instance_id
        plant = self._find_plant_by_id(instance_id)
        if not plant:
            return
        record_id = plant.get("preannotation_record_id")
        record = self._find_preannotation_record(record_id)
        if record:
            record["final_polygons"] = copy.deepcopy(plant.get("polygons", []))
            self._finalize_preannotation_record(record)

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
