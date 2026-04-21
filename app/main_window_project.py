import copy
import os
import traceback

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from config import SHORTCUTS
from utils.annotation_schema import current_timestamp, make_image_state, normalize_image_state
from utils.data_manager import load_annotation_from_coco
from utils.helpers import load_image
from utils.image_processor import preprocess_image
from utils.project_context import ensure_project_for_images
from utils.session_cache import (
    get_autosave_root,
    load_last_app_session,
    load_image_draft,
    load_session,
    save_last_app_session,
    save_image_draft,
    save_session,
)


class MainWindowProjectMixin:
    def _notify_restore_prompt(self):
        QApplication.beep()
        try:
            QApplication.alert(self, 6000)
        except Exception:
            pass
        try:
            if os.name == "nt":
                import ctypes

                class FLASHWINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", ctypes.c_uint),
                        ("hwnd", ctypes.c_void_p),
                        ("dwFlags", ctypes.c_uint),
                        ("uCount", ctypes.c_uint),
                        ("dwTimeout", ctypes.c_uint),
                    ]

                hwnd = int(self.winId())
                flash_info = FLASHWINFO(
                    ctypes.sizeof(FLASHWINFO),
                    hwnd,
                    0x0000000C,  # FLASHW_TRAY | FLASHW_TIMERNOFG
                    3,
                    0,
                )
                ctypes.windll.user32.FlashWindowEx(ctypes.byref(flash_info))
        except Exception:
            pass

    def _ask_restore_session(self, message):
        self._notify_restore_prompt()
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle("恢复上次会话")
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dialog.setDefaultButton(QMessageBox.Yes)
        if not self.windowIcon().isNull():
            dialog.setWindowIcon(self.windowIcon())
        return dialog.exec_() == QMessageBox.Yes

    @staticmethod
    def _normalize_paths_for_compare(paths):
        return [os.path.normcase(os.path.abspath(path)) for path in (paths or [])]

    @staticmethod
    def _filter_existing_images(paths):
        existing = []
        for path in paths or []:
            abs_path = os.path.abspath(path)
            if os.path.isfile(abs_path):
                existing.append(abs_path)
        return existing

    @staticmethod
    def _get_app_session_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _build_last_app_session_payload(self):
        return {
            "image_paths": [os.path.abspath(path) for path in (self.image_paths or [])],
            "current_image_index": int(self.current_image_index if self.current_image_index >= 0 else 0),
            "import_path": self.import_path or "",
            "save_path": self.save_path or "",
            "export_path": self.export_path or "",
            "sam_model_path": self.sam_manager.model_path or "",
            "sam_model_type": self.sam_manager.model_type or "",
            "autosave_root": self._get_autosave_root() or "",
        }

    def _ensure_project_context_for_images(self, image_paths):
        if not image_paths:
            return None
        normalized_paths = tuple(os.path.abspath(path) for path in image_paths)
        if getattr(self, "_project_context_key", None) == normalized_paths and self.project_paths:
            return self.project_paths
        class_names = []
        if self.project_metadata:
            class_names = list(self.project_metadata.get("class_names", []) or [])
        elif self.current_image_path in self.coco_container:
            class_names = list(self.coco_container[self.current_image_path].get("class_names", []) or [])
        project_id, metadata, paths = ensure_project_for_images(image_paths, class_names=class_names or None)
        self.project_id = project_id
        self.project_metadata = metadata
        self.project_paths = paths
        self._project_context_key = normalized_paths
        return paths

    def _get_autosave_root(self):
        project_root = (self.project_paths or {}).get("project_root")
        if not project_root:
            return None
        return get_autosave_root(project_root)

    def _build_current_image_draft_payload(self, image_path):
        annotation = self.coco_container.get(image_path)
        if image_path == self.current_image_path:
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

        if not annotation:
            annotation = {
                "plants": [],
                "current_plant_id": 1,
                "ignored_regions": [],
                "image_state": make_image_state(image_path),
            }

        return {
            "plants": copy.deepcopy(annotation.get("plants", [])),
            "current_plant_id": int(annotation.get("current_plant_id", 1) or 1),
            "ignored_regions": copy.deepcopy(annotation.get("ignored_regions", [])),
            "image_state": copy.deepcopy(
                normalize_image_state(image_path, annotation.get("image_state", make_image_state(image_path)))
            ),
            "preannotation_records": copy.deepcopy((self.preannotation_records_by_image or {}).get(image_path, [])),
        }

    def _persist_current_image_draft(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image or not self.image_paths:
            return
        self._ensure_project_context_for_images(self.image_paths)
        autosave_root = self._get_autosave_root()
        if not autosave_root:
            return
        payload = self._build_current_image_draft_payload(target_image)
        save_image_draft(autosave_root, target_image, payload)

    def _persist_session_cache(self):
        if not self.image_paths:
            return
        self._ensure_project_context_for_images(self.image_paths)
        autosave_root = self._get_autosave_root()
        if not autosave_root:
            return
        save_session(
            autosave_root,
            self.image_paths,
            self.current_image_index if self.current_image_index >= 0 else 0,
            import_path=self.import_path,
        )
        save_last_app_session(self._get_app_session_root(), self._build_last_app_session_payload())

    def _schedule_autosave(self):
        if not self.current_image_path:
            return
        if hasattr(self, "autosave_timer"):
            self.autosave_timer.start()

    def _flush_autosave(self):
        self._persist_current_image_draft(self.current_image_path)
        self._persist_session_cache()

    def _try_restore_cached_session(self, ask_user=True, autosave_root_override=None):
        if not self.image_paths:
            return False, 0, 0
        self._ensure_project_context_for_images(self.image_paths)
        autosave_root = autosave_root_override or self._get_autosave_root()
        if not autosave_root:
            return False, 0, 0

        session_payload = load_session(autosave_root)
        if not isinstance(session_payload, dict):
            return False, 0, 0

        cached_paths = self._normalize_paths_for_compare(session_payload.get("image_paths") or [])
        current_paths = self._normalize_paths_for_compare(self.image_paths)
        if cached_paths != current_paths:
            return False, 0, 0

        if ask_user:
            accepted = self._ask_restore_session("检测到未完成会话，是否恢复上次退出前的图片位置和标注草稿？")
            if not accepted:
                return False, 0, 0

        restored_count = 0
        for image_path in self.image_paths:
            draft_payload = load_image_draft(autosave_root, image_path)
            if not isinstance(draft_payload, dict):
                continue
            image_state = normalize_image_state(
                image_path,
                draft_payload.get("image_state", make_image_state(image_path)),
            )
            self.coco_container[image_path] = {
                "plants": copy.deepcopy(draft_payload.get("plants", [])),
                "current_plant_id": int(draft_payload.get("current_plant_id", 1) or 1),
                "ignored_regions": copy.deepcopy(draft_payload.get("ignored_regions", [])),
                "image_state": image_state,
            }
            self.preannotation_records_by_image[image_path] = copy.deepcopy(
                draft_payload.get("preannotation_records", [])
            )
            restored_count += 1

        import_path = session_payload.get("import_path")
        if import_path:
            self.import_path = import_path
        restored_index = int(session_payload.get("current_image_index", 0) or 0)
        restored_index = max(0, min(restored_index, len(self.image_paths) - 1))
        return True, restored_index, restored_count

    def _reset_state_for_image_list(self):
        self.image_sequence_map = {path: index for index, path in enumerate(self.image_paths, start=1)}
        self.preprocess_cache.clear()
        self.coco_container.clear()
        if hasattr(self, "pause_annotation_timer"):
            self.pause_annotation_timer()
        self.preannotation_adjustment_records = []
        self.preannotation_records_by_image = {}
        self.preannotation_record_counter = 1
        self.preannotation_fine_tune_sessions = {}
        self.current_image = None
        self.current_image_path = ""
        self.current_image_index = -1
        self.current_image_state = make_image_state("")
        self.current_annotation_hash = None
        self._clear_preannotation_candidate()
        self.btn_prev.setEnabled(len(self.image_paths) > 1)
        self.btn_next.setEnabled(len(self.image_paths) > 1)

    def _restore_sam_from_payload(self, session_payload):
        model_path = str((session_payload or {}).get("sam_model_path") or "").strip()
        model_type = str((session_payload or {}).get("sam_model_type") or "").strip() or "vit_b"
        if not model_path:
            return False, "未记录 SAM 模型路径"
        if not os.path.exists(model_path):
            return False, "SAM 模型文件不存在"
        try:
            self.sam_manager.load_model(model_path, model_type=model_type)
            if hasattr(self, "sam_info_text"):
                self.sam_info_text.append(f"已自动恢复 SAM 模型: {model_type} | {model_path}")
            return True, ""
        except Exception as error:
            if hasattr(self, "sam_info_text"):
                self.sam_info_text.append(f"SAM 自动恢复失败（不影响标注）: {error}")
            return False, str(error)

    def try_restore_last_session_on_startup(self):
        session_payload = load_last_app_session(self._get_app_session_root())
        if not isinstance(session_payload, dict):
            return False
        image_paths = self._filter_existing_images(session_payload.get("image_paths") or [])
        if not image_paths:
            return False

        accepted = self._ask_restore_session("检测到上次会话，是否直接恢复图片、标注进度与当前位置？")
        if not accepted:
            return False

        self.image_paths = image_paths
        self.import_path = session_payload.get("import_path") or self.import_path
        self.save_path = session_payload.get("save_path") or self.save_path
        self.export_path = session_payload.get("export_path") or self.export_path

        self._reset_state_for_image_list()
        self._ensure_project_context_for_images(self.image_paths)
        autosave_root = (session_payload.get("autosave_root") or "").strip() or None
        restored, restored_index, _ = self._try_restore_cached_session(
            ask_user=False,
            autosave_root_override=autosave_root,
        )
        startup_index = int(session_payload.get("current_image_index", 0) or 0)
        startup_index = max(0, min(startup_index, len(self.image_paths) - 1))
        self.goto_image(restored_index if restored else startup_index)
        self._persist_session_cache()
        self._restore_sam_from_payload(session_payload)
        return True

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
        self._schedule_autosave()
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

        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self._reset_state_for_image_list()
        self._ensure_project_context_for_images(self.image_paths)
        restored, restored_index, restored_count = self._try_restore_cached_session()
        self.goto_image(restored_index if restored else 0)
        self._persist_session_cache()

        QMessageBox.information(
            self,
            "加载成功",
            (
                f"成功加载 {len(self.image_paths)} 张图片"
                if not restored
                else f"成功加载 {len(self.image_paths)} 张图片，并恢复 {restored_count} 张图片草稿"
            ),
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
            self._persist_current_image_draft(self.current_image_path)

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
            self._persist_session_cache()
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
            draft_annotation = None
            if self.image_paths:
                self._ensure_project_context_for_images(self.image_paths)
                autosave_root = self._get_autosave_root()
                if autosave_root:
                    draft_payload = load_image_draft(autosave_root, self.current_image_path)
                    if isinstance(draft_payload, dict):
                        draft_annotation = {
                            "plants": copy.deepcopy(draft_payload.get("plants", [])),
                            "current_plant_id": int(draft_payload.get("current_plant_id", 1) or 1),
                            "ignored_regions": copy.deepcopy(draft_payload.get("ignored_regions", [])),
                            "image_state": normalize_image_state(
                                self.current_image_path,
                                draft_payload.get("image_state", make_image_state(self.current_image_path)),
                            ),
                        }
                        self.preannotation_records_by_image[self.current_image_path] = copy.deepcopy(
                            draft_payload.get("preannotation_records", [])
                        )
            if draft_annotation:
                annotation = draft_annotation
                self.coco_container[self.current_image_path] = annotation
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
