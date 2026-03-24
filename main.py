# 主入口文件
import copy
import os
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import ANNOTATION_DIR, SHORTCUTS
from components.help_dialog import HelpDialog
from components.image_label import ImageLabel
from components.toolbars import Toolbars
from services.inference_service import InferenceWorker
from services.training_manager import TrainingManager
from ui.annotation_properties_panel import AnnotationPropertiesPanel
from utils.annotation_schema import current_timestamp, make_image_state
from utils.data_manager import (
    export_coco_format as export_coco_file,
    export_completed_annotations,
    export_simple_json as export_simple_json_file,
    load_current_annotation,
    save_current_annotation,
)
from utils.dataset_builder import build_project_dataset
from utils.helpers import load_image
from utils.image_processor import preprocess_image
from utils.project_context import ensure_project_for_images, load_project, refresh_project_counters, update_image_record

try:
    import torch
    from models.sam_model import SamModel
except ImportError:
    print("Warning: SAM library not found. Please install segment-anything package.")
    torch = None
    SamModel = None

sys.excepthook = lambda exctype, value, tb: traceback.print_exception(exctype, value, tb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玉米标注与项目级预标注闭环工具")
        self.setMinimumSize(1024, 680)

        self.image_paths = []
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.current_image_state = make_image_state("")
        self.current_annotation_hash = None

        self.annotation_dir = ANNOTATION_DIR
        self.preprocess_cache = {}
        self.annotation_changed = False
        self.undo_stack = []
        self.redo_stack = []
        self.is_undo_redo = False

        self.project_id = None
        self.project_metadata = None
        self.project_paths = None

        # self.sam_model = None
        # self.sam_model_loaded = False
        # self.sam_model_path = SAM_MODEL_PATH
        # self.sam_model_type = SAM_MODEL_TYPE
        # self.sam_segmenting = False
        self.region_growing_enabled = False
        self.ignoring_region = False

        self.inference_worker = None
        self.current_inference_request = None
        self.training_manager = TrainingManager(self)
        self.training_status_text = "空闲"

        self.init_ui()
        self.resize_to_available_screen()
        self.init_shortcuts()
        self.bind_training_signals()
        self.restore_button_texts()
        self.restore_button_visuals()

        os.makedirs(self.annotation_dir, exist_ok=True)
        self.update_status_bar()

    def init_ui(self):
        """初始化 UI。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(main_splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.addWidget(Toolbars.create_auxiliary_toolbar(self))
        left_layout.addWidget(Toolbars.create_annotation_toolbar(self))
        left_layout.addWidget(Toolbars.create_plant_management_toolbar(self))
        left_layout.addWidget(Toolbars.create_navigation_toolbar(self))
        left_layout.addWidget(Toolbars.create_progress_label(self))
        left_layout.addStretch()
        left_scroll = self.create_scroll_panel(left_panel, min_width=220, preferred_width=260)
        main_splitter.addWidget(left_scroll)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        images_layout = QHBoxLayout()
        images_layout.setSpacing(10)
        center_layout.addLayout(images_layout)

        self.left_label = ImageLabel(is_summary=False, parent=self)
        self.left_label.setToolTip(
            "左键添加顶点/选择实例 | Enter 暂存区域 | Shift+Enter 保存实例 | 右键拖动 | 滚轮缩放"
        )
        images_layout.addWidget(self.left_label, 1)

        self.right_label = ImageLabel(is_summary=True, parent=self)
        self.right_label.setToolTip("正式实例总览，不显示候选层")
        images_layout.addWidget(self.right_label, 1)
        main_splitter.addWidget(center_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        self.properties_panel = AnnotationPropertiesPanel(self)
        right_layout.addWidget(self.properties_panel, 3)
        right_layout.addWidget(Toolbars.create_file_toolbar(self))
        right_layout.addWidget(Toolbars.create_export_toolbar(self))
        right_layout.addWidget(Toolbars.create_aux_toolbar(self))
        right_layout.addStretch()
        right_scroll = self.create_scroll_panel(right_panel, min_width=300, preferred_width=380)
        main_splitter.addWidget(right_scroll)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([260, 1200, 380])

        self.bind_properties_panel()

    def create_scroll_panel(self, widget, min_width, preferred_width):
        """为侧栏创建滚动容器，避免小分辨率下内容被截断。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(min_width)
        scroll.setMaximumWidth(max(preferred_width + 80, min_width))
        return scroll

    def resize_to_available_screen(self):
        """根据当前屏幕可用区域设置初始窗口大小并居中。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 900)
            return

        geometry = screen.availableGeometry()
        width = max(self.minimumWidth(), min(int(geometry.width() * 0.94), 1920))
        height = max(self.minimumHeight(), min(int(geometry.height() * 0.94), 1080))
        width = min(width, geometry.width())
        height = min(height, geometry.height())
        self.resize(width, height)

        x = geometry.x() + max(0, (geometry.width() - width) // 2)
        y = geometry.y() + max(0, (geometry.height() - height) // 2)
        self.move(x, y)

    def init_shortcuts(self):
        """初始化快捷键。"""
        QShortcut(QKeySequence(SHORTCUTS["SAVE_POLYGON"]), self, self.save_current_polygon)
        QShortcut(QKeySequence(SHORTCUTS["SAVE_PLANT"]), self, self.save_plant)
        QShortcut(QKeySequence(SHORTCUTS["UNDO"]), self, self.undo)
        QShortcut(QKeySequence(SHORTCUTS["DELETE_PLANT"]), self, self.delete_plant)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_EDGE_SNAP"]), self, self.toggle_edge_snap)
        QShortcut(QKeySequence(SHORTCUTS["LOAD_BATCH"]), self, self.load_batch_images)
        QShortcut(QKeySequence(SHORTCUTS["PREV_IMAGE"]), self, self.prev_image)
        QShortcut(QKeySequence(SHORTCUTS["NEXT_IMAGE"]), self, self.next_image)
        # QShortcut(QKeySequence(SHORTCUTS["TOGGLE_SAM_SEGMENTATION"]), self, self.toggle_sam_segmentation)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_REGION_GROWING"]), self, self.toggle_region_growing)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_IGNORE_REGION"]), self, self.toggle_ignore_region)

    def bind_properties_panel(self):
        """连接右侧属性面板事件。"""
        panel = self.properties_panel
        panel.entity_selected.connect(self.on_tree_entity_selected)
        panel.class_changed.connect(self.on_selected_class_changed)
        panel.owner_changed.connect(self.on_selected_owner_changed)
        panel.create_group_requested.connect(self.create_and_assign_plant_group)
        panel.btn_mark_completed.clicked.connect(self.mark_current_image_completed)
        panel.btn_mark_incomplete.clicked.connect(self.mark_current_image_incomplete)
        panel.btn_run_inference.clicked.connect(self.run_current_image_inference)
        panel.btn_accept_candidate.clicked.connect(self.accept_selected_candidate)
        panel.btn_accept_all_candidates.clicked.connect(self.accept_all_candidates)
        panel.btn_clear_candidates.clicked.connect(self.clear_candidates)
        panel.btn_delete_selected.clicked.connect(self.delete_plant)
        panel.btn_manual_train.clicked.connect(self.start_manual_training)
        panel.btn_rollback_model.clicked.connect(self.rollback_active_model)
        panel.btn_rebuild_split.clicked.connect(self.rebuild_validation_split)

    def bind_training_signals(self):
        """连接训练管理器信号。"""
        self.training_manager.training_state_changed.connect(self.on_training_state_changed)
        self.training_manager.training_progress_changed.connect(self.on_training_progress_changed)
        self.training_manager.training_finished.connect(self.on_training_finished)
        self.training_manager.active_model_changed.connect(self.on_active_model_changed)
        self.training_manager.project_counts_changed.connect(self.on_project_counts_changed)

    def configure_project_for_images(self, image_paths):
        """根据当前图片集合切换项目上下文。"""
        if not image_paths:
            return

        self.project_id, self.project_metadata, self.project_paths = ensure_project_for_images(image_paths)
        class_names = self.project_metadata.get("class_names", [])
        self.left_label.set_class_names(class_names)
        self.right_label.set_class_names(class_names)
        self.training_manager.set_project(self.project_id)
        self.refresh_project_metadata()
        self.refresh_properties_panel()

    def refresh_project_metadata(self):
        """从磁盘刷新项目元数据。"""
        if not self.project_id:
            return
        self.project_metadata, _, self.project_paths = load_project(self.project_id)
        self.project_metadata = refresh_project_counters(self.project_id)

    def mark_annotation_changed(self):
        """标记当前图片有未保存修改。"""
        self.annotation_changed = True
        if self.current_image_state:
            self.current_image_state["last_modified_at"] = current_timestamp()
        self.update_status_bar()

    def clear_annotation_changed(self):
        """清除未保存修改标记。"""
        self.annotation_changed = False
        self.update_status_bar()

    def persist_current_if_dirty(self, force=False):
        """在切图或切项目之前保存当前正式标注。"""
        if not self.current_image_path or self.current_image is None:
            return
        if self.annotation_changed or force:
            self.save_current_annotation_auto(force=force)

    def load_batch_images(self):
        """批量加载图片。"""
        self.persist_current_if_dirty()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "批量选择图片（可多选）",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_paths:
            return

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = [path for path in file_paths if os.path.splitext(path.lower())[1] in valid_extensions]
        if not image_paths:
            QMessageBox.warning(self, "警告", "未选择有效的图片文件")
            return

        self.image_paths = image_paths
        self.preprocess_cache.clear()
        self.configure_project_for_images(self.image_paths)
        self.current_image_index = -1
        self.btn_prev.setEnabled(len(self.image_paths) > 1)
        self.btn_next.setEnabled(len(self.image_paths) > 1)
        self.goto_image(0)

        QMessageBox.information(
            self,
            "加载成功",
            f"成功加载 {len(self.image_paths)} 张图片\n项目: {self.project_metadata.get('project_name', self.project_id)}",
        )

    def load_single_image(self):
        """加载单张图片。"""
        self.persist_current_if_dirty()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单张图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        self.image_paths = [file_path]
        self.preprocess_cache.clear()
        self.configure_project_for_images(self.image_paths)
        self.current_image_index = -1
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.goto_image(0)

    def goto_image(self, index):
        """跳转到指定图片。"""
        if index < 0 or index >= len(self.image_paths):
            return

        if self.current_image_index >= 0:
            self.persist_current_if_dirty()

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
            self.load_current_annotation_auto()
            self.update_snap_button_state()
            self.clear_undo_stack()
            self.clear_annotation_changed()
            self.update_plant_list()
            self.sync_summary_view()
            self.refresh_properties_panel()
            self.update_status_bar()
            self.maybe_auto_infer_current_image()
        except Exception as error:
            QMessageBox.critical(self, "错误", f"加载图片失败: {error}")
            traceback.print_exc()

    def prev_image(self):
        if self.current_image_index > 0:
            self.goto_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.goto_image(self.current_image_index + 1)

    def load_current_annotation_auto(self):
        """自动加载当前图片的正式标注。"""
        annotation = load_current_annotation(
            self.current_image_path,
            class_names=self.project_metadata.get("class_names") if self.project_metadata else None,
        )

        if annotation:
            self.left_label.set_class_names(annotation.get("class_names", []))
            self.right_label.set_class_names(annotation.get("class_names", []))
            self.left_label.set_annotation_state(
                annotation["plants"],
                plant_groups=annotation.get("plant_groups"),
                current_plant_id=annotation.get("current_plant_id", 1),
                next_owner_plant_id=annotation.get("next_plant_group_id", 1),
            )
            self.current_image_state = annotation.get("image_state", make_image_state(self.current_image_path))
            self.current_annotation_hash = annotation.get("annotation_hash")
        else:
            self.left_label.set_annotation_state([], plant_groups=[], current_plant_id=1, next_owner_plant_id=1)
            self.current_image_state = make_image_state(self.current_image_path, annotation_completed=False)
            self.current_annotation_hash = None

        self.left_label.clear_candidates()

    def save_current_annotation_auto(self, force=False):
        """自动保存当前正式标注，并同步项目级图片记录。"""
        if not self.current_image_path or not self.project_id:
            return False

        state = self.left_label.get_annotation_state()
        self.current_image_state["image_path"] = self.current_image_path
        self.current_image_state["last_modified_at"] = current_timestamp()

        success, save_path, payload = save_current_annotation(
            self.current_image_path,
            state["plants"],
            state["current_plant_id"],
            plant_groups=state.get("plant_groups"),
            image_state=self.current_image_state,
            project_id=self.project_id,
            class_names=self.project_metadata.get("class_names") if self.project_metadata else None,
            ignored_regions=self.left_label.ignored_regions,
        )
        if not success or not payload:
            return False

        self.current_annotation_hash = payload.get("annotation_hash")
        _, self.project_metadata = update_image_record(
            self.project_id,
            self.current_image_path,
            save_path,
            payload["image_state"],
            payload["annotation_hash"],
        )
        self.current_image_state = payload["image_state"]
        self.clear_annotation_changed()
        self.refresh_project_metadata()
        self.refresh_properties_panel()

        if self.current_image_state.get("annotation_completed"):
            self.training_manager.check_and_trigger_training(force=False)

        return True

    def maybe_auto_infer_current_image(self):
        """在未标注图片上自动触发预标注。"""
        self.start_current_image_inference(manual=False)

    def run_current_image_inference(self):
        """手动对当前图片执行一次 AI 预标注。"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        self.start_current_image_inference(manual=True)

    def start_current_image_inference(self, manual=False):
        """启动当前图片的预标注推理。

        自动模式只对“无正式标注且未完成”的图片触发；
        手动模式允许在当前已有正式标注的图片上重新查看候选层，但不会覆盖正式层。
        """
        self.left_label.clear_candidates()

        if not self.current_image_path:
            self.refresh_properties_panel()
            return False

        if not self.training_manager.has_active_model():
            self.training_status_text = "暂无模型，无法预标注"
            self.refresh_properties_panel()
            self.update_status_bar()
            if manual:
                QMessageBox.warning(self, "预标注", "当前项目暂无 active 模型，无法执行预标注")
            return False

        if not manual:
            if self.left_label.plants:
                self.training_status_text = "当前图片已有正式标注，未自动预标注"
                self.refresh_properties_panel()
                self.update_status_bar()
                return False

            if self.current_image_state.get("annotation_completed"):
                self.training_status_text = "当前图片已完成，未自动预标注"
                self.refresh_properties_panel()
                self.update_status_bar()
                return False

        if self.inference_worker and self.inference_worker.isRunning():
            self.inference_worker.quit()
            self.inference_worker.wait(100)

        self.current_inference_request = {
            "image_path": self.current_image_path,
            "manual": manual,
        }
        self.training_status_text = "正在执行 AI 预标注" if manual else "正在预标注"
        self.refresh_properties_panel()
        self.update_status_bar()

        self.inference_worker = InferenceWorker(
            self.training_manager.get_active_model_path(),
            self.current_image_path,
            self.project_metadata.get("class_names", []),
            model_version=self.training_manager.get_active_model_version(),
        )
        self.inference_worker.inference_succeeded.connect(self.on_inference_succeeded)
        self.inference_worker.inference_failed.connect(self.on_inference_failed)
        self.inference_worker.start()
        return True

    def on_inference_succeeded(self, candidates, image_path):
        """接收后台预标注结果。"""
        if image_path != self.current_image_path:
            return
        request = self.current_inference_request or {}
        manual = bool(request.get("manual"))
        if not manual and (self.left_label.plants or self.current_image_state.get("annotation_completed")):
            self.current_inference_request = None
            return
        self.left_label.set_candidates(candidates)
        if candidates:
            self.left_label.select_entity("candidate", candidates[0]["candidate_id"])
            self.training_status_text = f"预标注完成: {len(candidates)} 个候选"
        else:
            self.training_status_text = "预标注完成: 0 个候选（当前模型未检出）"
        self.refresh_properties_panel()
        self.update_status_bar()
        self.current_inference_request = None
        if manual and not candidates:
            QMessageBox.information(self, "预标注", "当前模型在这张图上未产生候选实例")

    def on_inference_failed(self, message, image_path):
        """预标注失败不影响正式标注。"""
        if image_path != self.current_image_path:
            return
        request = self.current_inference_request or {}
        manual = bool(request.get("manual"))
        self.training_status_text = f"预标注失败: {message}"
        self.refresh_properties_panel()
        self.update_status_bar()
        self.current_inference_request = None
        if manual:
            QMessageBox.warning(self, "预标注失败", message)

    def mark_current_image_completed(self):
        """标记当前图片为已完成。"""
        if not self.current_image_path:
            return
        self.current_image_state["annotation_completed"] = True
        self.current_image_state["dirty_since_last_train"] = True
        self.mark_annotation_changed()
        if self.save_current_annotation_auto(force=True):
            QMessageBox.information(self, "状态更新", "当前图片已标记为已完成")

    def mark_current_image_incomplete(self):
        """取消当前图片已完成状态。"""
        if not self.current_image_path:
            return
        self.current_image_state["annotation_completed"] = False
        self.current_image_state["dirty_since_last_train"] = False
        self.mark_annotation_changed()
        if self.save_current_annotation_auto(force=True):
            QMessageBox.information(self, "状态更新", "当前图片已取消已完成状态")

    def toggle_annotation_status(self):
        """兼容旧按钮：在已完成 / 未完成间切换。"""
        if self.current_image_state.get("annotation_completed"):
            self.mark_current_image_incomplete()
        else:
            self.mark_current_image_completed()

    def refresh_properties_panel(self):
        """刷新右侧项目状态、实例树和属性信息。"""
        project_name = self.project_metadata.get("project_name") if self.project_metadata else "未加载"
        active_model = self.training_manager.get_active_model_version() or "暂无模型"
        completed_count = self.project_metadata.get("completed_image_count", 0) if self.project_metadata else 0
        dirty_count = self.project_metadata.get("dirty_completed_image_count", 0) if self.project_metadata else 0
        self.restore_button_texts()
        self.properties_panel.update_project_info(
            project_name,
            active_model,
            self.training_status_text,
            completed_count,
            dirty_count,
        )
        self.properties_panel.populate_instance_tree(
            self.left_label.plant_groups,
            self.left_label.plants,
            self.left_label.candidate_instances,
        )
        selected_kind, selected_entity = self.left_label.get_selected_entity()
        self.properties_panel.update_selected_entity(
            selected_kind,
            selected_entity,
            self.project_metadata.get("class_names", []) if self.project_metadata else [],
            self.left_label.plant_groups,
        )
        if selected_kind and selected_entity:
            entity_id = selected_entity.get("id") if selected_kind == "formal" else selected_entity.get("candidate_id")
            self.properties_panel.select_tree_entity(selected_kind, entity_id)
        self.restore_button_visuals()

    def on_canvas_entity_selected(self, entity_kind, entity_id):
        """画布选择同步到右侧面板。"""
        self.refresh_properties_panel()
        self.update_plant_list()

    def on_tree_entity_selected(self, entity_kind, entity_id):
        """树选择同步到画布。"""
        self.left_label.select_entity(entity_kind, entity_id)
        self.refresh_properties_panel()
        self.update_plant_list()

    def on_selected_class_changed(self, class_id):
        """属性面板修改类别。"""
        selected_kind, _ = self.left_label.get_selected_entity()
        if self.left_label.set_selected_entity_class(class_id):
            if selected_kind == "formal":
                self.mark_annotation_changed()
                self.sync_summary_view()
            self.refresh_properties_panel()

    def on_selected_owner_changed(self, owner_plant_id):
        """属性面板修改所属植株组。"""
        selected_kind, _ = self.left_label.get_selected_entity()
        if self.left_label.set_selected_entity_owner(owner_plant_id):
            if selected_kind == "formal":
                self.mark_annotation_changed()
            self.refresh_properties_panel()

    def create_and_assign_plant_group(self):
        """创建新的植株组并绑定到当前选中对象。"""
        plant_group = self.left_label.create_plant_group()
        self.left_label.set_selected_entity_owner(plant_group["plant_id"])
        if self.left_label.selected_entity_kind == "formal":
            self.mark_annotation_changed()
        self.refresh_properties_panel()

    def on_training_state_changed(self, text):
        self.training_status_text = text
        self.refresh_project_metadata()
        self.refresh_properties_panel()
        self.update_status_bar()

    def on_training_progress_changed(self, value, text):
        self.training_status_text = text
        self.properties_panel.update_training_progress(value, text)
        self.restore_button_visuals()
        self.update_status_bar()

    def on_training_finished(self, success, message):
        self.training_status_text = message
        self.refresh_project_metadata()
        self.refresh_properties_panel()
        self.update_status_bar()
        if success:
            if self.current_image_path and not self.left_label.plants and not self.current_image_state.get("annotation_completed"):
                self.maybe_auto_infer_current_image()
            QMessageBox.information(self, "训练完成", message)
        else:
            QMessageBox.warning(self, "训练失败", message)
        self.restore_button_texts()
        self.restore_button_visuals()

    def on_active_model_changed(self, version_name):
        self.refresh_project_metadata()
        self.refresh_properties_panel()
        self.update_status_bar()

    def on_project_counts_changed(self, completed_count, dirty_count):
        if self.project_metadata:
            self.project_metadata["completed_image_count"] = completed_count
            self.project_metadata["dirty_completed_image_count"] = dirty_count
        self.refresh_properties_panel()

    def on_entity_geometry_modified(self):
        """几何拖拽完成后的同步入口。"""
        self.mark_annotation_changed()
        self.sync_summary_view()
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

    def restore_button_texts(self):
        """集中恢复动态和静态按钮文本，避免异常回调后出现空白按钮。"""
        if hasattr(self, "btn_save_polygon"):
            self.btn_save_polygon.setText(f"暂存当前区域 ({SHORTCUTS['SAVE_POLYGON']})")
        if hasattr(self, "btn_save_plant"):
            self.btn_save_plant.setText(f"保存整株 ({SHORTCUTS['SAVE_PLANT']})")
        if hasattr(self, "btn_undo"):
            self.btn_undo.setText(f"撤销 ({SHORTCUTS['UNDO']})")
        if hasattr(self, "btn_prev"):
            self.btn_prev.setText(f"上一张 ({SHORTCUTS['PREV_IMAGE']})")
        if hasattr(self, "btn_next"):
            self.btn_next.setText(f"下一张 ({SHORTCUTS['NEXT_IMAGE']})")
        if hasattr(self, "btn_delete"):
            self.btn_delete.setText(f"删除选中植株 ({SHORTCUTS['DELETE_PLANT']})")
        if hasattr(self, "btn_load_batch"):
            self.btn_load_batch.setText(f"批量加载图片 ({SHORTCUTS['LOAD_BATCH']})")
        if hasattr(self, "btn_load_single"):
            self.btn_load_single.setText("加载单张图片")
        if hasattr(self, "btn_export_json"):
            self.btn_export_json.setText("导出当前JSON")
        if hasattr(self, "btn_export_coco"):
            self.btn_export_coco.setText("导出当前COCO")
        if hasattr(self, "btn_export_yolo"):
            self.btn_export_yolo.setText("导出项目YOLO数据集")
        if hasattr(self, "btn_export_annotated"):
            self.btn_export_annotated.setText("批量导出已完成")
        if hasattr(self, "btn_help"):
            self.btn_help.setText("使用说明")
        if hasattr(self, "btn_load_sam"):
            self.btn_load_sam.setText("加载SAM模型")
        if hasattr(self, "btn_toggle_annotation"):
            if self.current_image_state.get("annotation_completed"):
                self.btn_toggle_annotation.setText("取消当前图片已完成")
            else:
                self.btn_toggle_annotation.setText("标记当前图片为已完成")
        if hasattr(self, "btn_region_growing"):
            if self.region_growing_enabled:
                self.btn_region_growing.setText(f"退出膨胀点选 ({SHORTCUTS['TOGGLE_REGION_GROWING']})")
            else:
                self.btn_region_growing.setText(f"膨胀点选 ({SHORTCUTS['TOGGLE_REGION_GROWING']})")
        if hasattr(self, "btn_sam_segment"):
            if self.sam_segmenting:
                self.btn_sam_segment.setText(f"退出分割 ({SHORTCUTS['TOGGLE_SAM_SEGMENTATION']})")
            else:
                self.btn_sam_segment.setText(f"SAM 分割 ({SHORTCUTS['TOGGLE_SAM_SEGMENTATION']})")
        if hasattr(self, "btn_toggle_snap"):
            self.update_snap_button_state()
        if hasattr(self, "properties_panel"):
            self.properties_panel.restore_button_texts()

    def restore_button_visuals(self):
        """训练异常后强制刷新按钮调色板和重绘，避免按钮文字可见性丢失。"""
        for button in self.findChildren(QPushButton):
            palette = button.palette()
            background = palette.color(QPalette.Button)
            luminance = (
                background.red() * 299 + background.green() * 587 + background.blue() * 114
            ) / 1000
            text_color = QColor("#111111") if luminance >= 160 else QColor("#f5f5f5")
            palette.setColor(QPalette.ButtonText, text_color)
            palette.setColor(QPalette.WindowText, text_color)
            button.setPalette(palette)
            button.ensurePolished()
            if button.style():
                button.style().unpolish(button)
                button.style().polish(button)
            button.update()

    # def load_sam_model(self):
    #     """加载 SAM 模型。"""
    #     if SamModel is None:
    #         QMessageBox.warning(self, "警告", "SAM 库未安装，请安装 segment-anything 包")
    #         return

    #     model_path, _ = QFileDialog.getOpenFileName(self, "选择 SAM 模型文件", ".", "PTH 文件 (*.pth)")
    #     if not model_path:
    #         return

    #     progress = QProgressDialog("正在加载 SAM 模型...", "取消", 0, 100, self)
    #     progress.setWindowModality(Qt.WindowModal)
    #     progress.setValue(0)
    #     try:
    #         self.sam_model = SamModel(model_type=self.sam_model_type)
    #         progress.setValue(20)
    #         success = self.sam_model.load_model(model_path)
    #         if success:
    #             progress.setValue(100)
    #             self.sam_model_loaded = True
    #             self.btn_sam_segment.setEnabled(True)
    #             QMessageBox.information(self, "成功", f"SAM 模型加载成功，设备: {self.sam_model.get_device()}")
    #         else:
    #             QMessageBox.critical(self, "错误", "加载 SAM 模型失败")
    #     except Exception as error:
    #         QMessageBox.critical(self, "错误", f"加载 SAM 模型失败: {error}")
    #     finally:
    #         progress.close()

    # def toggle_sam_segmentation(self):
    #     """切换 SAM 分割模式。"""
    #     if not self.sam_model_loaded:
    #         QMessageBox.warning(self, "警告", "请先加载 SAM 模型")
    #         return
    #     if not self.current_image:
    #         QMessageBox.warning(self, "警告", "请先加载图片")
    #         return
    #     if self.region_growing_enabled:
    #         self.toggle_region_growing()

    #     self.sam_segmenting = not self.sam_segmenting
    #     if self.sam_segmenting:
    #         self.btn_sam_segment.setText(f"退出分割 ({SHORTCUTS['TOGGLE_SAM_SEGMENTATION']})")
    #         self.left_label.sam_segmenting = True
    #         self.left_label.sam_predictor = self.sam_model.predictor
    #         self.left_label.sam_prompt_points = []
    #         import numpy as np

    #         self.sam_model.set_image(np.array(self.current_image.convert("RGB")))
    #     else:
    #         self.btn_sam_segment.setText(f"SAM 分割 ({SHORTCUTS['TOGGLE_SAM_SEGMENTATION']})")
    #         self.left_label.sam_segmenting = False
    #         self.left_label.sam_prompt_points = []
    #     self.left_label.update_display()
    #     self.update_status_bar()

    def toggle_region_growing(self):
        """切换区域生长模式。"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        if self.sam_segmenting:
            self.toggle_sam_segmentation()

        self.region_growing_enabled = not self.region_growing_enabled
        if self.region_growing_enabled:
            self.btn_region_growing.setText(f"退出膨胀点选 ({SHORTCUTS['TOGGLE_REGION_GROWING']})")
            self.left_label.region_growing_enabled = True
            self.left_label.ignoring_region = False
        else:
            self.btn_region_growing.setText(f"膨胀点选 ({SHORTCUTS['TOGGLE_REGION_GROWING']})")
            self.left_label.region_growing_enabled = False
            self.left_label.region_growing_mask = None
        self.left_label.update_display()
        self.update_status_bar()
    
    def toggle_ignore_region(self):
        """切换忽略区域绘制模式。"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        self.ignoring_region = not self.ignoring_region
        if self.ignoring_region:
            self.btn_ignore_region.setText(f"退出忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            self.left_label.ignoring_region = True
            self.left_label.region_growing_enabled = False
            self.left_label.current_ignored_points = []
        else:
            self.btn_ignore_region.setText(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            self.left_label.ignoring_region = False
        self.left_label.update_display()
        self.update_status_bar()

    def show_help(self):
        dialog = HelpDialog(self)
        dialog.exec_()

    def save_current_polygon(self):
        if self.ignoring_region:
            if self.left_label.save_current_ignored_region():
                self.mark_annotation_changed()
                self.update_status_bar()
        else:
            if self.left_label.save_current_polygon():
                self.mark_annotation_changed()
                self.update_status_bar()

    def save_plant(self):
        """兼容旧快捷键：保存当前手工实例。"""
        saved_id = self.left_label.confirm_preview_and_save()
        if saved_id:
            saved_instance = next((plant for plant in self.left_label.plants if plant["id"] == saved_id), None)
            if saved_instance:
                self.push_undo_action("add_instance", saved_instance)
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()
            self.refresh_properties_panel()
        else:
            QMessageBox.warning(self, "警告", "请先暂存至少一个区域")

    def accept_selected_candidate(self):
        accepted_id = self.left_label.accept_selected_candidate()
        if accepted_id:
            instance = next((plant for plant in self.left_label.plants if plant["id"] == accepted_id), None)
            if instance:
                self.push_undo_action("add_instance", instance)
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()
            self.refresh_properties_panel()

    def accept_all_candidates(self):
        accepted_ids = self.left_label.accept_all_candidates()
        if accepted_ids:
            for accepted_id in accepted_ids:
                instance = next((plant for plant in self.left_label.plants if plant["id"] == accepted_id), None)
                if instance:
                    self.push_undo_action("add_instance", instance)
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            self.mark_annotation_changed()
            self.refresh_properties_panel()

    def clear_candidates(self):
        self.left_label.clear_candidates()
        self.refresh_properties_panel()

    def undo(self):
        """撤销临时绘制或实例级增删。"""
        if self.left_label.undo_last_action():
            self.mark_annotation_changed()
            self.update_undo_redo_state()
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
        deleted = self.left_label.delete_selected_entity()
        if deleted:
            if selected_kind == "formal":
                self.mark_annotation_changed()
            self.sync_summary_view()
            self.update_plant_list()
            self.update_undo_redo_state()
            self.refresh_properties_panel()
            self.update_status_bar()

    def on_plant_selected(self, text):
        if not text:
            return
        try:
            parts = text.split()
            if len(parts) < 2:
                return
            instance_id = int(parts[1])
            if (
                self.left_label.selected_entity_kind == "formal"
                and int(self.left_label.selected_entity_id or 0) == instance_id
            ):
                return
            self.left_label.select_entity("formal", instance_id)
            self.refresh_properties_panel()
        except (ValueError, IndexError):
            return

    def update_plant_list(self):
        current_selected_id = None
        if self.left_label.selected_entity_kind == "formal":
            try:
                current_selected_id = int(self.left_label.selected_entity_id)
            except (TypeError, ValueError):
                current_selected_id = None

        self.combo_plants.blockSignals(True)
        self.combo_plants.clear()
        for plant in self.left_label.plants:
            self.combo_plants.addItem(f"实例 {plant['id']} {plant.get('class_name', '')}")
            if current_selected_id is not None and int(plant["id"]) == current_selected_id:
                self.combo_plants.setCurrentIndex(self.combo_plants.count() - 1)
        self.combo_plants.blockSignals(False)

    def update_undo_redo_state(self):
        can_undo = bool(
            self.undo_stack
            or self.left_label.current_points
            or self.left_label.current_plant_polygons
        )
        self.btn_undo.setEnabled(can_undo)

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

    def export_simple_json(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        
        # 让用户选择导出路径
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_filename = f"{base_name}_annotation.json"
        export_path, _ = QFileDialog.getSaveFileName(
            self, "导出 JSON 文件", default_filename, "JSON 文件 (*.json)"
        )
        
        if not export_path:
            return  # 用户取消选择
        
        # 调用导出函数
        export_result = export_simple_json_file(
            self.current_image_path,
            self.left_label.plants,
            plant_groups=self.left_label.plant_groups,
            image_state=self.current_image_state,
            export_path=export_path,
            class_names=self.project_metadata.get("class_names") if self.project_metadata else None,
            ignored_regions=self.left_label.ignored_regions,
        )
        
        if export_result:
            QMessageBox.information(self, "成功", f"JSON 导出成功：{export_result}")

    def export_coco_format(self):
        if not self.current_image_path or not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        
        # 让用户选择导出路径
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_filename = f"{base_name}_coco.json"
        export_path, _ = QFileDialog.getSaveFileName(
            self, "导出 COCO 文件", default_filename, "JSON 文件 (*.json)"
        )
        
        if not export_path:
            return  # 用户取消选择
        
        # 调用导出函数
        export_result = export_coco_file(
            self.current_image_path,
            self.left_label.plants,
            self.current_image.width,
            self.current_image.height,
            export_path=export_path,
            class_names=self.project_metadata.get("class_names") if self.project_metadata else None,
            ignored_regions=self.left_label.ignored_regions,
        )
        
        if export_result:
            QMessageBox.information(self, "成功", f"COCO 导出成功：{export_result}")

    def export_yolo_dataset(self):
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        # 让用户选择导出目录
        default_dir = os.path.join(ANNOTATION_DIR, f"yolo_dataset_{current_timestamp()}")
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择 YOLO 数据集导出目录", default_dir
        )
        
        if not export_dir:
            return  # 用户取消选择
        
        try:
            dataset_info = build_project_dataset(self.project_id, rebuild_split=False, dataset_root=export_dir)
            QMessageBox.information(
                self,
                "成功",
                f"YOLO 数据集已生成：{dataset_info['dataset_root']}\n"
                f"train={dataset_info['train_count']} val={dataset_info['val_count']}",
            )
        except Exception as error:
            QMessageBox.critical(self, "错误", f"导出 YOLO 数据集失败: {error}")

    def export_annotated_images(self):
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        # 让用户选择导出目录
        default_dir = os.path.join(ANNOTATION_DIR, f"completed_export_{current_timestamp()}")
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", default_dir
        )
        
        if not export_dir:
            return  # 用户取消选择
        
        # 调用导出函数
        export_result, exported_count = export_completed_annotations(self.project_id, export_dir=export_dir)
        if export_result:
            QMessageBox.information(self, "成功", f"已完成图片 JSON 导出 {exported_count} 份到：{export_result}")
        else:
            QMessageBox.warning(self, "警告", "当前项目没有已完成图片")

    def start_manual_training(self):
        success, message = self.training_manager.check_and_trigger_training(force=True)
        if not success:
            QMessageBox.warning(self, "手动训练", message)

    def rollback_active_model(self):
        success, message = self.training_manager.rollback_to_previous()
        if success:
            QMessageBox.information(self, "模型回退", f"已回退到 {message}")
        else:
            QMessageBox.warning(self, "模型回退", message)

    def rebuild_validation_split(self):
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        try:
            dataset_info = build_project_dataset(self.project_id, rebuild_split=True)
            QMessageBox.information(
                self,
                "验证集已重建",
                f"固定验证集已重建\ntrain={dataset_info['train_count']} val={dataset_info['val_count']}",
            )
        except Exception as error:
            QMessageBox.critical(self, "错误", f"重建验证集失败: {error}")

    def update_status_bar(self):
        """更新底部状态栏。"""
        base_msg = "就绪"

        if self.project_metadata:
            base_msg += f" | 项目: {self.project_metadata.get('project_name', self.project_id)}"
            active_model = self.training_manager.get_active_model_version() or "暂无模型"
            base_msg += f" | 模型: {active_model}"
            base_msg += f" | Dirty Completed: {self.project_metadata.get('dirty_completed_image_count', 0)}"

        if self.image_paths and self.current_image_index >= 0:
            progress_text = f"{self.current_image_index + 1}/{len(self.image_paths)}"
            image_name = os.path.basename(self.current_image_path)
            completed_flag = "[已完成]" if self.current_image_state.get("annotation_completed") else "[未完成]"
            base_msg += f" | 当前: {image_name} {completed_flag}"
            base_msg += f" | 图片: {progress_text}"
            self.image_progress_label.setText(progress_text)
        else:
            self.image_progress_label.setText("0/0")

        base_msg += " | 边缘吸附: 开启" if self.left_label.edge_snap_enabled else " | 边缘吸附: 关闭"
        base_msg += f" | 正式实例: {len(self.left_label.plants)}"
        base_msg += f" | 候选实例: {len(self.left_label.candidate_instances)}"

        if self.left_label.current_points or self.left_label.current_plant_polygons:
            base_msg += (
                f" | 当前多边形顶点: {len(self.left_label.current_points)}"
                f" | 已暂存区域: {len(self.left_label.current_plant_polygons)}"
            )

        if self.annotation_changed:
            base_msg += " | 【有未保存的修改】"

        # if self.sam_segmenting:
        #     base_msg += " | SAM 分割模式"
        if self.region_growing_enabled:
            base_msg += " | 膨胀点选模式"
        if self.ignoring_region:
            base_msg += " | 忽略区域模式"

        base_msg += f" | 状态: {self.training_status_text}"
        self.statusBar().showMessage(base_msg)

    def save_undo_state(self):
        """保留接口，当前版本不做完整快照撤销。"""
        return

    def sync_summary_view(self):
        """同步右侧总览。"""
        self.right_label.plants = copy.deepcopy(self.left_label.plants)
        self.right_label.plant_groups = copy.deepcopy(self.left_label.plant_groups)
        self.right_label.selected_entity_kind = self.left_label.selected_entity_kind
        self.right_label.selected_entity_id = self.left_label.selected_entity_id
        self.right_label.selected_plant_id = self.left_label.selected_plant_id
        self.right_label.update_display()

    def closeEvent(self, event):
        """关闭前保存当前正式标注。"""
        try:
            self.persist_current_if_dirty(force=False)
        finally:
            event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())