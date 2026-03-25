# 主入口文件
import copy
import os
import sys
import traceback
import json

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

from config import SHORTCUTS
from components.help_dialog import HelpDialog
from components.image_label import ImageLabel
from components.toolbars import Toolbars
from services.inference_service import InferenceWorker
from services.training_manager import TrainingManager
from ui.annotation_properties_panel import AnnotationPropertiesPanel
from utils.annotation_schema import current_timestamp, make_image_state
from utils.data_manager import (
    load_annotation_from_coco,
    batch_export_annotations,
    batch_import_annotations,
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

        # 使用当前工作目录作为标注保存目录
        self.annotation_dir = os.getcwd()
        self.preprocess_cache = {}
        self.annotation_changed = False
        self.undo_stack = []
        self.redo_stack = []
        self.is_undo_redo = False

        self.project_id = None
        self.project_metadata = None
        self.project_paths = None

        # 新增：COCO容器，存储每个图片的标注数据
        self.coco_container = {}
        # 新增：保存路径记忆
        self.save_path = None
        self.import_path = None
        self.export_path = None

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

        # 记忆导入路径
        if image_paths:
            self.import_path = os.path.dirname(image_paths[0])

        # 建立图像索引，不一次性读入内存
        self.image_paths = image_paths
        self.preprocess_cache.clear()
        self.coco_container.clear()  # 清空COCO容器
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

    def goto_image(self, index):
        """跳转到指定图片。"""
        if index < 0 or index >= len(self.image_paths):
            return

        # 保存当前图片的标注数据到COCO容器
        if self.current_image_path:
            annotation_state = self.left_label.get_annotation_state()
            annotation = {
                "plants": annotation_state["plants"],
                "plant_groups": annotation_state["plant_groups"],
                "current_plant_id": annotation_state["current_plant_id"],
                "next_plant_group_id": annotation_state["next_owner_plant_id"],
                "ignored_regions": self.left_label.ignored_regions,
                "image_state": self.current_image_state,
                "class_names": self.left_label.class_names
            }
            self.coco_container[self.current_image_path] = annotation

        image_path = self.image_paths[index]
        try:
            # 根据索引从硬盘读取图片
            image = load_image(image_path)
            if not image:
                QMessageBox.warning(self, "警告", f"无法加载图片: {image_path}")
                return

            self.current_image = image
            self.current_image_path = image_path
            self.current_image_index = index

            # 加载预处理数据
            preprocessed_data = self.preprocess_cache.get(image_path)
            if not preprocessed_data:
                preprocessed_data = preprocess_image(self.current_image)
                self.preprocess_cache[image_path] = preprocessed_data

            # 设置图片
            self.left_label.set_image(self.current_image, preprocessed_data)
            self.right_label.set_image(self.current_image, preprocessed_data)

            # 根据索引从COCO容器读取标注数据
            self.load_annotation_from_coco_container()

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

    def load_annotation_from_coco_container(self):
        """从COCO容器加载标注。"""
        if not self.current_image_path:
            return

        # 尝试从COCO容器加载
        if self.current_image_path in self.coco_container:
            annotation = self.coco_container[self.current_image_path]
        else:
            # 尝试从文件加载
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            coco_path = os.path.join(self.save_path or os.getcwd(), f"{base_name}_coco.json")
            annotation = load_annotation_from_coco(coco_path)
            if annotation:
                self.coco_container[self.current_image_path] = annotation

        if annotation:
            self.left_label.set_class_names(annotation.get("class_names", []))
            self.right_label.set_class_names(annotation.get("class_names", []))
            self.left_label.set_annotation_state(
                annotation["plants"],
                plant_groups=annotation.get("plant_groups"),
                current_plant_id=annotation.get("current_plant_id", 1),
                next_owner_plant_id=annotation.get("next_plant_group_id", 1),
            )
            # 设置忽略区域
            self.left_label.ignored_regions = annotation.get("ignored_regions", [])
            self.current_image_state = annotation.get("image_state", make_image_state(self.current_image_path))
            self.current_annotation_hash = annotation.get("annotation_hash")
        else:
            self.left_label.set_annotation_state([], plant_groups=[], current_plant_id=1, next_owner_plant_id=1)
            # 清空忽略区域
            self.left_label.ignored_regions = []
            self.current_image_state = make_image_state(self.current_image_path, annotation_completed=False)
            self.current_annotation_hash = None

        self.left_label.clear_candidates()

    def prev_image(self):
        if self.current_image_index > 0:
            self.goto_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.goto_image(self.current_image_index + 1)

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
        QMessageBox.information(self, "状态更新", "当前图片已标记为已完成")

    def mark_current_image_incomplete(self):
        """取消当前图片已完成状态。"""
        if not self.current_image_path:
            return
        self.current_image_state["annotation_completed"] = False
        self.current_image_state["dirty_since_last_train"] = False
        self.mark_annotation_changed()
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
        if hasattr(self, "btn_export_annotated"):
            self.btn_export_annotated.setText("批量导出已完成")
        if hasattr(self, "btn_help"):
            self.btn_help.setText("使用说明")
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
        if hasattr(self, "btn_ignore_region"):
            if self.ignoring_region:
                self.btn_ignore_region.setText(f"退出忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            else:
                self.btn_ignore_region.setText(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
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

    def toggle_region_growing(self):
        """切换区域生长模式。"""
        if not self.current_image:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

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

    def toggle_ai_assist(self):
        """切换AI辅助功能。"""
        self.ai_assist_enabled = not getattr(self, 'ai_assist_enabled', True)
        if self.ai_assist_enabled:
            self.btn_toggle_ai.setText("AI辅助: 开启")
            # 启用AI辅助相关功能
        else:
            self.btn_toggle_ai.setText("AI辅助: 关闭")
            # 禁用AI辅助相关功能
        self.left_label.update_display()
        self.update_status_bar()

    def clear_last_ignore_region(self):
        """清除上一个忽略区域。"""
        if self.left_label.ignored_regions:
            self.left_label.ignored_regions.pop()
            self.left_label.update_display()
            self.sync_summary_view()
            self.mark_annotation_changed()
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

    def export_annotated_images(self):
        """批量导出已完成的标注"""
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        # 筛选已完成的图片
        completed_images = []
        for image_path in self.image_paths:
            if image_path in self.coco_container:
                state = self.coco_container[image_path].get("image_state", {})
                if state.get("annotation_completed"):
                    completed_images.append(image_path)
            else:
                # 尝试从文件加载状态
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                coco_path = os.path.join(self.save_path or os.getcwd(), f"{base_name}_coco.json")
                annotation = load_annotation_from_coco(coco_path)
                if annotation and annotation.get("image_state", {}).get("annotation_completed"):
                    completed_images.append(image_path)
                    self.coco_container[image_path] = annotation
        
        if not completed_images:
            QMessageBox.warning(self, "警告", "当前没有已完成的图片")
            return
        
        # 让用户选择导出目录
        default_dir = self.export_path or os.path.join(os.getcwd(), f"completed_export_{current_timestamp()}")
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", default_dir
        )
        
        if not export_dir:
            return  # 用户取消选择
        
        # 记忆导出路径
        self.export_path = export_dir
        
        # 创建进度对话框
        progress = QProgressDialog("正在批量导出标注...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.show()
        
        # 定义进度回调函数
        def progress_callback(current, total, message):
            progress.setValue(int(current / total * 100))
            progress.setLabelText(message)
            QApplication.processEvents()
            return not progress.wasCanceled()
        
        # 执行批量导出
        result = batch_export_annotations(
            completed_images,
            export_dir,
            project_id=self.project_id,
            progress_callback=progress_callback,
            coco_container=self.coco_container
        )
        
        progress.close()
        
        # 显示导出结果
        result_text = f"批量导出完成：\n"
        result_text += f"成功导出: {result['exported']}\n"
        result_text += f"错误: {result['errors']}\n"
        QMessageBox.information(self, "批量导出完成", result_text)

    def import_batch_data(self):
        """批量导入数据到当前项目"""
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        
        # 让用户选择导入目录
        import_dir = QFileDialog.getExistingDirectory(
            self, "选择导入目录", self.import_path or ""
        )
        
        if not import_dir:
            return  # 用户取消选择
        
        # 记忆导入路径
        self.import_path = import_dir
        
        # 让用户选择冲突处理策略
        msg_box = QMessageBox()
        msg_box.setWindowTitle("冲突处理策略")
        msg_box.setText("请选择标注冲突处理策略：")
        msg_box.addButton("跳过", QMessageBox.YesRole)
        msg_box.addButton("覆盖", QMessageBox.NoRole)
        msg_box.addButton("取消", QMessageBox.RejectRole)
        
        reply = msg_box.exec_()
        if reply == 2:  # 取消
            return
        
        conflict_strategy = "skip" if reply == 0 else "overwrite"
        
        # 创建进度对话框
        progress = QProgressDialog("正在批量导入标注...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.show()
        
        # 定义进度回调函数
        def progress_callback(current, total, message):
            progress.setValue(int(current / total * 100))
            progress.setLabelText(message)
            QApplication.processEvents()
            return not progress.wasCanceled()
        
        # 调用批量导入函数
        result = batch_import_annotations(
            import_dir,
            self.image_paths,
            self.project_id,
            conflict_strategy=conflict_strategy,
            progress_callback=progress_callback
        )
        
        # 处理临时COCO容器，与当前容器和图像索引对比
        temp_coco_container = result.get("temp_coco_container", {})
        for image_path, annotation in temp_coco_container.items():
            # 检查图像索引是否一致
            if image_path in self.image_paths:
                # 索引一致，覆盖当前容器中的数据
                self.coco_container[image_path] = annotation
                
                # 如果当前显示的就是这张图片，更新显示的标注状态
                if image_path == self.current_image_path:
                    self.load_annotation_from_coco_container()
        
        progress.close()
        
        # 显示导入结果
        result_text = f"批量导入完成：\n"
        result_text += f"成功导入: {result['imported']}\n"
        result_text += f"跳过: {result['skipped']}\n"
        result_text += f"错误: {result['errors']}\n"
        QMessageBox.information(self, "批量导入完成", result_text)

    def export_yolo_dataset(self):
        """导出为 YOLO 格式数据集。"""
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return

        # 让用户选择导出目录
        default_dir = self.export_path or os.path.join(os.getcwd(), f"yolo_export_{current_timestamp()}")
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", default_dir
        )
        if not export_dir:
            return

        # 记忆导出路径
        self.export_path = export_dir

        # 显示进度对话框
        progress = QProgressDialog("正在导出 YOLO 数据集...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.show()

        # 定义进度回调
        def progress_callback(current, total, message):
            progress.setValue(int(current / total * 100))
            progress.setLabelText(message)
            QApplication.processEvents()
            return not progress.wasCanceled()

        # 构建数据集
        try:
            result = build_project_dataset(
                self.project_id,
                rebuild_split=False,
                dataset_root=export_dir
            )

            progress.close()

            QMessageBox.information(
                self,
                "导出成功",
                f"YOLO 数据集导出成功！\n保存在: {result['dataset_root']}"
            )
        except Exception as e:
            progress.close()
            QMessageBox.warning(
                self,
                "导出失败",
                f"导出失败: {str(e)}"
            )

    def start_manual_training(self):
        """手动触发一次训练。"""
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        # 显示确认对话框
        reply = QMessageBox.question(
            self,
            "开始训练",
            "确定要开始训练吗？\n训练过程可能会持续一段时间。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.training_manager.start_training()

    def rollback_active_model(self):
        """回滚到上一个模型版本。"""
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        success, message = self.training_manager.rollback_model()
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "失败", message)

    def rebuild_validation_split(self):
        """重新划分验证集。"""
        if not self.project_id:
            QMessageBox.warning(self, "警告", "当前没有活动项目")
            return
        
        success, message = self.training_manager.rebuild_validation_split()
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "失败", message)

    def sync_summary_view(self):
        """同步右侧总览视图。"""
        if not self.current_image:
            return

        # 复制左侧标注状态到右侧
        plants = copy.deepcopy(self.left_label.plants)
        plant_groups = copy.deepcopy(self.left_label.plant_groups)
        ignored_regions = copy.deepcopy(self.left_label.ignored_regions)

        self.right_label.set_annotation_state(
            plants,
            plant_groups=plant_groups,
            current_plant_id=self.left_label.current_plant_id,
            next_owner_plant_id=self.left_label.next_owner_plant_id,
        )
        self.right_label.ignored_regions = ignored_regions
        self.right_label.update_display()

    def update_status_bar(self):
        """更新状态栏信息。"""
        status_parts = []

        # 图片信息
        if self.current_image_path:
            image_name = os.path.basename(self.current_image_path)
            status_parts.append(f"图片: {image_name}")
            if self.current_image_index >= 0 and self.image_paths:
                status_parts.append(f"({self.current_image_index + 1}/{len(self.image_paths)})")

        # 标注状态
        if self.current_image_state:
            completed = self.current_image_state.get("annotation_completed", False)
            status_parts.append(f"状态: {'已完成' if completed else '未完成'}")

        # 未保存修改
        if self.annotation_changed:
            status_parts.append("⚠ 有未保存修改")

        # 边缘吸附状态
        if hasattr(self, "left_label") and hasattr(self.left_label, "edge_snap_enabled"):
            status_parts.append(f"边缘吸附: {'开启' if self.left_label.edge_snap_enabled else '关闭'}")

        # 训练状态
        if self.training_status_text:
            status_parts.append(f"训练: {self.training_status_text}")

        status_text = " | ".join(status_parts)
        self.statusBar().showMessage(status_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())