import os
from time import monotonic

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import SHORTCUTS
from components.image_label import ImageLabel
from components.toolbars import Toolbars
from components.toolbars import _apply_toolbar_button_accents
from models.sam_manager import SamManager
from services.sam_training_manager import SamTrainingManagerV2
from ui.annotation_properties_panel import AnnotationPropertiesPanel
from utils.annotation_schema import (
    format_elapsed_seconds,
    make_image_state,
    normalize_annotation_timing_state,
)
from utils.interaction_state import InteractionStateMachine
from utils.preannotation_records import SEMANTIC_ACTION_AUTO


class MainWindowBase(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玉米标注与项目级预标注工具")
        self.setMinimumSize(1024, 680)

        self.image_paths = []
        self.image_sequence_map = {}
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.current_image_state = make_image_state("")
        self.current_annotation_hash = None

        self.annotation_dir = os.getcwd()
        self.preprocess_cache = {}
        self.annotation_changed = False
        self.undo_stack = []
        self.redo_stack = []
        self.is_undo_redo = False

        self.project_id = None
        self.project_metadata = None
        self.project_paths = None

        self.coco_container = {}
        self.save_path = None
        self.import_path = None
        self.export_path = None

        self.ignoring_region = False

        self.sam_manager = SamManager()
        self.sam_training_manager = SamTrainingManagerV2(self.sam_manager)
        self.sam_training_worker = None
        self.current_preannotation_candidate = None
        self.preannotation_adjustment_records = []
        self.preannotation_records_by_image = {}
        self.preannotation_record_counter = 1
        self.preannotation_fine_tune_sessions = {}
        self.preannotation_default_reason_code = None
        self.preannotation_default_semantic_action = SEMANTIC_ACTION_AUTO
        self._updating_preannotation_controls = False
        self.interaction_state_machine = InteractionStateMachine()
        self.annotation_timer = QTimer(self)
        self.annotation_timer.setInterval(1000)
        self.annotation_timer.timeout.connect(self.update_timing_panel)
        self._timer_running = False
        self._timer_started_monotonic = None

        self.apply_window_theme()
        self.init_ui()
        _apply_toolbar_button_accents(self)
        self.resize_to_available_screen()
        self.init_shortcuts()
        self.restore_button_texts()
        self.apply_toolbar_compaction()
        self.restore_button_visuals()
        self._update_preannotation_controls()

        self.update_status_bar()

    def _resolve_image_sequence(self, image_path=None):
        target_image = image_path or self.current_image_path
        if not target_image:
            return None
        sequence = self.image_sequence_map.get(target_image)
        if sequence is not None:
            return sequence
        if target_image in self.image_paths:
            sequence = self.image_paths.index(target_image) + 1
            self.image_sequence_map[target_image] = sequence
            return sequence
        if self.current_image_index >= 0:
            return self.current_image_index + 1
        return None

    def apply_window_theme(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f1ec;
            }
            QWidget {
                color: #2f241d;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QGroupBox {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #fbfaf7, stop:1 #f1eee7);
                border: 1px solid #ddd7cb;
                border-radius: 14px;
                margin-top: 10px;
                padding: 14px 10px 10px 10px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 2px 7px;
                color: #7d6757;
                background: #f4f0e8;
                border-radius: 7px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #faf8f3, stop:1 #efe9df);
                border: 1px solid #d5ccbe;
                border-radius: 10px;
                padding: 8px 10px;
                text-align: left;
                font-weight: 600;
            }
            QPushButton[accent="primary"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d98f52, stop:1 #b76534);
                color: #fffaf5;
                border: 1px solid #9f532b;
            }
            QPushButton[accent="primary"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e39c62, stop:1 #c56f3b);
                border: 1px solid #8f4723;
            }
            QPushButton[accent="muted"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e8e5df, stop:1 #d8d3ca);
                color: #3f3a34;
                border: 1px solid #b9b1a5;
            }
            QPushButton[accent="muted"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dfdbd4, stop:1 #ccc6bc);
                border: 1px solid #9f978a;
            }
            QPushButton[accent="danger"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f8d7cf, stop:1 #df8e7b);
                border: 1px solid #be5d48;
                color: #5a1f12;
            }
            QPushButton[accent="danger"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f4c8bc, stop:1 #d97a64);
                border: 1px solid #a84d39;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f7f3ea, stop:1 #ebe3d6);
                border: 1px solid #c9bead;
            }
            QPushButton:pressed {
                background: #e7dfd2;
                border: 1px solid #bcae98;
            }
            QPushButton:disabled {
                background: #ebe7df;
                color: #988677;
                border: 1px solid #d3c4b4;
            }
            QComboBox, QTextEdit {
                background: #fffdf9;
                border: 1px solid #d8c4ad;
                border-radius: 9px;
                padding: 5px 8px;
                selection-background-color: #d98f52;
                selection-color: white;
            }
            QComboBox:hover, QTextEdit:hover {
                border: 1px solid #c98d5b;
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QLabel {
                color: #4c3b31;
            }
            QStatusBar {
                background: #ece8de;
                color: #5b4637;
                border-top: 1px solid #d9d0c2;
            }
            QScrollBar:vertical {
                background: #ebe7de;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #c7baaa;
                min-height: 30px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def apply_toolbar_compaction(self):
        if hasattr(self, "btn_delete_staging_polygon"):
            self.btn_delete_staging_polygon.setText(
                f"删除选中区域/去除区域 ({SHORTCUTS['DELETE_STAGING_POLYGON']})"
            )
        for button_name, accent in (
            ("btn_save_plant", "primary"),
            ("btn_sam_train", "primary"),
            ("btn_sam_preannotate", "primary"),
            ("btn_timer_start", "primary"),
            ("btn_save_staging_areas", "danger"),
            ("btn_delete", "danger"),
            ("btn_delete_staging_polygon", "danger"),
            ("btn_removal_region", "danger"),
            ("btn_ignore_preannotation", "muted"),
            ("btn_refresh", "muted"),
            ("btn_toggle_annotation", "muted"),
            ("btn_prev", "muted"),
            ("btn_next", "muted"),
            ("btn_timer_pause", "muted"),
            ("btn_toggle_projection", "muted"),
            ("btn_help", "muted"),
            ("btn_debug_coco", "muted"),
        ):
            button = getattr(self, button_name, None)
            if button is None:
                continue
            button.setProperty("accent", accent)
            if button.style():
                button.style().unpolish(button)
                button.style().polish(button)

    def sync_interaction_state(self):
        if self.left_label.preannotation_box_mode:
            return self.interaction_state_machine.force(InteractionStateMachine.PREANNOTATION_BOX)
        if self.left_label.candidate_instances:
            return self.interaction_state_machine.force(InteractionStateMachine.PREANNOTATION_CANDIDATE)
        if self.ignoring_region:
            return self.interaction_state_machine.force(InteractionStateMachine.IGNORE_REGION)
        if self.left_label.removing_region:
            return self.interaction_state_machine.force(InteractionStateMachine.REMOVAL_REGION)
        if self.left_label.mode == "fine_tune":
            if self.left_label.split_staging_mode:
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_SPLIT_STAGING)
            if getattr(self.left_label, "merge_staging_mode", False):
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_MERGE_STAGING)
            if getattr(self.left_label, "delete_vertex_mode", False):
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_DELETE_VERTEX)
            if self.left_label.add_vertex_mode:
                return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE_ADD_VERTEX)
            return self.interaction_state_machine.force(InteractionStateMachine.FINE_TUNE)
        return self.interaction_state_machine.force(InteractionStateMachine.IDLE)

    def init_ui(self):
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
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(Toolbars.create_auxiliary_toolbar(self))
        left_layout.addWidget(Toolbars.create_annotation_toolbar(self))
        left_layout.addWidget(Toolbars.create_plant_management_toolbar(self))
        left_layout.addWidget(Toolbars.create_navigation_toolbar(self))
        left_layout.addWidget(Toolbars.create_timing_toolbar(self))
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

        right_canvas_panel = QWidget()
        right_canvas_layout = QVBoxLayout(right_canvas_panel)
        right_canvas_layout.setContentsMargins(0, 0, 0, 0)
        right_canvas_layout.setSpacing(8)

        self.right_label = ImageLabel(is_summary=True, parent=self)
        self.right_label.setToolTip("正式实例总览，不显示候选层")
        right_canvas_layout.addWidget(self.right_label, 3)
        self.preannotation_panel = Toolbars.create_preannotation_toolbar(self)
        right_canvas_layout.addWidget(self.preannotation_panel, 1)
        right_canvas_layout.setStretch(0, 3)
        right_canvas_layout.setStretch(1, 1)
        images_layout.addWidget(right_canvas_panel, 1)
        main_splitter.addWidget(center_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.properties_panel = AnnotationPropertiesPanel(self)
        right_layout.addWidget(self.properties_panel, 3)
        right_layout.addWidget(Toolbars.create_file_toolbar(self))
        right_layout.addWidget(Toolbars.create_sam_toolbar(self))
        right_layout.addWidget(Toolbars.create_export_toolbar(self))
        right_layout.addWidget(Toolbars.create_aux_toolbar(self))
        right_layout.addStretch()
        right_scroll = self.create_scroll_panel(right_panel, min_width=280, preferred_width=330)
        main_splitter.addWidget(right_scroll)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([250, 1250, 320])

    def create_scroll_panel(self, widget, min_width, preferred_width):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(min_width)
        scroll.setMaximumWidth(max(preferred_width + 80, min_width))
        return scroll

    def resize_to_available_screen(self):
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
        QShortcut(QKeySequence(SHORTCUTS["SAVE_POLYGON"]), self, self.save_current_polygon)
        QShortcut(QKeySequence(SHORTCUTS["SAVE_PLANT"]), self, self.save_plant)
        QShortcut(QKeySequence(SHORTCUTS["UNDO"]), self, self.undo)
        QShortcut(QKeySequence(SHORTCUTS["REDO"]), self, self.redo)
        QShortcut(QKeySequence(SHORTCUTS["DELETE_PLANT"]), self, self.delete_plant)
        QShortcut(QKeySequence(SHORTCUTS["DELETE_STAGING_POLYGON"]), self, self.delete_selected_staging_polygon_shortcut)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_EDGE_SNAP"]), self, self.toggle_edge_snap)
        QShortcut(QKeySequence(SHORTCUTS["LOAD_BATCH"]), self, self.load_batch_images)
        QShortcut(QKeySequence(SHORTCUTS["PREV_IMAGE"]), self, self.prev_image)
        QShortcut(QKeySequence(SHORTCUTS["NEXT_IMAGE"]), self, self.next_image)
        QShortcut(QKeySequence(SHORTCUTS["TOGGLE_IGNORE_REGION"]), self, self.toggle_ignore_region)

    def restore_button_texts(self):
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
        if hasattr(self, "btn_timer_start"):
            self.btn_timer_start.setText("开始/继续计时")
        if hasattr(self, "btn_timer_pause"):
            self.btn_timer_pause.setText("暂停计时")
        if hasattr(self, "btn_delete"):
            self.btn_delete.setText(f"删除选中植株 ({SHORTCUTS['DELETE_PLANT']})")
        if hasattr(self, "btn_delete_staging_polygon"):
            self.btn_delete_staging_polygon.setText(
                f"删除选中区域/去除区域 ({SHORTCUTS['DELETE_STAGING_POLYGON']})"
            )
        if hasattr(self, "btn_delete_vertex"):
            self.btn_delete_vertex.setText("删除顶点")
        if hasattr(self, "btn_load_batch"):
            self.btn_load_batch.setText(f"批量加载图片 ({SHORTCUTS['LOAD_BATCH']})")
        if hasattr(self, "btn_load_folder"):
            self.btn_load_folder.setText("从文件夹加载图片（非 COCO）")
        if hasattr(self, "btn_export_annotated"):
            self.btn_export_annotated.setText("批量导出已完成(coco格式)")
        if hasattr(self, "btn_help"):
            self.btn_help.setText("使用说明")
        if hasattr(self, "btn_debug_coco"):
            self.btn_debug_coco.setText("调试COCO容器")
        if hasattr(self, "btn_toggle_annotation"):
            if self.current_image_state.get("annotation_completed"):
                self.btn_toggle_annotation.setText("取消当前图片已完成")
            else:
                self.btn_toggle_annotation.setText("标记当前图片为已完成")
        if hasattr(self, "btn_ignore_region"):
            if self.ignoring_region:
                self.btn_ignore_region.setText(f"退出忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
            else:
                self.btn_ignore_region.setText(f"忽略区域 ({SHORTCUTS['TOGGLE_IGNORE_REGION']})")
        if hasattr(self, "btn_toggle_snap"):
            self.update_snap_button_state()
        if hasattr(self, "btn_start_training"):
            self.btn_start_training.setText("开始训练")
        if hasattr(self, "btn_sam_train"):
            self.btn_sam_train.setText("开始训练")
        if hasattr(self, "btn_sam_preannotate"):
            self.btn_sam_preannotate.setText("取消框选预标注" if self.left_label.preannotation_box_mode else "框选预标注")
        if hasattr(self, "btn_sam_select_mode"):
            self.btn_sam_select_mode.setText("接受候选并微调")
        if hasattr(self, "btn_save_staging_areas"):
            self.btn_save_staging_areas.setText("拒绝当前 proposal")
        if hasattr(self, "btn_ignore_preannotation"):
            self.btn_ignore_preannotation.setText("忽略当前 proposal")
        if hasattr(self, "btn_export_preannotation_records"):
            self.btn_export_preannotation_records.setText("导出预标注调整记录")
        if hasattr(self, "btn_export_weights"):
            self.btn_export_weights.setText("导出当前权重")
        if hasattr(self, "btn_select_weights"):
            self.btn_select_weights.setText("选择权重")
        if hasattr(self, "properties_panel"):
            self.properties_panel.restore_button_texts()
        self.update_timing_panel()

    def _get_timing_state(self):
        if not self.current_image_state:
            self.current_image_state = make_image_state(self.current_image_path or "")
        timing_state = normalize_annotation_timing_state(self.current_image_state.get("annotation_timing"))
        self.current_image_state["annotation_timing"] = timing_state
        return timing_state

    def _get_live_timing_totals(self):
        timing_state = self._get_timing_state()
        total_seconds = float(timing_state.get("total_seconds", 0.0))
        manual_seconds = float(timing_state.get("manual_seconds", 0.0))
        sam_seconds = float(timing_state.get("sam_seconds", 0.0))

        if self._timer_running and self._timer_started_monotonic is not None:
            delta = max(0.0, monotonic() - self._timer_started_monotonic)
            total_seconds += delta
            if timing_state.get("active_mode") == "sam":
                sam_seconds += delta
            else:
                manual_seconds += delta

        return total_seconds, manual_seconds, sam_seconds

    def _commit_annotation_timer_segment(self, reason="pause"):
        if not self._timer_running or self._timer_started_monotonic is None:
            return 0.0

        elapsed = max(0.0, monotonic() - self._timer_started_monotonic)
        timing_state = self._get_timing_state()
        timing_state["total_seconds"] = float(timing_state.get("total_seconds", 0.0)) + elapsed
        timing_state["manual_seconds"] = float(timing_state.get("manual_seconds", 0.0)) + elapsed
        timing_state.setdefault("sessions", []).append(
            {
                "mode": "annotation",
                "started_at_monotonic": None,
                "ended_reason": reason,
                "elapsed_seconds": round(elapsed, 3),
                "recorded_at": self.current_image_state.get("last_modified_at"),
            }
        )
        self._timer_running = False
        self._timer_started_monotonic = None
        return elapsed

    def start_annotation_timer(self):
        if not self.current_image_path:
            return
        if self.current_image_state.get("annotation_completed", False):
            return
        self._get_timing_state()
        if self._timer_running:
            return
        self._timer_running = True
        self._timer_started_monotonic = monotonic()
        if not self.annotation_timer.isActive():
            self.annotation_timer.start()
        self.update_timing_panel()

    def pause_annotation_timer(self):
        self._commit_annotation_timer_segment(reason="pause")
        self.annotation_timer.stop()
        self.update_timing_panel()

    def update_timing_panel(self):
        if not hasattr(self, "label_timing_total"):
            return

        if not self.current_image_path:
            if hasattr(self, "label_timing_status"):
                self.label_timing_status.setText("状态: 未加载图片")
            self.label_timing_total.setText("总耗时: 00:00:00")
            return

        timing_state = self._get_timing_state()
        total_seconds, _, _ = self._get_live_timing_totals()
        if self.current_image_state.get("annotation_completed"):
            running_label = "已完成"
        else:
            running_label = "计时中" if self._timer_running else "已暂停"
        self.label_timing_status.setText(f"状态: {running_label}")
        self.label_timing_total.setText(f"总耗时: {format_elapsed_seconds(total_seconds)}")

        if hasattr(self, "btn_timer_pause"):
            self.btn_timer_pause.setEnabled(self._timer_running)
        if hasattr(self, "update_status_bar"):
            self.update_status_bar()

    def restore_button_visuals(self):
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

